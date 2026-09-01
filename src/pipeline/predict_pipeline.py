"""Serving-side inference. No Streamlit, no Flask — plain Python in, plain
Python out, so the same pipeline can back any front end.

``PatientData`` turns the values a form collects into the feature dict the
models expect, and ``PredictPipeline`` loads the artifacts once and answers
four questions about that patient: PMOS probability, the four risk bands, the
SHAP contributions behind the diagnosis, and (optionally) an ultrasound class.
"""

import io
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import artifact_path, load_object, risk_band

RISK_TARGETS = ['Metabolic_Risk', 'CVD_Risk', 'Reproductive_Risk', 'Psych_Risk']

# 'Marraige Status (Yrs)' is a genuine model feature but the UI has no sensible
# question for it and sends a fixed placeholder, so its SHAP bar would be
# meaningless. Hide it from the chart rather than dropping it from the model.
HIDE_FROM_SHAP = ['Marraige Status (Yrs)']

CNN_CLASSES = ['Dominant Follicle', 'Normal', 'PCO']
CNN_IMAGE_SIZE = (224, 224)


@dataclass
class PatientData:
    """One patient's inputs, as collected by the form.

    ``to_dataframe`` is where the tidy attribute names are mapped back onto the
    column names the models were trained with. Those strings come from the
    source Excel and are inconsistent on purpose: ``'I   beta-HCG(mIU/mL)'`` has
    three internal spaces and ``'Marraige Status (Yrs)'`` is misspelled. They
    have to match exactly or the ``df[features]`` lookup raises a KeyError.
    """

    follicle_r: int = 5
    follicle_l: int = 5
    lh: float = 2.5
    amh: float = 2.0
    hcg: float = 2.0
    prg: float = 0.3
    cycle: int = 2  # 2 = regular, 4 = irregular
    weight_gain: int = 0
    hair_growth: int = 0
    skin_darkening: int = 0
    pimples: int = 0
    hair_loss: int = 0
    fast_food: int = 0
    marriage_years: int = 5  # placeholder; see HIDE_FROM_SHAP

    def to_dict(self) -> dict:
        return {
            'Follicle No. (R)': self.follicle_r,
            'Follicle No. (L)': self.follicle_l,
            'LH(mIU/mL)': self.lh,
            'AMH(ng/mL)': self.amh,
            'I   beta-HCG(mIU/mL)': self.hcg,
            'PRG(ng/mL)': self.prg,
            'Cycle(R/I)': self.cycle,
            'Weight gain(Y/N)': int(self.weight_gain),
            'hair growth(Y/N)': int(self.hair_growth),
            'Skin darkening (Y/N)': int(self.skin_darkening),
            'Pimples(Y/N)': int(self.pimples),
            'Hair loss(Y/N)': int(self.hair_loss),
            'Fast food (Y/N)': int(self.fast_food),
            'Marraige Status (Yrs)': self.marriage_years,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class PredictPipeline:
    """Loads the artifacts once, then answers questions about a patient.

    Construction reads from disk, so build one instance and reuse it. Under
    Streamlit that means wrapping it in ``@st.cache_resource``.
    """

    def __init__(self, artifacts_dir: str | None = None):
        self.artifacts_dir = artifacts_dir
        try:
            self.pmos_model = load_object(self._path('best_model.pkl'))
            self.final_features = load_object(self._path('final_features.pkl'))
            self.risk_features = load_object(self._path('risk_features.pkl'))
            self.repro_features = load_object(self._path('repro_features.pkl'))
            self.risk_models = {
                t: load_object(self._path(f'risk_model_{t}.pkl')) for t in RISK_TARGETS
            }
            self.xgb_model = load_object(self._path('xgb_shap_model.pkl'))
            self.explainer = load_object(self._path('xgb_explainer.pkl'))
            logging.info('Prediction artifacts loaded from %s', self._path(''))
        except Exception as e:
            raise CustomException(e, sys)

        # Reproductive_Risk uses a shorter feature list. AMH and Cycle(R/I) are
        # left out because they feed the rule that defines the label, so
        # including them would let the model read its own answer. Its AUC is
        # lower than the others by design — do not "fix" it by adding them back.
        self.feature_sets = {
            'Metabolic_Risk': self.risk_features,
            'CVD_Risk': self.risk_features,
            'Reproductive_Risk': self.repro_features,
            'Psych_Risk': self.risk_features,
        }

        self._cnn = None
        self._cnn_loaded = False

    def _path(self, name: str) -> str:
        if self.artifacts_dir:
            return os.path.join(self.artifacts_dir, name)
        return artifact_path(name)

    # ------------------------------------------------------------- diagnosis

    def predict_pmos(self, features: dict) -> float:
        """Probability that this patient is PMOS positive."""
        try:
            X = pd.DataFrame([features])[self.final_features].fillna(0)
            return float(self.pmos_model.predict_proba(X)[0][1])
        except Exception as e:
            raise CustomException(e, sys)

    # ----------------------------------------------------------------- risks

    def predict_risks(self, features: dict) -> dict:
        """Probability and band for each of the four risk dimensions.

        The labels these models learned are rule-derived proxies built in
        Block 6, not observed outcomes. Treat the output as a stratification
        aid, not a diagnosis.
        """
        try:
            results = {}
            for target, feats in self.feature_sets.items():
                X = pd.DataFrame([features])[feats].fillna(0)
                prob = float(self.risk_models[target].predict_proba(X)[0][1])
                label, color = risk_band(prob)
                results[target] = {'prob': prob, 'label': label, 'color': color}
            return results
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ XAI

    def shap_contributions(self, features: dict, hide: list | None = None):
        """Per-feature SHAP values for this prediction, as a tidy frame.

        Sorted by absolute contribution ascending, which is the order a
        horizontal bar chart wants. Positive values push toward PMOS positive.
        """
        try:
            hide = HIDE_FROM_SHAP if hide is None else hide
            X = pd.DataFrame([features])[self.final_features].fillna(0)
            values = self.explainer.shap_values(X)[0]

            df = pd.DataFrame(
                {
                    'Feature': self.final_features,
                    'SHAP Value': values,
                    'Patient Val': X.iloc[0].values,
                }
            )
            df = df[~df['Feature'].isin(hide)]
            return df.sort_values('SHAP Value', key=abs, ascending=True)
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------ ultrasound

    @property
    def cnn(self):
        """The ultrasound CNN, or ``None`` when TensorFlow is unavailable.

        Loaded lazily: TensorFlow is a heavy import and most sessions never
        upload an image. It is also an optional dependency, so a deployment
        without it degrades to "no classification" rather than failing.
        """
        if not self._cnn_loaded:
            self._cnn_loaded = True
            try:
                import tensorflow as tf

                self._cnn = tf.keras.models.load_model(
                    self._path('pmos_cnn_final.keras')
                )
                logging.info('Ultrasound CNN loaded')
            except Exception as e:
                logging.warning('Ultrasound CNN unavailable: %s', e)
                self._cnn = None
        return self._cnn

    def predict_ultrasound(self, image_bytes: bytes):
        """Classify an ovarian ultrasound. Returns ``(class, confidence)``.

        Returns ``(None, None)`` if the CNN could not be loaded or the bytes
        are not a readable image — the caller should treat that as "no result"
        rather than an error.
        """
        model = self.cnn
        if model is None:
            return None, None
        try:
            from PIL import Image

            img = (
                Image.open(io.BytesIO(image_bytes))
                .convert('RGB')
                .resize(CNN_IMAGE_SIZE)
            )
            arr = np.expand_dims(np.array(img) / 255.0, 0)
            probs = model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            return CNN_CLASSES[idx], float(probs[idx])
        except Exception as e:
            logging.warning('Ultrasound prediction failed: %s', e)
            return None, None


if __name__ == '__main__':
    pipeline = PredictPipeline()
    patient = PatientData(follicle_r=15, follicle_l=14, amh=8.0, weight_gain=1)
    features = patient.to_dict()

    print(f'PMOS probability: {pipeline.predict_pmos(features) * 100:.1f}%')
    for target, info in pipeline.predict_risks(features).items():
        print(f'  {target:20s} {info["prob"] * 100:5.1f}%  {info["label"]}')
