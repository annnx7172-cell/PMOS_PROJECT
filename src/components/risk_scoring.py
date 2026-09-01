"""Block 6 (Risk Scoring) as a runnable component.

Block 6 as committed derives four risk labels from rules over BMI, weight
gain, AMH and symptom counts, trains an XGBoost classifier per dimension with
cross-validated out-of-sample probabilities, and prints its results — but its
"Save Risk Models" cell is empty, so none of that ever reaches
``artifacts/risk_model_*.pkl``. This component is that missing save step,
otherwise faithful to the notebook: same four rules, same feature drops, same
``cross_val_predict`` scoring, same final refit on all 541 patients.

The four labels are proxies, not observed outcomes:

* ``Metabolic_Risk``    — BMI > 25 or weight gain
* ``CVD_Risk``          — BMI > 27.5 and weight gain
* ``Reproductive_Risk`` — AMH > 4.5 or an irregular cycle
* ``Psych_Risk``        — at least 3 of 5 symptom flags present

``Reproductive_Risk`` is trained on a feature list that drops ``AMH(ng/mL)``
and ``Cycle(R/I)`` on top of the usual ``Marraige Status (Yrs)`` drop, because
both feed the rule that defines its own label — leaving them in would let the
model read its own answer. Its AUC (~0.70) is lower than the others *by
design*; do not "fix" it by adding them back.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import ARTIFACTS_DIR, load_object, save_object

RISK_TARGETS = ['Metabolic_Risk', 'CVD_Risk', 'Reproductive_Risk', 'Psych_Risk']
TARGET_COLUMN = 'PCOS (Y/N)'


@dataclass
class RiskScoringConfig:
    # Same non-destructive default as the other components — this writes
    # beside the shipped artifacts, never over them, unless overridden.
    output_dir: str = os.path.join(ARTIFACTS_DIR, 'retrained')
    random_state: int = 42
    cv_splits: int = 5


class RiskScoring:
    def __init__(self, config: RiskScoringConfig | None = None):
        self.config = config or RiskScoringConfig()

    def _assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Metabolic_Risk'] = (
            (df['BMI'] > 25) | (df['Weight gain(Y/N)'] == 1)
        ).astype(int)

        df['CVD_Risk'] = (
            (df['BMI'] > 27.5) & (df['Weight gain(Y/N)'] == 1)
        ).astype(int)

        df['Reproductive_Risk'] = (
            (df['AMH(ng/mL)'] > 4.5) | (df['Cycle(R/I)'] == 4)
        ).astype(int)

        df['Psych_Risk'] = (
            df['hair growth(Y/N)']
            + df['Skin darkening (Y/N)']
            + df['Weight gain(Y/N)']
            + df['Pimples(Y/N)']
            + df['Hair loss(Y/N)']
            >= 3
        ).astype(int)

        for target in RISK_TARGETS:
            logging.info(
                '%s: %d positive (%.1f%%)',
                target,
                df[target].sum(),
                df[target].mean() * 100,
            )
        return df

    def _feature_sets(self, final_features: list) -> dict:
        # 'Marraige Status (Yrs)' has no bearing on any of the four rules and
        # is dropped from all of them.
        risk_features = [f for f in final_features if f != 'Marraige Status (Yrs)']
        # Reproductive_Risk additionally drops the two features its own label
        # is defined from.
        repro_features = [
            f for f in risk_features if f not in ('AMH(ng/mL)', 'Cycle(R/I)')
        ]
        return {
            'Metabolic_Risk': risk_features,
            'CVD_Risk': risk_features,
            'Reproductive_Risk': repro_features,
            'Psych_Risk': risk_features,
        }, risk_features, repro_features

    def initiate_risk_scoring(
        self, clean_data_path: str, final_features_path: str
    ) -> dict:
        """Score all four risk dimensions and write models + a scores CSV.

        Returns the output directory, the per-dimension metrics and the two
        risk feature lists.
        """
        logging.info('Entered the risk scoring component')
        try:
            df = pd.read_csv(clean_data_path)
            df.columns = df.columns.str.strip()
            final_features = load_object(final_features_path)

            df = self._assign_labels(df)
            feature_sets, risk_features, repro_features = self._feature_sets(
                final_features
            )

            cv = StratifiedKFold(
                n_splits=self.config.cv_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )

            out = self.config.output_dir
            os.makedirs(out, exist_ok=True)

            metrics = {}
            for target in RISK_TARGETS:
                features = feature_sets[target]
                y = df[target]
                X = df[features].fillna(df[features].median())

                pos_weight = (y == 0).sum() / (y == 1).sum()
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    scale_pos_weight=pos_weight,
                    eval_metric='logloss',
                    verbosity=0,
                )

                # Out-of-sample probabilities for every patient — none of them
                # come from a fold the model was trained on.
                cv_probs = cross_val_predict(
                    model, X, y, cv=cv, method='predict_proba'
                )[:, 1]
                cv_preds = (cv_probs > 0.5).astype(int)

                metrics[target] = {
                    'auc': round(roc_auc_score(y, cv_probs), 4),
                    'f1': round(f1_score(y, cv_preds), 4),
                    'acc': round(accuracy_score(y, cv_preds), 4),
                }
                logging.info('%s: %s', target, metrics[target])

                df[f'{target}_prob'] = cv_probs
                df[f'{target}_label'] = pd.cut(
                    cv_probs, bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Moderate', 'High']
                )

                # Refit on all 541 patients for the model that ships — the CV
                # loop above is for honest metrics, not for the saved weights.
                model.fit(X, y)
                save_object(os.path.join(out, f'risk_model_{target}.pkl'), model)

            save_object(os.path.join(out, 'risk_features.pkl'), risk_features)
            save_object(os.path.join(out, 'repro_features.pkl'), repro_features)

            # Match the shipped CSV's column order: probs for all four, then
            # labels for all four.
            score_columns = (
                [TARGET_COLUMN]
                + [f'{t}_prob' for t in RISK_TARGETS]
                + [f'{t}_label' for t in RISK_TARGETS]
            )
            df[score_columns].to_csv(
                os.path.join(out, 'patient_risk_scores.csv'), index=False
            )
            logging.info('Risk scoring artifacts written to %s', out)

            return {
                'output_dir': out,
                'metrics': metrics,
                'risk_features': risk_features,
                'repro_features': repro_features,
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    result = RiskScoring().initiate_risk_scoring(
        os.path.join(ARTIFACTS_DIR, 'pmos_eda_clean.csv'),
        os.path.join(ARTIFACTS_DIR, 'final_features.pkl'),
    )
    for target, m in result['metrics'].items():
        print(f'{target:20s} AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}  Acc={m["acc"]:.4f}')
    print(f'\nWritten to: {result["output_dir"]}')
