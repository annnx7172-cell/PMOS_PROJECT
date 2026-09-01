"""Block 2 (Preprocessing) as a runnable component.

Takes the cleaned CSV from data ingestion and produces the train/test splits,
the fitted ``StandardScaler`` and the 13-name ``final_features`` list that the
diagnosis model and SHAP explainer are built on.

Selection chain, in the order Block 2 applies it:

1. strip whitespace from column names, coerce object columns to numeric
2. impute (mode for binary columns, median otherwise)
3. drop identifier and near-constant columns
4. LassoCV on standardised features — keep the non-zero coefficients
5. drop three high-VIF / clinically weak names from that survivor set
6. stratified 80/20 split, then SMOTETomek on the training half only
7. fit the scaler on the balanced training half

Chi-square, ANOVA and mutual information are computed as well, but only as a
diagnostic report — Block 2 never used them to drive the final cut, and
reproducing ``final_features.pkl`` depends on that.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.components.data_ingestion import TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging
from src.utils import ARTIFACTS_DIR, save_object


@dataclass
class DataTransformationConfig:
    # Default output is a sibling folder, NOT artifacts/ itself — the committed
    # pickles there are the ones the deployed dashboard loads, and some of them
    # cannot be regenerated from the notebooks as committed. Pass
    # output_dir=ARTIFACTS_DIR explicitly (or run the pipeline with --overwrite)
    # when you really do intend to replace them.
    output_dir: str = os.path.join(ARTIFACTS_DIR, 'retrained')
    test_size: float = 0.2
    random_state: int = 42
    drop_columns: list = field(
        default_factory=lambda: [
            'Sl. No',
            'Patient File No.',
            'Unnamed: 44',
            'Blood Group',
            'Pregnant(Y/N)',
        ]
    )
    # Dropped after LASSO: high VIF against the rest of the survivor set, and
    # weak on clinical grounds.
    vif_drop_columns: list = field(
        default_factory=lambda: ['Pulse rate(bpm)', 'Age (yrs)', 'Cycle length(days)']
    )


class DataTransformation:
    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()

    # ---------------------------------------------------------------- cleaning

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # The source Excel has inconsistent spacing; strip the ends but leave
        # the internal spacing alone. 'I   beta-HCG(mIU/mL)' keeps its three
        # internal spaces, because the feature lists and the dashboard's
        # patient dict both spell it that way.
        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        missing = df.isnull().sum()
        for col in missing[missing > 0].index:
            fill = df[col].mode()[0] if df[col].nunique() <= 2 else df[col].median()
            df[col] = df[col].fillna(fill)

        present = [c for c in self.config.drop_columns if c in df.columns]
        df = df.drop(columns=present)
        logging.info('Dropped %d identifier/constant columns: %s', len(present), present)
        return df

    # ------------------------------------------------------- selection reports

    def _selection_report(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Chi-square, ANOVA and MI scores. Diagnostic only, not used to cut."""
        binary_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
        continuous_cols = [
            c for c in X.columns if c not in binary_cols and X[c].nunique() > 5
        ]
        X_filled = X.fillna(X.median())

        chi2_scores, chi2_pvals = chi2(X[binary_cols].fillna(0), y)
        f_scores, f_pvals = f_classif(X_filled[continuous_cols], y)
        mi_scores = mutual_info_classif(
            X_filled, y, random_state=self.config.random_state
        )

        return {
            'chi2': pd.DataFrame(
                {'Feature': binary_cols, 'Chi2': chi2_scores, 'p': chi2_pvals}
            ).sort_values('Chi2', ascending=False),
            'anova': pd.DataFrame(
                {'Feature': continuous_cols, 'F': f_scores, 'p': f_pvals}
            ).sort_values('F', ascending=False),
            'mutual_info': pd.DataFrame(
                {'Feature': X.columns, 'MI': mi_scores}
            ).sort_values('MI', ascending=False),
        }

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        X_filled = X.fillna(X.median())

        # LASSO is scale sensitive, so standardise first. This scaler is
        # throwaway — the one that ships is fitted later on the balanced split.
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X_filled), columns=X.columns
        )
        lasso = LassoCV(cv=5, random_state=self.config.random_state, max_iter=10000)
        lasso.fit(X_scaled, y)
        logging.info('LassoCV best alpha: %.4f', lasso.alpha_)

        coef = pd.Series(lasso.coef_, index=X.columns)
        survivors = coef[coef != 0].sort_values(key=abs, ascending=False).index.tolist()
        logging.info('LASSO kept %d of %d features', len(survivors), X.shape[1])

        X_lasso = X_filled[survivors]
        vif = pd.DataFrame(
            {
                'Feature': survivors,
                'VIF': [
                    variance_inflation_factor(X_lasso.values, i)
                    for i in range(X_lasso.shape[1])
                ],
            }
        ).sort_values('VIF', ascending=False)
        logging.info('VIF above 10: %s', vif[vif['VIF'] > 10]['Feature'].tolist())

        final_features = [
            f for f in survivors if f not in self.config.vif_drop_columns
        ]
        logging.info('Final feature set (%d): %s', len(final_features), final_features)
        return final_features

    # ------------------------------------------------------------------- entry

    def initiate_data_transformation(self, clean_data_path: str) -> dict:
        """Run the full chain and write the splits, scaler and feature list.

        Returns a dict of the paths written plus the selected feature list.
        """
        logging.info('Entered the data transformation component')
        try:
            df = pd.read_csv(clean_data_path)
            df = self._clean(df)

            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]

            report = self._selection_report(X, y)
            logging.info('Top MI features:\n%s', report['mutual_info'].head(10))

            final_features = self._select_features(X, y)
            X_final = X.fillna(X.median())[final_features]

            X_train, X_test, y_train, y_test = train_test_split(
                X_final,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
            logging.info(
                'Split: %d train / %d test', X_train.shape[0], X_test.shape[0]
            )

            # Resample the training half only — the test set must stay a real
            # sample of the population.
            X_train, y_train = SMOTETomek(
                random_state=self.config.random_state
            ).fit_resample(X_train, y_train)
            logging.info(
                'After SMOTETomek: PMOS- %d / PMOS+ %d',
                int((y_train == 0).sum()),
                int((y_train == 1).sum()),
            )

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), columns=final_features
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), columns=final_features
            )

            out = self.config.output_dir
            os.makedirs(out, exist_ok=True)

            X_train.to_csv(os.path.join(out, 'X_train.csv'), index=False)
            X_test.to_csv(os.path.join(out, 'X_test.csv'), index=False)
            y_train.to_csv(os.path.join(out, 'y_train.csv'), index=False)
            y_test.to_csv(os.path.join(out, 'y_test.csv'), index=False)
            X_train_scaled.to_csv(os.path.join(out, 'X_train_scaled.csv'), index=False)
            X_test_scaled.to_csv(os.path.join(out, 'X_test_scaled.csv'), index=False)

            save_object(os.path.join(out, 'scaler.pkl'), scaler)
            save_object(os.path.join(out, 'final_features.pkl'), final_features)
            logging.info('Transformation artifacts written to %s', out)

            return {
                'output_dir': out,
                'final_features': final_features,
                'selection_report': report,
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    result = DataTransformation().initiate_data_transformation(
        DataIngestion().initiate_data_ingestion()
    )
    print(f'{len(result["final_features"])} features -> {result["output_dir"]}')
