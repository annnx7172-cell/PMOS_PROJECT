"""Block 3 (ML Models) as a runnable component.

Fits Logistic Regression, Random Forest, XGBoost, SVM and a soft-voting
ensemble of the first three, scores them all, and writes each one plus a
``best_model.pkl``.

Every model is a ``Pipeline`` with its own ``StandardScaler`` in front, so the
saved object takes raw (unscaled) feature values. That is what lets the
dashboard hand a patient dict straight to ``predict_proba`` without touching
``scaler.pkl``.

Selection is on PMOS+ recall rather than accuracy: a missed positive costs more
than a false alarm here, and the shipped model has historically been the
Logistic Regression on that basis.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import ARTIFACTS_DIR, evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    # Same non-destructive default as DataTransformationConfig.
    output_dir: str = os.path.join(ARTIFACTS_DIR, 'retrained')
    random_state: int = 42


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    def _build_models(self) -> dict:
        seed = self.config.random_state
        return {
            'model_lr': Pipeline(
                [
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(random_state=seed, max_iter=1000)),
                ]
            ),
            'model_rf': Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'model',
                        RandomForestClassifier(
                            n_estimators=100, random_state=seed, n_jobs=-1
                        ),
                    ),
                ]
            ),
            'model_xgb': Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'model',
                        XGBClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=4,
                            random_state=seed,
                            eval_metric='logloss',
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            'model_svm': Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'model',
                        SVC(kernel='rbf', probability=True, random_state=seed, C=1.0),
                    ),
                ]
            ),
        }

    def initiate_model_trainer(self, data_dir: str) -> dict:
        """Train every model against the splits in ``data_dir``.

        Returns the leaderboard and the name of the model chosen as best.
        """
        logging.info('Entered the model trainer component')
        try:
            X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
            y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
            y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()

            out = self.config.output_dir
            os.makedirs(out, exist_ok=True)

            models = self._build_models()
            fitted, rows = {}, []

            for name, model in models.items():
                model.fit(X_train, y_train)
                fitted[name] = model

                scores = evaluate_model(name, model, X_train, y_train, X_test, y_test)
                scores['Recall_PMOS+'] = round(
                    recall_score(y_test, model.predict(X_test)), 4
                )
                rows.append(scores)

                save_object(os.path.join(out, f'{name}.pkl'), model)
                logging.info('%s: %s', name, scores)

            # Soft voting over LR + RF + XGB. SVM is left out — it is the
            # weakest of the four and drags the vote down.
            ensemble = VotingClassifier(
                estimators=[
                    ('lr', fitted['model_lr']),
                    ('rf', fitted['model_rf']),
                    ('xgb', fitted['model_xgb']),
                ],
                voting='soft',
            )
            ensemble.fit(X_train, y_train)
            fitted['model_ensemble'] = ensemble

            scores = evaluate_model(
                'model_ensemble', ensemble, X_train, y_train, X_test, y_test
            )
            scores['Recall_PMOS+'] = round(
                recall_score(y_test, ensemble.predict(X_test)), 4
            )
            rows.append(scores)
            save_object(os.path.join(out, 'model_ensemble.pkl'), ensemble)
            logging.info('model_ensemble: %s', scores)

            leaderboard = pd.DataFrame(rows).sort_values(
                ['Recall_PMOS+', 'ROC-AUC'], ascending=False
            )
            leaderboard.to_csv(os.path.join(out, 'model_report.csv'), index=False)

            best_name = leaderboard.iloc[0]['Model']
            save_object(os.path.join(out, 'best_model.pkl'), fitted[best_name])
            logging.info(
                'Best model on PMOS+ recall: %s (recall %.4f, ROC-AUC %.4f)',
                best_name,
                leaderboard.iloc[0]['Recall_PMOS+'],
                leaderboard.iloc[0]['ROC-AUC'],
            )

            return {
                'output_dir': out,
                'best_model': best_name,
                'leaderboard': leaderboard,
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    result = ModelTrainer().initiate_model_trainer(
        os.path.join(ARTIFACTS_DIR, 'retrained')
    )
    print(result['leaderboard'].to_string(index=False))
    print(f'\nBest: {result["best_model"]} -> {result["output_dir"]}')
