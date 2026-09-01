"""Chains ingestion -> transformation -> training -> risk scoring.

By default everything lands in ``artifacts/retrained/`` and nothing already in
``artifacts/`` is touched. That default is deliberate: the models in
``artifacts/`` are what the deployed dashboard loads, and until this pipeline
existed several of them could not be regenerated from the notebooks as
committed.

    python -m src.pipeline.train_pipeline                # safe, writes to artifacts/retrained/
    python -m src.pipeline.train_pipeline --overwrite    # replaces artifacts/ in place

A full run reproduces the diagnosis half of the platform (splits, scaler,
feature list, the five classifiers) and the risk-scoring half (the four rule-
derived labels, their XGBoost models, ``risk_features``/``repro_features`` and
``patient_risk_scores.csv``). Clustering (Block 5), the SHAP explainer
(Block 7) and the recommendation engine (Block 8) still live only in the
notebooks, so ``--overwrite`` leaves those artifacts alone.
"""

import argparse
import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.risk_scoring import RiskScoring, RiskScoringConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import ARTIFACTS_DIR


def run_training_pipeline(output_dir: str | None = None) -> dict:
    """Run all four components in order and return a combined summary."""
    try:
        logging.info('Training pipeline started')

        clean_path = DataIngestion().initiate_data_ingestion()

        transform_config = DataTransformationConfig()
        if output_dir:
            transform_config.output_dir = output_dir
        transformation = DataTransformation(transform_config)
        transform_result = transformation.initiate_data_transformation(clean_path)

        trainer_config = ModelTrainerConfig(output_dir=transform_result['output_dir'])
        result = ModelTrainer(trainer_config).initiate_model_trainer(
            transform_result['output_dir']
        )
        result['final_features'] = transform_result['final_features']

        # Risk scoring works off the full cleaned dataset and the just-selected
        # feature list, not the train/test splits above.
        risk_config = RiskScoringConfig(output_dir=transform_result['output_dir'])
        final_features_path = os.path.join(
            transform_result['output_dir'], 'final_features.pkl'
        )
        risk_result = RiskScoring(risk_config).initiate_risk_scoring(
            clean_path, final_features_path
        )
        result['risk_metrics'] = risk_result['metrics']

        logging.info('Training pipeline finished; best model %s', result['best_model'])
        return result
    except Exception as e:
        raise CustomException(e, sys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='write straight to artifacts/, replacing the shipped pickles',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='write somewhere else entirely (overrides --overwrite)',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (ARTIFACTS_DIR if args.overwrite else None)
    if output_dir == ARTIFACTS_DIR:
        print('Writing to artifacts/ — the shipped pickles will be replaced.')

    result = run_training_pipeline(output_dir)

    print()
    print(result['leaderboard'].to_string(index=False))
    print()
    print(f'Features   : {len(result["final_features"])}')
    print(f'Best model : {result["best_model"]}')
    print(f'Written to : {result["output_dir"]}')
    print()
    for target, m in result['risk_metrics'].items():
        print(f'{target:20s} AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}  Acc={m["acc"]:.4f}')


if __name__ == '__main__':
    main()
