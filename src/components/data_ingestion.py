"""Block 1 (EDA) reduced to its data-producing steps.

Reads the raw Excel sheet, drops duplicates, coerces the columns the source
file stores as text, and writes the cleaned frame that data transformation
consumes. The plotting and inspection cells of Block 1 are deliberately left in
the notebook — this component only produces the artifact.
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import ARTIFACTS_DIR, NOTEBOOK_DIR

TARGET_COLUMN = 'PCOS (Y/N)'


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(
        NOTEBOOK_DIR, 'data', 'PCOS_data_without_infertility.xlsx'
    )
    sheet_name: str = 'Full_new'
    clean_data_path: str = os.path.join(ARTIFACTS_DIR, 'pmos_eda_clean.csv')
    # The source Excel stores AMH as text with stray non-numeric entries.
    numeric_coerce_columns: list = field(default_factory=lambda: ['AMH(ng/mL)'])


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> str:
        """Produce ``pmos_eda_clean.csv`` and return its path."""
        logging.info('Entered the data ingestion component')
        try:
            df = pd.read_excel(
                self.config.raw_data_path, sheet_name=self.config.sheet_name
            )
            logging.info(
                'Read raw dataset: %d rows x %d columns', df.shape[0], df.shape[1]
            )

            duplicates = int(df.duplicated().sum())
            if duplicates:
                df = df.drop_duplicates()
                logging.info('Dropped %d duplicate rows', duplicates)

            for col in self.config.numeric_coerce_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logging.info(
                        '%s coerced to numeric, %d values became NaN',
                        col,
                        int(df[col].isnull().sum()),
                    )

            os.makedirs(os.path.dirname(self.config.clean_data_path), exist_ok=True)
            df.to_csv(self.config.clean_data_path, index=False)
            logging.info('Cleaned data written to %s', self.config.clean_data_path)

            return self.config.clean_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    path = DataIngestion().initiate_data_ingestion()
    print(f'Cleaned data written to {path}')
