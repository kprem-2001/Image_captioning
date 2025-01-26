import os
import sys
from dataclasses import dataclass
from src.logging import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion

@dataclass
class TrainingPipelineConfig:
    data_ingestion_path : str = os.path.join('artifacts')
    download_data_path : str = os.path.join('artifacts','data.zip')
    model_data_path : str = os.pathjoin(data_ingestion_path , 'data_new')
    