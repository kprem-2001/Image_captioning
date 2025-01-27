import os
import sys
from src.logger import logging
from dataclasses import dataclass

from src.exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion


@dataclass
class TrainPipelineConfig:
    data_ingestion_path: str = os.path.join('artifacts')
    download_file_path: str = os.path.join('artifacts', 'data.zip')
    model_data_path: str = os.path.join(data_ingestion_path, 'data_new')  
    model_path: str = os.path.join('artifacts', 'model')


class TrainPipeline:
    def __init__(self):
        '''
        This class is responsible for the training of the models and storing them in a file system.
        Also downloading the data
        '''
        self.train_pipeline_config = TrainPipelineConfig()
        self.data_ingestion = DataIngestion(data_path=self.train_pipeline_config.data_ingestion_path,
                                            download_file_path=self.train_pipeline_config.download_file_path)
        self.model_trainer = ModelTrainer(data_path=self.train_pipeline_config.model_data_path,
                                            model_path=self.train_pipeline_config.model_path)
    
    def train(self):
        try:
            logging.info('Model Training Pipeline Started...')
            self.data_ingestion.download()
            self.data_ingestion.unzip()

            self.model_trainer.load_dataset()
            self.model_trainer.train_model()
            logging.info('Model training pipeline finished!!!')

        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)

