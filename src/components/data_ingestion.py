import gdown
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
from zipfile import ZipFile

@dataclass
class DataIngestionConfig:
    data_path: str 
    download_file_path: str  
    gdrive_url: str = "https://drive.google.com/uc?id=1YKkLcS1CZ8xwK00POP_iSF1BE8TorWlK" 

class DataIngestion:
    def __init__(self, data_path: str, download_file_path: str):
        self.ingestion_config = DataIngestionConfig(
            data_path=data_path,
            download_file_path=download_file_path
        )
    
    def download(self):
        try:
            logging.info("Data is downloading")
            gdown.download(
                self.ingestion_config.gdrive_url,
                self.ingestion_config.download_file_path
            )
            logging.info("Download finished")
            return self.ingestion_config.download_file_path
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def unzip(self):
        try:
            logging.info("Extraction started")
            with ZipFile(self.ingestion_config.download_file_path, "r") as file:
                file.extractall(self.ingestion_config.data_path)
            logging.info("Extraction done !!!")
            return self.ingestion_config.data_path    
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)