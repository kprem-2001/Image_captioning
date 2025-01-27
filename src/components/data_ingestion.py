import gdown
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys,os
from zipfile import ZipFile

@dataclass
class DataIngestionConfig:
    data_path: str 
    download_file_path: str  
    gdrive_url: str = "https://drive.google.com/uc?export=download&id=1bUU_C8F6YhPbwSp9v_1NWZz4lJ-jLgVJ" 

class DataIngestion:
    def __init__(self, data_path: str, download_file_path: str):
        self.ingestion_config = DataIngestionConfig(
            data_path=data_path,
            download_file_path=download_file_path
        )
    
    def download(self):
        try:
            if os.path.exists(self.ingestion_config.download_file_path):
                logging.info("File already exists. Skipping download.")
                return self.ingestion_config.download_file_path
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
            # Define the path for the 'data_new' directory
            data_new_path = os.path.join(self.ingestion_config.data_path, "data_new")
            
            # Create the 'data_new' directory if it doesn't exist
            os.makedirs(data_new_path, exist_ok=True)
            
            logging.info("Extraction started")
            with ZipFile(self.ingestion_config.download_file_path, "r") as file:
                # Extract all files into the 'data_new' directory
                file.extractall(data_new_path)
            
            logging.info("Extraction done !!!")
            return data_new_path  # Return the path to the 'data_new' directory
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)