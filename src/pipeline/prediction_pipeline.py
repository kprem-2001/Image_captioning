import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import torch


@dataclass
class PredictionPipelineConfig:
    model_path: str = "artifacts/model"
    model_name: str = "Salesforce/blip-image-captioning-base"


class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.processor = None
        self.model = None
        self.load_processor_model()

    def predict(self, input_image: Image) -> str:
        try:
            logging.info("Prediction Pipeline Started...")
            # Set the device (CPU or GPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Prepare inputs
            inputs = self.processor(images=input_image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values

            # Move model to the appropriate device
            self.model.to(device)

            # Generate predictions
            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)

            # Decode the predictions to text
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logging.info("Prediction complete")
            return caption

        except Exception as e:
            logging.exception("Error during prediction")
            raise CustomException(e, sys)

    def load_processor_model(self):
        try:
            logging.info("Loading processor and model...")
            if os.path.exists(self.prediction_pipeline_config.model_path):
                self.processor = AutoProcessor.from_pretrained(
                    self.prediction_pipeline_config.model_name
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.prediction_pipeline_config.model_path
                )
                logging.info("Processor and model loaded successfully")
            else:
                raise FileNotFoundError(
                    f"Model path does not exist: {self.prediction_pipeline_config.model_path}"
                )

        except Exception as e:
            logging.exception("Error while loading processor and model")
            raise CustomException(e, sys)
