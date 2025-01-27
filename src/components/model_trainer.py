import os
import sys
from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    data_path: str
    model_path: str
    model_name: str = "Salesforce/blip-image-captioning-base"


class ModelTrainer:
    def __init__(self, data_path: str, model_path: str):
        self.model_trainer_config = ModelTrainerConfig(data_path=data_path, model_path=model_path)
        self.dataset = None
        self.processor = None
        self.model = None

    def load_dataset(self):
        try:
            logging.info("Loading Datasets ...")
            image_files = []
            captions = []
            
            # Open and read the captions.txt file
            with open(os.path.join(self.model_trainer_config.data_path, "captions.txt"), "r") as f:
                next(f)  # Skip header line
                line_count = 0  # To track number of lines processed
                current_image = None
                image_captions = []
                
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    line_count += 1
                    try:
                        img_name, caption = line.split(",", 1)  # Split only on the first comma
                        img_path = os.path.join(self.model_trainer_config.data_path, "Images", img_name.strip())

                        if current_image and current_image != img_name:  # If the image name changes, save the previous image's captions
                            image_files.append(current_image)
                            captions.append(image_captions)
                            image_captions = []  # Reset for the new image

                        # Append caption for the current image
                        current_image = img_name.strip()
                        image_captions.append(caption.strip())

                        # Check if image exists at the path
                        if not os.path.exists(img_path):
                            logging.warning(f"Image not found: {img_path}")

                    except ValueError:
                        logging.warning(f"Skipping malformed line: {line} (Line {line_count})")
                        continue

                # Append last image captions
                if current_image:
                    image_files.append(current_image)
                    captions.append(image_captions)

            # Log the number of images and captions loaded
            logging.info(f"Loaded {len(image_files)} images and {len(captions)} captions after processing {line_count} lines.")

            # Ensure the dataset isn't empty
            if len(image_files) == 0 or len(captions) == 0:
                raise ValueError("No valid images or captions found.")

            # Create the dataset dictionary
            dataset_dict = {
                "image": image_files,
                "text": captions
            }

            # Convert the dictionary into a Hugging Face Dataset
            self.dataset = Dataset.from_dict(dataset_dict)

            # Initialize the processor for the model
            self.processor = AutoProcessor.from_pretrained(self.model_trainer_config.model_name)

            # Wrap the dataset with our custom Dataset class
            self.dataset = ImageCaptioningDataset(self.dataset, self.processor, self.model_trainer_config.data_path)

            logging.info(f"Dataset loaded successfully with {len(image_files)} images.")

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def train_model(self):
        try:
            logging.info("Model training started")
            train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=4)

            model = BlipForConditionalGeneration.from_pretrained(self.model_trainer_config.model_name)

            # Add gradient accumulation and learning rate scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            num_epochs = 5  # Increase epochs for better training
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0

                for idx, batch in enumerate(train_dataloader):
                    input_ids = batch.pop("input_ids").to(device)
                    pixel_values = batch.pop("pixel_values").to(device)

                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids
                    )

                    loss = outputs.loss
                    total_loss += loss.item()

                    if idx % 100 == 0:
                        logging.info(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item():.4f}")

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss = total_loss / len(train_dataloader)
                scheduler.step(avg_loss)
                logging.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

            # Save the trained model
            model.save_pretrained(self.model_trainer_config.model_path)
            logging.info("Training Complete!")

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


class ImageCaptioningDataset(TorchDataset):
    def __init__(self, dataset, processor, data_path):
        self.dataset = dataset
        self.processor = processor
        self.data_path = data_path  # Store the data_path here

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_path = os.path.join(self.data_path, "Images", item['image'])  # Correct the path usage here
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Error loading image {item['image']}: {e}")
            return {}

        # Get the caption (we will use the first caption if there are multiple)
        caption = item['text'][0] if isinstance(item['text'], list) else item['text']
        
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=128,  # Add max length for consistency
            truncation=True,  # Enable truncation
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


