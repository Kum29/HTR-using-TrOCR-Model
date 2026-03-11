import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch.nn.functional as F
import os
import time  # <-- import time

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

processor_large = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten", local_files_only=True)
model_large = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten", local_files_only=True).to(device)
model_large = torch.compile(model_large, mode="reduce-overhead")  # Compile the model for faster inference
processor_base = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", local_files_only=True)
model_base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", local_files_only=True,).to(device)
model_base = torch.compile(model_base, mode="reduce-overhead")  # Compile the model for faster inference
if device == "cuda":
    model_base = model_base.half() #floating from 32 to 16 bit
    model_large = model_large.half()

def load_lines_from_folder(folder_path):
    # Get all image files (common formats)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    file_list = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    # Load images
    images = []
    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path) #conver to numpy array
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: failed to read {img_path}")
    return images

folder_path = r"C:\Users\KUMKUM\cropped_lines"
lines = load_lines_from_folder(folder_path)
