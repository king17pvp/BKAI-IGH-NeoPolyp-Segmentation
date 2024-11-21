import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
import argparse
import segmentation_models_pytorch as smp

from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./pretrained_models/best_model.pt", map_location=device)
COLOR_DICT= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}
TRANSFORMATION = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

parser = argparse.ArgumentParser(prog='Polysegment Inference')
parser.add_argument("-i", "--image_path", type=str, default="", help="The path to the image you want to infer")

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)   
def infer(image_path, model, device):
    global COLOR_DICT
    global TRANSFORMATION
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h= image.shape[0], image.shape[1]
    image = cv2.resize(image, (256, 256))
    tmp = TRANSFORMATION(image=image)
    transformed_image = tmp['image'].unsqueeze(0).to(device)
    with torch.inference_mode():
        output_mask = model(transformed_image).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (h, w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, COLOR_DICT)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("saved_segmentation.jpeg", mask_rgb)


args = parser.parse_args()
infer(args.image_path, model, device)