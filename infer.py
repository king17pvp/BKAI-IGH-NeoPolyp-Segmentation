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

from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchinfo import summary
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='Polysegment Inference')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("./pretrained_models/best_model.pt").to(device)