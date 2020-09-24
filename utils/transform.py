import cv2
from albumentations.pytorch import ToTensor
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
import numpy as np
import random
import torch
from torchvision import transforms


def get_transform(phase):
    list_trans = []

    if phase == "train":
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(), ToTensor()])
    list_trans = Compose(list_trans)
    return list_trans
