import os
import numpy as np
import torch
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
# from albumentations.pytorch import ToTensor
# from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
import cv2

transform = transforms.Compose([transforms.ToPILImage(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),transforms.ToTensor()])

class CustomDataset(Dataset):

    def __init__(self, df, img_dir, mask_dir, phase):

        self.fname = df['0'].values.tolist()
        self.fname1 = df['1'].values.tolist()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.phase = phase
        # self.transform = get_transform(phase)
        self.transform = transform

    def __len__(self):
        return len(self.fname)

    @classmethod
    def preprocess(cls, img):

        img = img.resize((112,112))

        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img,axis=2)

        # # # HWC to CHW
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img/255

        return img

    def __getitem__(self, idx):
        img_path = self.fname[idx]
        mask_path = self.fname1[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_aug = self.preprocess(img)
        mask_aug = self.preprocess(mask)

        print(type(img_aug))

        # print(type(mask_aug))
        
        img_aug = self.transform(img_aug)
            
        # mask_aug = self.transform(mask)

        return {
            'image': img_aug,
            'mask': mask_aug,
            'image_path' : img_path,
            "mask_path" : mask_path,
        }