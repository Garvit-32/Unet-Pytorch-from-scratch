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

        img = np.array(img).astype('uint8')

        if len(img.shape) == 2:
            img = np.expand_dims(img,axis=2)

        # # HWC to CHW
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img/255

        return img

    def __getitem__(self, idx):
        img_path = self.fname[idx]
        mask_path = self.fname1[idx]

        # img = cv2.imread(img_path)
        # img = cv2.resize(img,(112,112))
        # mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask,(112,112))
        # mask = cv2.imread(mask_path)

        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)

        img = Image.open(img_path)
        # img = ImageOps.grayscale(img)
        img = img.resize((112,112))
        # img = np.array(img)

        # print("img.shape",img.shape)

        mask = Image.open(mask_path)
        # mask = ImageOps.grayscale(mask)
        mask = mask.resize((112,112))
        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # mask = np.array(mask)


        img_aug = self.preprocess(img)
        mask_aug = self.preprocess(mask)


        # img = torch.from_numpy(img_aug).type(torch.FloatTensor)
        # mask = torch.from_numpy(mask_aug).type(torch.FloatTensor)
        
        # augmentation = self.transform(image=img,mask=mask)
        # img_aug = augmentation['image']
        # mask_aug = augmentation['mask']

        # img_aug = self.transform(img)
        # mask_aug = self.transform(mask)



        # print("img_aug",img_aug.shape)
        # print("mask_aug",mask_aug.shape)

        # print(img_path)
        # plt.imshow(img_aug.permute(1, 2, 0))
        # plt.show()

        # return {
        #     'image': img_aug,
        #     'mask': mask_aug,
        # }

        return {
            'image': torch.from_numpy(img_aug).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_aug).type(torch.FloatTensor),
            'image_path' : img_path,
            "mask_path" : mask_path,
        }
