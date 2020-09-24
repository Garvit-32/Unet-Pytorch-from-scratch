import os
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .transform import get_transform


class CustomDataset(Dataset):

    def __init__(self, df, img_dir, mask_dir, phase):

        self.fname = df['0'].values.tolist()
        self.fname1 = df['1'].values.tolist()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.transform = get_transform(phase)
        # self.ids = [path for path in os.listdir(self.img_dir)]

    def __len__(self):
        # return len(self.ids)
        return len(self.fname)

    # @classmethod
    # def preprocess(cls, pil_img):
    #     pil_image = pil_img.resize((128, 128))

    #     img_nd = np.array(pil_img)

    #     # HWC to CHW
    #     img_trans = img_nd.transpose((2, 0, 1))
    #     if img_trans.max() > 1:
    #         img_trans = img_trans/255

    #     return img_trans

    def __getitem__(self, idx):
        img_path = self.fname[idx]
        mask_path = self.fname1[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # img = Image.open(img_file)
        # mask = Image.open(mask_file)
        # img = self.preprocess(img)
        # mask = self.preprocess(mask)

        augmentation = self.transform(image=img, mask=mask)
        img_aug = augmentation['image']
        mask_aug = augmentation['mask']

        return {
            'image': img_aug,
            'mask': mask_aug
        }

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        # }
