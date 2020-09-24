import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# train_data_dir = './data/train'
# ids = [path for path in os.listdir(train_data_dir)]
# print(ids)


# image = Image.open(r"data\train\s2 (14).jpg")
# image.show()
# image = image.resize((128, 128))
# image.show()


# def plot_img_and_mask(img, mask):
#     classes = mask.shape[2] if len(mask.shape) > 2 else 1
#     fig, ax = plt.subplots(1, classes + 1)
#     ax[0].set_title('Input image')
#     ax[0].imshow(img)
#     if classes > 1:
#         for i in range(classes):
#             ax[i+1].set_title(f'Output mask (class {i+1})')
#             ax[i+1].imshow(mask[:, :, i])
#     else:
#         ax[1].set_title(f'Output mask')
#         ax[1].imshow(mask)
#     plt.xticks([]), plt.yticks([])
#     plt.show()


# img = cv2.imread(r"data\train\butterfly (98).jpg")
# mask = cv2.imread(r"mask\train\butterfly (98).png")

# plot_img_and_mask(img, mask)


# train = pd.read_csv('train.csv')
# print(len(train))
# print(train)
# print(train.iloc[0, 0])
# print(train['1'].values.tolist())

mask = cv2.imread(r"mask\train\butterfly (34).png", cv2.IMREAD_GRAYSCALE)

cv2.imshow('image', mask)
cv2.waitKey(0)
