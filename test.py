import os
import numpy as np
from PIL import Image,ImageOps
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm



# pil = cv2.imread(r"data/train/butterfly (3).png")

# # pil = Image.open(r"mask/train/butterfly (4).png")
# # pil = ImageOps.grayscale(pil)
# # pil = np.asarray(pil)
# # print(pil)
# pil = np.array(pil)

# pil = pil.transpose((2, 0, 1))
# print(pil.shape)

# cv2.imshow('image',pil)

# cv2.waitKey(0)

# print(np.asarray((mask*255)))

# # cv2.imshow('image', mask)
# # cv2.waitKey(0)



#     img = cv2.imread(i)
#     if img.shape[0] == 174:
#         print(i)
#     else:
#         pass
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # i = i.replace('mask','mask1')

    # cv2.imwrite(i,img_gray)


# pil =  Image.open(r'data/train/butterfly7.jpg')
# pil =  Image.open(r'mask/train/butterfly7.png')
# pil = ImageOps.grayscale(pil)
# pil = np.array(pil)
# pil = np.expand_dims(pil,axis=2)
# print(pil.shape)

print('Hello')