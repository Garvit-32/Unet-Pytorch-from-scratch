import logging
import argparse
import os 
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

from model.unet import UNet
from utils.data_vis import plot_img_and_mask

from utils.dataset import CustomDataset
def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(CustomDataset.preprocess(full_img))

    # img = torch.from_numpy(full_img.transpose((2, 0, 1)))
    # img = np.array(img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_output_filenames():
    in_files = [r'data/train/butterfly (4).png']
    out_files = []
    
    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    in_files = [r'data/test/butterfly (78).jpg']  
    out_files = ['predict2.png']

    net = UNet(n_channels=3, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(r"checkpoints/CP_epoch200.pth", map_location=device))


    for i, file in enumerate(in_files):

        img = Image.open(file)

        # img = cv2.imread(fn)
        # img = cv2.resize(img,(112,112))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=0.5,
                           device=device)

        
        out_fn = out_files[i]
        result = mask_to_image(mask)
        result.save(out_files[i])
        plot_img_and_mask(img, mask)
