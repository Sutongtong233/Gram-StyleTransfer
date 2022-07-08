from matplotlib import artist
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from torch import optim
from model import *
from PIL import Image

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
    is <= 400 pixels in the x-y dims.'''
    image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image




if __name__ == "__main__":

    # loss weight hyperparams
    content_weight = 1
    style_weight = 1
    content = load_image('/home/tongtong/python_project/CV/Gram-StyleTransfer/images/content/1.jpg')
    print(content.shape)