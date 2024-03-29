# Transformations obtained from https://github.com/hendrycks/robustness with slight modification
# to adapt to images with different sizes and random sampling

import cv2
import logging
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from io import BytesIO
from PIL import Image
from wand.image import Image as WandImage
import os
import sys
import random
import pathlib2
import wand.color as WandColor
from scipy import fftpack as fp
from shutil import copy as copy_file
from wand.api import library as wandlibrary
# from skimage.color import rgb2gray, rgb2grey
from scipy.ndimage.filters import gaussian_filter
from .constant import ROOT, DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# /////////////// Distortions ///////////////
TRANSFORMATION_LEVEL = 1000


def save_array(dest, arr):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(dest)


def gaussian_noise(x, i):
    # c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    c = np.linspace(0.08, 0.9, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c[i]), 0, 1) * 255, c[i]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x, i):
    radius = np.linspace(1, 10, TRANSFORMATION_LEVEL)
    alias_blur = np.linspace(0, 1, TRANSFORMATION_LEVEL)
    c = np.stack([radius, alias_blur], 1)

    x = np.array(x) / 255.
    kernel = disk(radius=c[i][0], alias_blur=c[i][1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255, c[i]


def frost(x, i):
    scale = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
    np.random.shuffle(scale)
    constant = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
    np.random.shuffle(constant)
    c = np.stack([scale, constant], 1)	 
    
    idx = np.random.randint(5)
    filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpeg', 'frost5.jpeg', 'frost6.jpeg'][idx]
    frost = Image.open(os.path.join(ROOT, 'frost-images', filename))

    # print(frost)
    x = np.asarray(x)
    h, w, ch = x.shape
    frost = np.asarray(frost.resize((w, h)))
    # randomly crop and convert to rgb
    frost = frost[..., [2, 1, 0]]
    return np.clip(c[i][0] * x + c[i][1] * frost, 0, 255), c[i]


def contrast(x, i):
    c = np.linspace(0.01, 0.9, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c[i] + means, 0, 1) * 255, c[i]


def brightness(x, i):
    c = np.linspace(0, 1, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c[i], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255, c[i]


def jpeg_compression(x, i) -> Image:
    c = list(range(1, (TRANSFORMATION_LEVEL + 1)))

    output = BytesIO()
    quality = int(c[i]/(TRANSFORMATION_LEVEL/100))
    x.save(output, 'JPEG', quality=quality)
    x = Image.open(output)

    return x, c[i]

