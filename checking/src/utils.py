import os
import shutil

import pathlib2
import numpy as np
from typing import Union
from PIL.ImageFile import ImageFile
from .Imagenet_c_transformations import *
from .constant import GAUSSIAN_NOISE, DEFOCUS_BLUR, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, \
    TRANSFORMATION_LEVEL, COLOR_JITTER, RGB


def clear_dir(path: Union[str, pathlib2.Path]):
    path = pathlib2.Path(path) if isinstance(path, str) else path
    if path.exists():
        for p in path.iterdir():
            os.remove(str(p)) if p.is_file() else shutil.rmtree(str(p))

def get_image_based_on_transformation(transformation: str, image_path: str) -> Union[ImageFile, np.ndarray]:
    if transformation in [GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, RGB, COLOR_JITTER,
                          DEFOCUS_BLUR]:
        img = Image.open(image_path)
    else:
        img = np.asarray(cv2.imread(image_path), dtype=np.float32)
    return img
