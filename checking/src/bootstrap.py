import pandas as pd
from abc import ABC
from tqdm import tqdm
from typing import Union
from .utils import clear_dir, get_image_based_on_transformation
from .Imagenet_c_transformations import *
from .constant import *
from src.dataset import DatasetInfo
from PIL.ImageFile import ImageFile
from typing import Tuple
import torch
import random
import psutil
import gc
from pathlib2 import Path
from sewar import vifp

random.seed(100)
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

color_jitter_range = np.linspace(-0.5, 0.5, 1000)

class Bootstrapper(ABC):
    def __init__(self, num_sample_iter: int, sample_size: int, source: Union[str, pathlib2.Path],
                 destination: Union[str, pathlib2.Path], dataset_info: DatasetInfo, transformation_type: str,
                 threshold: float):
        self.num_sample_iter = num_sample_iter
        self.sample_size = sample_size
        self.source = pathlib2.Path(source)
        self.destination = pathlib2.Path(destination)
        self.data = None
        self.bootstrap_df = None

    def _prepare(self):
        raise NotImplementedError

    def run(self):
        bootstrap_df = self.save()
        Bootstrapper.check_output(bootstrap_df)

    @staticmethod
    def check_output(self, bootstrap_df: pd.DataFrame):
        assert bootstrap_df.columns


def bootstrap_transform(original_image: Union[ImageFile, np.ndarray], transformation: str) -> Tuple[np.ndarray, int]:
    if transformation == GAUSSIAN_NOISE:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = gaussian_noise(original_image, param_index)
    elif transformation == DEFOCUS_BLUR:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = defocus_blur(original_image, param_index)
    #elif transformation == FROST:
    elif FROST in transformation:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = frost(original_image, param_index)
    elif transformation == BRIGHTNESS:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = brightness(original_image, param_index)
    elif transformation == CONTRAST:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = contrast(original_image, param_index)
    elif transformation == JPEG_COMPRESSION:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = jpeg_compression(original_image, param_index)
        img2 = np.asarray(img2)
        # ============= different transformation types end =============
    else:
        raise ValueError("Invalid Transformation")
    return img2, param_index


def bootstrap(images_info_df: pd.DataFrame, num_sample_iter: int, sample_size: int,
              transformation_type: str, threshold: float, bootstrap_path) -> pd.DataFrame:
    """Run bootstrap and make the transformation decisions.
    Input dataset info dataframe should contain the following columns:
    - id
    - filename
    - path
    - width
    - height

    Output dataframe contains the following columns:
    - iteration_id
    - within_iter_id
    - image_id
    - transformation_type
    - transformation_parameter
    - original_image_path
    - target_image_path
    - vd_score

    :param images_info_df: DataFrame containing image info, contains columns: id, filename, path, width, height
    :type images_info_df: pd.DataFrame
    :param num_sample_iter: number of bootstrap iteration
    :type num_sample_iter: int
    :param sample_size: number of images sampled for each bootstrap iteration
    :type sample_size: int
    :param transformation_type: transformation type
    :type sample_size: str
    :param threshold:
    :type threshold float
    :param matlab_engine
    :return: bootstrap info dataframe
    :rtype: pd.DataFrame
    """
    logger.info("Run Bootstrap")
    clear_dir(bootstrap_path)

    progress_bar = tqdm(total=num_sample_iter * sample_size)
    bootstrap_decisions = []

    for i in range(1, num_sample_iter + 1):
        sample_df = images_info_df.sample(n=sample_size, replace=False)

        iter_path = bootstrap_path / f'iter{i}'
        iter_path.mkdir(parents=True, exist_ok=True)
        images_dir = iter_path / 'JPEGImages'
        annotations_dir = iter_path / 'Annotations'
        seg_class_dir = iter_path / 'SegmentationClass'
        seg_obj_dir = iter_path / 'SegmentationObject'
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        seg_class_dir.mkdir(parents=True, exist_ok=True)
        seg_obj_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)


        within_iter_count = 1
        image_ids_selected = set()
        for index, cur_row in sample_df.iterrows():
            if cur_row['id'] in image_ids_selected:
                continue
            image_path = cur_row['path']
            img = get_image_based_on_transformation(transformation_type, image_path)
            c = 0
            while True:
                try:
                    c += 1
                    img2, param_index = bootstrap_transform(img, transformation_type)
                    # try to transform image and test IQA
                    img_g = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
                    IQA_score = vifp(np.asarray(img_g).astype('uint8'), np.asarray(img2_g).astype('uint8'),  sigma_nsq=0.4)
                except:
                    c -= 1
                    new_row = images_info_df.sample(n=1)
                    for index, cur_row in new_row.iterrows():    
                        image_path = cur_row['path']
                        img = get_image_based_on_transformation(transformation_type, image_path)
                    continue

                gc.collect()

                if 1 - IQA_score < threshold:  # TODO: use 0.8
                    bootstrap_decisions.append({
                        'iteration_id': i,
                        'within_iter_id': within_iter_count,
                        'image_id': cur_row['id'],
                        'transformation_type': transformation_type,
                        'transformation_parameter': param_index,
                        'vd_score': 1 - IQA_score
                    })
                    image_ids_selected.add(cur_row['id'])

                    name = cur_row['id'] + '.jpg'
                    output_path = images_dir / name
                    im = Image.fromarray(img2.astype('uint8'))
                    im.save(str(output_path))
                    annotation_filename = cur_row['id'] + ".xml"
                    orig_anno_file =  Path(str(VOC_ROOT) +'/VOC2012/Annotations/'+annotation_filename)
                    os.symlink(orig_anno_file, annotations_dir / annotation_filename)

                    seg_filename = cur_row['id'] + ".png"
                    orig_seg_class_file =  Path(str(VOC_ROOT) +'/VOC2012/SegmentationClass/'+seg_filename)
                    orig_seg_obj_file =  Path(str(VOC_ROOT) +'/VOC2012/SegmentationObject/'+seg_filename)
                    os.symlink(orig_seg_class_file, seg_class_dir / seg_filename)
                    os.symlink(orig_seg_obj_file, seg_obj_dir / seg_filename)
                    break
            progress_bar.set_postfix_str(f"Memory Perc: {psutil.virtual_memory().percent}")
            
            within_iter_count += 1
            progress_bar.update(n=1)

        main_dir = iter_path / "ImageSets" / "Main"
        main_dir.mkdir(parents=True, exist_ok=True)
        with open(str(main_dir / 'val.txt'), 'w') as f:
            for image_id in image_ids_selected:
                f.write(image_id + "\n")
    return pd.DataFrame(data=bootstrap_decisions)
