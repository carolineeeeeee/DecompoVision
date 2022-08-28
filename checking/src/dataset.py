import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from typing import Any, Tuple, Optional, Callable
from pathlib2 import Path
import xml.etree.ElementTree as ET
import os
from src.constant import *

class DatasetInfo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_info_df(self) -> pd.DataFrame:
        """Should have columns
        - id
        - filename
        - path
        - width
        - height

        :raises NotImplementedError: [description]
        :return: pandas DataFrame with very basic image info data
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

class PascalVOCDatasetInfo(DatasetInfo):
    def __init__(self, root: Path, image_set: str = "val"):
        super(PascalVOCDatasetInfo, self).__init__()
        self.image_root = VOC_ROOT / "VOC2012" / "JPEGImages"
        self.annotation_root = VOC_ROOT / "VOC2012" / "Annotations"
        self.image_set = image_set
        assert self.image_set in ["seg", "train", "val", "trainval"]
        self.image_list_file = VOC_ROOT / "VOC2012" / "ImageSets" / "Main" / f"{self.image_set}.txt"
        seg_gt_path = str(VOC_ROOT/ "VOC2012" / "SegmentationClass")
        with open(str(self.image_list_file), "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        self.image_filenames = [f"{id_}.jpg" for id_ in self.image_ids if os.path.exists(seg_gt_path + '/' + id_ + '.png')]

    def verify_root_validity(self) -> bool:
        """
        whether the image_root and annotation_root used to initialize this dataset info object is valid, or compatible
        image files is what we look at, and annotation files can be a super set of image files
        :return: is valid
        """
        image_filenames = os.listdir(str(self.image_root))
        # this is different from self.image_ids, this is what's on disk, self.image_ids depends on image_set
        image_ids = set([filename.split(".")[0] for filename in image_filenames])
        annotation_filenames = os.listdir(str(self.annotation_root))
        annotation_ids = set([filename.split(".")[0] for filename in annotation_filenames])
        return len(image_ids & annotation_ids & set(self.image_ids)) >= len(self.image_ids)

    def image_info_df(self) -> pd.DataFrame:
        """

        :return: image info dataframe
        :rtype: pd.DataFrame
        """
        image_ids = set([filename.split(".")[0] for filename in self.image_filenames])
        data = {"id": [], "width": [], "height": [], "filename": [], "path": []}
        for id_ in image_ids:
            tree = ET.parse(str(self.annotation_root / f"{id_}.xml"))
            root = tree.getroot()
            filename = f"{id_}.jpg"
            data["id"].append(id_)
            data["width"].append(int(root.findall("size/width")[0].text))
            data["height"].append(int(root.findall("size/height")[0].text))
            data["filename"].append(filename)
            data["path"].append(str(self.image_root / filename))
        return pd.DataFrame(data=data)


class CustomPascalVOCDataset(Dataset):
    def __init__(self, root: str, orig_root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, ):
        super(CustomPascalVOCDataset, self).__init__()
        self.dataset = VOCDetection(orig_root, year, image_set, download, transform, target_transform, transforms)
        self.images_dir = Path(root).absolute() / 'images'
        self.annotations_dir = Path(root).absolute() / 'annotations'
        assert self.images_dir.exists()
        assert self.annotations_dir.exists()
        self.image_names = os.listdir(str(self.images_dir))
        self.image_paths = [self.images_dir / filename for filename in self.image_names]
        self.image_ids = [filename.split(".")[0] for filename in self.image_names]
        self.i2filename = {i: os.path.basename(path) for i, path in enumerate(self.dataset.images)}
        self.filename2i = {v: k for k, v in self.i2filename.items()}

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        The algorithm to match our custom dataset images to the original pascal voc images is very simple
        The original dataset takes in an index, the index is based on all images
        so I have to map custom index to original index

        1. Map custom index to the filename of selected image
        2. Map filename to original index
        """
        filename = self.image_names[index]
        original_index = self.filename2i[filename]
        return self.dataset[original_index]
