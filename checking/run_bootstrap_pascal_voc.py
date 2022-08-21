from pathlib2 import Path
from src.dataset import PascalVOCDatasetInfo
import torch
from src.constant import BOOTSTRAP_DIR, GAUSSIAN_NOISE, TRANSFORMATIONS, ROOT
from src import bootstrap
import argparse
from tabulate import tabulate
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transformation", required=True, choices=TRANSFORMATIONS)
    parser.add_argument("-th", "--threshold", required=True)
    parser.add_argument("-p", "--save_path", required=True)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    bootstrap_path = args.save_path / 'data' / f'bootstrap-{args.transformation}'
    image_set = "val"
    bootstrap_path.mkdir(parents=True, exist_ok=True)

    dataset_info = PascalVOCDatasetInfo(args.save_path, image_set=image_set)
    image_info_df = dataset_info.image_info_df()
    bootstrap_df = bootstrap.bootstrap(image_info_df, 50, 200, args.transformation, float(args.threshold), bootstrap_path)
    
    bootstrap_df.to_csv(f"bootstrap_dfs/bootstrap_df-{args.transformation}.csv")
