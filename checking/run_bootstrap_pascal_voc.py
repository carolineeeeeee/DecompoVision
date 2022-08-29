from pathlib2 import Path
from src.dataset import PascalVOCDatasetInfo
import torch
from src.constant import *
from src import bootstrap
import argparse
from tabulate import tabulate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transformation", required=True, choices=TRANSFORMATIONS)
    parser.add_argument("-th", "--threshold", required=True)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    bootstrap_path = BOOTSTRAP_DIR / f'bootstrap-{args.transformation}_{str(args.threshold)}'
    image_set = "val"
    bootstrap_path.mkdir(parents=True, exist_ok=True)

    dataset_info = PascalVOCDatasetInfo(BOOTSTRAP_DIR, image_set=image_set)
    image_info_df = dataset_info.image_info_df()
    bootstrap_df = bootstrap.bootstrap(image_info_df, 5, 5, args.transformation, float(args.threshold), bootstrap_path)
    
    if not os.path.exists('bootstrap_dfs'):
        os.mkdir('bootstrap_dfs')
    bootstrap_df.to_csv(f"bootstrap_dfs/bootstrap_df-{args.transformation}_{str(args.threshold)}.csv")
