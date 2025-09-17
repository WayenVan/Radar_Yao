from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
import polars as pl

from scipy.io import loadmat


class RadarDataset(Dataset):
    def __init__(
        self, data_root, split="train", seleted_range_bin=None, transform=None
    ):
        self.data_root = data_root
        self.transforms = transform
        dataset = load_dataset(
            "parquet", data_files={split: os.path.join(data_root, f"{split}.parquet")}
        )[split].to_polars()

        self.dataset = dataset
        if seleted_range_bin is not None:
            self.dataset = self.dataset.filter(
                pl.col("selected_range_bin").is_in(seleted_range_bin)
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        depth_data = np.load(
            self.full_path(row["file_path_depth"].item()), allow_pickle=True
        )
        rgb_data = np.load(
            self.full_path(row["file_path_rgb"].item()), allow_pickle=True
        )
        radar_data = loadmat(self.full_path(row["file_path_radar"].item()))[
            "spec_db_slice"
        ][::-1]

        ret = dict(depth_data=depth_data, rgb_data=rgb_data, radar_data=radar_data)

        if self.transforms:
            ret = self.transforms(ret)

        return ret

    def full_path(self, relative_path):
        return os.path.join(self.data_root, relative_path)
