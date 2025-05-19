from torch.utils.data import Dataset
import numpy as np
from .radar_index import RadarDataIndex
from pathlib import Path


def standardize(data):
    return (data - np.mean(data, axis=(-1, -2), keepdims=True)) / np.std(
        data, axis=(-1, -2), keepdims=True
    )


class RadarDataset:
    def __init__(self, data_root) -> None:
        self.data_root = Path(data_root)
        self.index = RadarDataIndex(data_root)
        self.ids = self.index.ids

    def __getitem__(self, index):
        id = self.ids[index]

        depth_data_files, radar_data_files, rgb_data_files = self.index.get_data_by_id(
            id
        )

        # load radar data
        #

        radar_azi = []
        radar_ele = []
        radar_spec_db = []

        for file in radar_data_files:
            radar_data = np.load(file)
            radar_azi.append(radar_data["azi"])
            radar_ele.append(radar_data["ele"])
            radar_spec_db.append(radar_data["spec_db"])
        radar_azi = np.stack(radar_azi, axis=0)
        radar_ele = np.stack(radar_ele, axis=0)
        radar_spec_db = np.stack(radar_spec_db, axis=0)

        depth_data = np.stack([np.load(file) for file in depth_data_files], axis=0)
        rgb_data = np.stack([np.load(file) for file in rgb_data_files], axis=0)
        return depth_data, rgb_data, radar_azi, radar_ele, radar_spec_db
        """
        rgb_data.shape = (50, 480, 640)
        depth_data.shape = (50, 480, 640)
        radar_spec_db.shape = (50, 61, 31, 50)
        radar_azi.shape = (50, 31)
        radar_ele.shape = (50, 61)
        """

    def __len__(self):
        return len(self.index.ids)

    def get_cats(self):
        return self.index.cats


if __name__ == "__main__":
    dataset = RadarDataset("dataset")
    print(dataset[0])
