from torch.utils.data import Dataset
import numpy as np
from .radar import RadarIndex
from pathlib import Path


class RadarDataset:
    def __init__(self, data_root) -> None:
        self.data_root = Path(data_root)
        self.index = RadarIndex(data_root)

    def __getitem__(self, index):
        id = self.index.id_list[index]
        state_dict = self.index.id_dict[id]

        file = state_dict["file"]

        # _files: ['aoa_spectrum.npy', 'aod_spectrum.npy']
        data = np.load(Path(self.data_root) / file)

        # TODO: adjust output data
        return data["aoa_spectrum"]

    def __len__(self):
        return len(self.index.id_list)


if __name__ == "__main__":
    dataset = RadarDataset("dataset")
    print(dataset[0])
