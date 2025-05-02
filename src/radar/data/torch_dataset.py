from torch.utils.data import Dataset
import numpy as np
from .radar import RadarIndex
from pathlib import Path


def standardize(data):
    return (data - np.mean(data, axis=(-1, -2), keepdims=True)) / np.std(
        data, axis=(-1, -2), keepdims=True
    )


class RadarDataset:
    def __init__(self, data_root) -> None:
        self.data_root = Path(data_root)
        self.index = RadarIndex(data_root)

    def __getitem__(self, index):
        id = self.index.id_list[index]
        state_dict = self.index.id_dict[id]

        file = state_dict["file"]

        # _files: ['aoa_spectrum.npy', 'aod_spectrum.npy']
        # shape: [10, 91, 14] [time, angle, distance]
        data = np.load(Path(self.data_root) / file)

        # TODO: adjust output data
        aoa_spectrum, aod_spectrum = (
            standardize(data["aoa_spectrum"]),
            standardize(data["aod_spectrum"]),
        )
        label = state_dict["gesture"]
        return dict(
            aoa_spectrum=aoa_spectrum,
            aod_spectrum=aod_spectrum,
            label=label,
        )

    def __len__(self):
        return len(self.index.id_list)

    def get_cats(self):
        return self.index.cats


if __name__ == "__main__":
    dataset = RadarDataset("dataset")
    print(dataset[0])
