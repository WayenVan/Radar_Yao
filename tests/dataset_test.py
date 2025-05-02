import sys
import pytest
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")
from radar.data.torch_dataset import RadarDataset
from radar.data.datamodule import RadarDataModule


def test_radar_dataset_shape():
    dataset = RadarDataset("dataset")
    for i in range(len(dataset)):
        data = dataset[i]
        print(data["aoa_spectrum"].shape, data["aod_spectrum"].shape, data["label"])
        assert len(data) == 3, f"Expected 3 items, got {len(data)}"
        assert data["aoa_spectrum"].shape == (10, 91, 14), (
            f"Expected shape (10, 91, 14), got {data[0].shape}"
        )
        assert data["aod_spectrum"].shape == (10, 91, 14), (
            f"Expected shape (10, 91, 14), got {data[1].shape}"
        )
        assert isinstance(data["label"], int), (
            f"Expected label to be an int, got {type(data[2])}"
        )


def test_radar_datamodule():
    cfg = {"batch_size": 32, "num_workers": 4, "dataset": {"data_root": "dataset"}}
    cfg = DictConfig(cfg)
    dm = RadarDataModule(cfg)
    dm.setup("fit")
    loader = dm.train_dataloader()
    for batch in loader:
        assert len(batch) == 3, f"Expected 3 items, got {len(batch)}"
        print(
            batch["aoa_spectrum"].shape,
            batch["aod_spectrum"].shape,
            batch["label"].shape,
        )


def test_radar_dataset_len():
    dataset = RadarDataset("dataset")
    print(len(dataset))
    assert len(dataset) > 0


if __name__ == "__main__":
    # test_radar_datamodule()
    test_radar_dataset_shape()
