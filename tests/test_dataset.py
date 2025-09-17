import sys
from omegaconf import DictConfig, OmegaConf
from diffusers import UNet2DConditionModel
from datasets import load_dataset, load_from_disk

sys.path.append("src")
from radar.data.datasets.radar import RadarDataset


def test_radar_dataset():
    dataset = RadarDataset("dataset/rnb-radar", seleted_range_bin=[76, 77])
    print(len(dataset))
    for i in range(len(dataset)):
        sample = dataset[i]
        print(
            sample["depth_data"].shape,
            sample["rgb_data"].shape,
            sample["radar_data"].shape,
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
    # test_radar_dataset_shape()
    test_radar_dataset()
