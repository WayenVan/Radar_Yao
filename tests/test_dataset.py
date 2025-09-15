import sys
from omegaconf import DictConfig, OmegaConf
from diffusers import UNet2DConditionModel


sys.path.append("src")
from radar.data.torch_dataset import RadarDataset
from radar.data.datamodule import RadarDataModule


def test_radar_dataset_shape():
    dataset = RadarDataset("/root/shared-data/Radar_Yao/dataset/radar-data/")
    for i in range(5):
        data = dataset[i]
        print(data)


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
