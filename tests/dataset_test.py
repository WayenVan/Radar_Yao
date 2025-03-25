import sys
import pytest

sys.path.append("src")
from radar.data.torch_dataset import RadarDataset


def test_radar_dataset_shape():
    dataset = RadarDataset("dataset")
    print(dataset[0].shape)
    assert dataset[0].shape is not None


def test_radar_dataset_len():
    dataset = RadarDataset("dataset")
    print(len(dataset))
    assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main(["-s", __file__ + "::test_radar_dataset_shape"])
