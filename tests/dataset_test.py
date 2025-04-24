import sys
import pytest

sys.path.append("src")
from radar.data.torch_dataset import RadarDataset


def test_radar_dataset_shape():
    dataset = RadarDataset("dataset")
    for i in range(len(dataset)):
        data = dataset[i]
        assert len(data) == 2, f"Expected 2 items, got {len(data)}"
        assert data[0].shape == (10, 91, 14), (
            f"Expected shape (10, 91, 14), got {data[0].shape}"
        )
        assert data[1].shape == (10, 91, 14), (
            f"Expected shape (10, 91, 14), got {data[1].shape}"
        )


def test_radar_dataset_len():
    dataset = RadarDataset("dataset")
    print(len(dataset))
    assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main(["-s", __file__ + "::test_radar_dataset_shape"])
