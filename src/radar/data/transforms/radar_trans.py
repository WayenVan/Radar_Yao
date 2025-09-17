import torch
from torchvision.transforms.functional import resize


class RadarTrans:
    def __init__(self, depth_size=(48, 64), radar_size=(48, 64)):
        self.depth_size = depth_size
        self.radar_size = radar_size

    def __call__(self, sample):
        sample["depth_data"] = (
            torch.from_numpy(sample["depth_data"]).float().unsqueeze(0)
        )
        sample["rgb_data"] = (
            torch.from_numpy(sample["rgb_data"]).float().permute(2, 0, 1)
        )
        sample["radar_data"] = (
            torch.from_numpy(sample["radar_data"].copy()).unsqueeze(0).float()
        )

        # resize
        sample["depth_data"] = resize(sample["depth_data"], self.depth_size)
        sample["radar_data"] = resize(sample["radar_data"], self.radar_size)

        return sample
