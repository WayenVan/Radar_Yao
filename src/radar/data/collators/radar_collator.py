import torch
import numpy as np


class RadarCollator:
    def __call__(self, batch):
        zipped = {key: [d[key] for d in batch] for key in batch[0]}
        for key, value in zipped.items():
            if isinstance(value[0], torch.Tensor):
                zipped[key] = torch.stack(value)
            elif isinstance(value[0], np.ndarray):
                zipped[key] = torch.tensor(np.stack(value))

        ret = zipped
        ret["label"] = zipped["depth_data"]
        ret["r_conditional_input"] = zipped["radar_data"]
        return ret
