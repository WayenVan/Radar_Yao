import sys
import pytest
import numpy as np

sys.path.append("src")
from radar.data.torch_dataset import RadarDataset

dataset = RadarDataset("dataset")

import matplotlib.pyplot as plt


print(dataset[0][0].shape)
print(dataset[0][1].shape)

data = np.concatenate([dataset[0][0][0, :, :], dataset[0][1][0, :, :]], axis=-1)
plt.imshow(data)
plt.show()
