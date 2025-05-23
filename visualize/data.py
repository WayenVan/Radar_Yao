import sys

sys.path.append("src")
import cv2
from radar.data.torch_dataset import RadarDataset


dataset = RadarDataset("/root/shared-data/Radar_Yao/dataset/radar-data/")
data = dataset[4]

video = data[1]


for i in range(video.shape[0]):
    cv2.imwrite(f"outputs/video_{i}.png", cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))
