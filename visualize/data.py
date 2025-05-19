import sys

sys.path.append("src")
import cv2
from radar.data.torch_dataset import RadarDataset


dataset = RadarDataset("/root/shared-data/Radar_Yao/dataset/radar-data/")
data = dataset[0]

video = data[1]

print(video.shape)
