import sys

sys.path.append("src")

from radar.modeling_diff.unets import U_Net
import numpy as np
from diffusers import DDPMPipeline
from diffusers.pipelines.ddpm import DDPMPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput, DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.models.unets import UNet2DModel
from diffusers import AutoModel

# from diffusers import AutoModel
import json

import torch
from typing import List, Optional, Union
import PIL
import PIL.Image
import requests

from diffusers import DiffusionPipeline

# pipeline = DiffusionPipeline.from_pretrained(
#     "stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True
# )


# pipeline.save_pretrained("outputs/sd-v1-5")
class YourCustomPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], torch.Tensor]


class YourCustomPipeline(DiffusionPipeline):
    def __init__(
        self,
        my_unet: U_Net,
    ):
        super().__init__()
        self.register_modules(
            my_unet=my_unet,
        )

    def __call__(self, *args, **kwargs):
        return YourCustomPipelineOutput(images=torch.randn(1, 3, 256, 256))


unet = U_Net(3, 1, (224, 224))
print(unet.dtype)

pipe = YourCustomPipeline(my_unet=unet)
pipe.save_pretrained("outputs/your-custom-unet-cfg")

with open("outputs/your-custom-unet-cfg/config.json", "r") as f:
    cfg = json.load(f)

print(U_Net)
