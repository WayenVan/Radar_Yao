import torch
import urllib
from PIL import Image
from torchvision import transforms
from torch import nn
from torchvision.transforms import functional as F


class DeepLabSegWrapper(nn.Module):
    def __init__(self, human_index=17, segmentation_input_size=(256, 256)):
        super(DeepLabSegWrapper, self).__init__()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(segmentation_input_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True
        )
        self.model.eval()

        self.human_index = human_index
        self.segmentation_input_size = segmentation_input_size

    @torch.no_grad()
    def forward(self, images):
        """
        @param images: tensor of shape (B, C, H, W)
        """
        H, W = images.shape[2:]
        preprocessed = self.preprocess(images)
        output = self.model(preprocessed)["out"]
        output_predictions = output.argmax(1)
        # [B,  H, W]
        output_predictions = (output_predictions == self.human_index).float()
        # [B, H, W]

        return F.resize(output_predictions, (H, W), antialias=True)


input_image = Image.open("outputs/video_0.png")
input_image = input_image.convert("RGB")

input_tensor = F.to_tensor(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
input_batch = torch.broadcast_to(input_batch, (10, -1, -1, -1))

model = DeepLabSegWrapper(human_index=12)
model.eval()

output_predictions = model(input_batch)

out = Image.fromarray(output_predictions[0].cpu().numpy())  # 0 is human?
out = out.convert("L")
out.save("outputs/deeplab.jpg")
