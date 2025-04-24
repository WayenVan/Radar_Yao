import torch
from torch import nn
from timm.models.resnet import BasicBlock, downsample_conv
import einops


def create_cnn(name="default"):
    if name == "default":
        return nn.Sequential(
            BasicBlock(
                1,
                64,
                stride=2,
                downsample=downsample_conv(1, 64, kernel_size=3, stride=2),
            ),
            BasicBlock(
                64,
                128,
                stride=2,
                downsample=downsample_conv(64, 128, kernel_size=3, stride=2),
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ), 128


class Baseline(nn.Module):
    """
    A simple baseline with ResBlock + Transformer
    """

    def __init__(
        self,
        num_classes: int,
        cnn_arch: str = "default",
        nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resnet, self.d_model = create_cnn(cnn_arch)
        self.tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * ff_factor,
            dropout=dropout,
            activation="gelu",
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.d_model), requires_grad=True
        )
        self.classifer = nn.Linear(self.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, t, angle, distance]
        B, T, H, W = x.shape
        x = einops.rearrange(x, "b t a d -> (b t) 1 a d")
        x = self.resnet(x)
        x = einops.rearrange(x, "(b t) c -> b t c", b=B, t=T)
        cls_token = einops.repeat(self.cls_token, "1 1 c -> b 1 c", b=B)
        x = torch.cat([cls_token, x], dim=1)
        x = self.tf_encoder_layer(x)
        cls_token = x[:, 0]
        logits = self.classifer(cls_token)
        return logits


if __name__ == "__main__":
    model = Baseline(10)
    x = torch.randn(2, 10, 91, 14)
    y = model(x)
    print(y.shape)  # [2, 10]
