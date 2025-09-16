from diffusers.utils import BaseOutput
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from diffusers.models.unets.unet_2d_blocks import DownBlock2D, DownEncoderBlock2D


@dataclass
class ConditionalEncoderOutput(BaseOutput):
    hidden_states: torch.Tensor


class SimpleConditionalEncoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, block_out_channels: list[int]
    ):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.down_block = torch.nn.ModuleList()
        block_in_channel = block_out_channels[0]
        for block_out_channel in block_out_channels:
            assert block_out_channel % 2 == 0, "block_out_channel must be even"
            self.down_block.append(
                DownEncoderBlock2D(
                    in_channels=block_in_channel,
                    out_channels=block_out_channel,
                    add_downsample=True,
                )
            )
            block_in_channel = block_out_channel

        self.conv_out = torch.nn.Conv2d(
            block_out_channels[-1], out_channels, kernel_size=3, padding=1
        )
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> ConditionalEncoderOutput:
        # x: [batch_size, in_channels, height, width]
        x = self.conv_in(x)
        for down in self.down_block:
            x = down(x)
        x = self.conv_out(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)  # [batch_size, out_channels]
        return ConditionalEncoderOutput(hidden_states=x)


if __name__ == "__main__":
    model = SimpleConditionalEncoder(
        in_channels=3, out_channels=128, block_out_channels=[64, 128, 256]
    ).cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    output = model(x)
    print(output.hidden_states.shape)  # Should print torch.Size([1, 128])
