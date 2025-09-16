import sys

sys.path.append("src")
from radar.modeling_diff.unet_conditional import UNet2DConditionModel
import torch


def test_unet_conditional():
    model = UNet2DConditionModel(
        sample_size=64,
        in_channels=1,
        out_channels=1,
        r_conditional_encoder_type="simple_conditional_encoder",
        r_conditional_encoder_kwargs=dict(
            in_channels=1, out_channels=32, block_out_channels=[64, 128, 256]
        ),
    ).cuda()

    x = torch.randn(1, 1, 64, 64).cuda()
    t = torch.randint(0, 1000, (1,)).cuda()
    # c = torch.randn(1, 32).cuda()
    cond = torch.randn(1, 1, 64, 64).cuda()

    out = model(x, t, r_conditional_input=cond)
    print(out.sample.shape)  # Should print torch.Size([1, 3, 64, 64])


if __name__ == "__main__":
    test_unet_conditional()
