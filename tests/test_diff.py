import sys

sys.path.append("src")


from radar.modeling_diff.diff_pipline import RDDPMPipeline
from radar.modeling_diff.unet_conditional import UNet2DConditionModel
from radar.misc.utils import instantiate
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# from diffusers.models.unets import UNet2DModel, UNet2DConditionModel
#
from hydra import compose, initialize


import torch


def test_diff_config():
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="default_train")
        unet = instantiate(cfg.model.unet).cuda()
        scheduler = instantiate(cfg.model.scheduler)
        pipline = instantiate(cfg.model.pipline, unet=unet, scheduler=scheduler)

        print(pipline)
        cond = torch.randn(1, 1, 64, 64).cuda()
        out = pipline(
            num_inference_steps=1000, output_type="pil", r_conditional_input=cond
        ).images
        pipline.save_pretrained("outputs/rddpm-pipeline-test")


def test_diff():
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=256,
        block_out_channels=(64, 128, 256, 512),
        r_conditional_encoder_type="simple_conditional_encoder",
        r_conditional_encoder_kwargs=dict(
            in_channels=1, out_channels=32, block_out_channels=[64, 128, 256]
        ),
    ).cuda()
    scheduler = DDPMScheduler(num_train_timesteps=10000)
    pipline = RDDPMPipeline(unet=unet, scheduler=scheduler)

    cond = torch.randn(1, 1, 64, 64).cuda()
    out = pipline(
        num_inference_steps=1000, output_type="pil", r_conditional_input=cond
    ).images

    pipline.save_pretrained("outputs/rddpm-pipeline-test")


if __name__ == "__main__":
    # test_diff()
    test_diff_config()
