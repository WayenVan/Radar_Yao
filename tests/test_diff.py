import sys

sys.path.append("src")


from radar.modeling_diff.diff_pipline import RDDPMPipeline
from radar.modeling_diff.unets import U_Net
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel


def test_diff():
    unet = U_Net(in_ch=1, out_ch=1, sample_size=(64, 64)).cuda()
    scheduler = DDPMScheduler(num_train_timesteps=10000)

    pipline = RDDPMPipeline(unet=unet, scheduler=scheduler)

    out = pipline(num_inference_steps=1000, output_type="pil").images

    import matplotlib.pyplot as plt

    plt.imshow(out[0])


if __name__ == "__main__":
    test_diff()
