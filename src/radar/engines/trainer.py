from omegaconf import DictConfig
from transformers.trainer import Trainer, TrainingArguments
from radar.modeling_diff.diff_pipline import RDDPMPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from typing import Optional
import os
import torch
from torch.nn import functional as F


class DiffusionTrainer(Trainer):
    def __init__(self, scheduler: SchedulerMixin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler: SchedulerMixin = scheduler

    @staticmethod
    def loss_fn(model, scheduler, label, r_conditional_input):
        """
        x: [batch_size, 1, height, width]
        r_conditional_input: [batch_size, 1, height, width]
        """
        noise = torch.randn_like(label)
        random_timestep = torch.randint(
            0, scheduler.config.num_train_timesteps, (label.shape[0],), dtype=torch.long
        ).to(label.device)
        noisy_latents = scheduler.add_noise(label, noise, random_timestep)
        noise_pred = model(
            noisy_latents, random_timestep, r_conditional_input=r_conditional_input
        ).sample
        return F.mse_loss(noise_pred, noise)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        r_conditional_input = inputs.pop("r_conditional_input").to(
            self.accelerator.device
        )
        label = inputs.pop("label").to(self.accelerator.device)

        loss = self.loss_fn(model, self.scheduler, label, r_conditional_input)
        self.log(dict(steps=self.state.global_step))

        return (loss, None) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is not None:
            raise NotImplementedError

        # 创建包装后的模型
        pipline = RDDPMPipeline(
            unet=self.model,
            scheduler=self.scheduler,
        )

        # 使用包装后的模型进行保存
        pipline.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )

        # 保存其他必要文件（processing_class, training_args等）
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
