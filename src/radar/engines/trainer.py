from omegaconf import DictConfig
from transformers.trainer import Trainer, TrainingArguments
from radar.modeling_diff.diff_pipline import RDDPMPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from typing import Optional
import os
import torch
from torch.nn import functional as F
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_utils import EvalLoopOutput
import numpy as np
from sklearn.metrics import mean_squared_error

from transformers.modeling_utils import unwrap_model
from .callbacks import DiffusionTrainerCallbackHandler


class DiffusionTrainer(Trainer):
    def __init__(self, scheduler: SchedulerMixin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns is True:
            raise ValueError(
                "DiffusionTrainer does not support `remove_unused_columns` yet. Please set `remove_unused_columns=False` when instantiating the Trainer."
            )
        if kwargs.get("compute_metrics", None) is not None:
            raise ValueError(
                "DiffusionTrainer does not support `compute_metrics`. The trainner already has its own metrics computation."
            )
        self.scheduler: SchedulerMixin = scheduler
        self.compute_metrics = self._compute_metrics
        self.unet_config = unwrap_model(self.model).config

        self.callback_handler = DiffusionTrainerCallbackHandler(
            self,
            self.callback_handler.callbacks,  # WARN: replaceing the original callback handler
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )

        # redefine callback handler to pass trainer instance

    @staticmethod
    def _compute_metrics(pred: EvalLoopOutput) -> Dict[str, float]:
        preds = pred.predictions
        labels = pred.label_ids
        mse = mean_squared_error(
            labels.reshape(labels.shape[0], -1),
            preds.reshape(preds.shape[0], -1),
        )
        return {"mse": mse}

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
        inputs = self._prepare_inputs(inputs)

        r_conditional_input = inputs.pop("r_conditional_input")
        label = inputs.pop("label")

        assert label.shape[2:] == tuple(self.unet_config.sample_size), (
            f"Unexpected image shape {label.shape[2:]}, "
            f"expected {self.unet_config.sample_size} following the config."
        )

        loss = self.loss_fn(model, self.scheduler, label, r_conditional_input)

        # if (
        #     self.args.logging_steps > 0
        #     and self.state.global_step % self.args.logging_steps == 0
        # ):
        #     self.log(dict(steps=self.state.global_step))

        return (loss, None) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 自定义预测步骤
        model.eval()
        with torch.no_grad():
            # 创建包装后的模型
            inputs = self._prepare_inputs(inputs)
            r_conditional_input = inputs.pop("r_conditional_input")
            labels = inputs.pop("label")

            pipline = RDDPMPipeline(
                unet=self.model,
                scheduler=self.scheduler,
            )
            pred = pipline(
                batch_size=r_conditional_input.shape[0],
                r_conditional_input=r_conditional_input,
                num_inference_steps=self.scheduler.config.num_train_timesteps,
                output_type="tensor",
                slience=True,
            )

        return None, pred.images, labels

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is not None:
            raise NotImplementedError

        # 创建包装后的模型
        pipline = RDDPMPipeline(
            unet=unwrap_model(self.model),  # NOTE: in case when we have
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
