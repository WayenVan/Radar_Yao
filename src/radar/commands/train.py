import hydra

from omegaconf import DictConfig, OmegaConf
from ..misc.utils import instantiate
import torch
import os
from accelerate import Accelerator
from ..engines.trainer import DiffusionTrainer
from ..engines.callbacks import SaveBestMetricCallback
from transformers.trainer import TrainingArguments
from transformers import set_seed
from torch.utils.data import DataLoader, Subset


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="default_train"
)
def main(cfg: DictConfig):
    # creat datset
    train_transform = instantiate(cfg.data.transform)
    train_set = instantiate(
        cfg.data.dataset,
        transform=train_transform,
    )
    # subset with random 100 samples for eval
    eval_set = Subset(train_set, torch.randperm(len(train_set))[:100])

    # Create models
    unet = instantiate(cfg.model.unet).cpu()
    scheduler = instantiate(cfg.model.scheduler)

    # create callbacks
    callbacks = [
        SaveBestMetricCallback(metric_name="eval_mse"),
    ]
    # create trainer
    training_args = TrainingArguments(**cfg.engine.training_args)
    trainer = DiffusionTrainer(
        model=unet,
        scheduler=scheduler,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=instantiate(cfg.data.collator),
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    main()
