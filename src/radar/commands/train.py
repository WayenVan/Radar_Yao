import hydra

from omegaconf import DictConfig, OmegaConf
from ..misc.utils import instantiate
import torch
import os
from accelerate import Accelerator
from ..engines.trainer import DiffusionTrainer
from transformers.trainer import TrainingArguments
from torch.utils.data import DataLoader, Subset

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))


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
    eval_set = Subset(train_set, list(range(2)))

    # Create models
    unet = instantiate(cfg.model.unet).cpu()
    scheduler = instantiate(cfg.model.scheduler)

    # create trainer
    training_args = TrainingArguments(**cfg.engine.training_args)
    trainer = DiffusionTrainer(
        model=unet,
        scheduler=scheduler,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=instantiate(cfg.data.collator),
    )
    trainer.train()


if __name__ == "__main__":
    main()
