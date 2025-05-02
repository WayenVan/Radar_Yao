import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import sys

sys.path.append("src")
from radar.model.base import BaseModel
from radar.model.datamodule import DataModule


@hydra.main(version_base=None, config_path="../configs", config_name="default_train")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    BaseModel(cfg, categorys=["A", "B", "C", "D"])
    DataModule


if __name__ == "__main__":
    train()
