import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import sys

sys.path.append("src")
from radar.model.base import BaseModel


@hydra.main(version_base=None, config_path="../configs", config_name="default_train")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    BaseModel(cfg, categorys=["A", "B", "C", "D"])


if __name__ == "__main__":
    my_app()
