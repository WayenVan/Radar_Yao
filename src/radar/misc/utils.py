from omegaconf import OmegaConf, DictConfig
import importlib


def instantiate(cfg: DictConfig, **kwargs):
    """
    A fixed instantiate function which is compatible with huggingface ConfigMixin
    """
    _cfg = OmegaConf.to_container(cfg, resolve=True)  # need to turn into the

    class_path = _cfg.pop("_target_")
    cfg_kwargs = _cfg

    module_path, class_name = class_path.rsplit(".", 1)  # 分离模块和类名
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    cls_kwargs = {**cfg_kwargs, **kwargs}

    return cls(**cls_kwargs)
