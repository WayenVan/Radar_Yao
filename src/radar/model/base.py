from lightning import LightningModule
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch

from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict, namedtuple
from torchmetrics import Accuracy


class BaseModel(LightningModule):
    def __init__(self, cfg: DictConfig, categorys: List[str]) -> None:
        super(BaseModel).__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss = instantiate(cfg.loss)
        self.categorys = categorys  # ['A', 'B', 'C', 'D']

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=len(categorys))
        self.val_acc = Accuracy(task="multiclass", num_classes=len(categorys))

    def forward(self, x: Any) -> Any:
        output = self.model(x)
        # output :[b, c]
        return output

    @torch.no_grad()
    def predict(self, x: Any) -> Any:
        output = self.model(x)
        logits = torch.argmax(output.out, dim=-1)
        predicted_labels = [self.categorys[i] for i in logits]
        return predicted_labels

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch["data"], batch["label"]
        output = self.model(x)

        # calculate accuracy
        batch_accu = self.train_acc(output.out, y)
        self.log("train_acc", batch_accu, prog_bar=True, logger=True)

        # log loss
        losses = self.loss(output, y)
        for name in losses._fields:
            value = getattr(losses, name)
            self.log(f"train_loss_{name}", value, prog_bar=True, logger=True)

        loss = losses.out
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch["data"], batch["label"]
        output = self.model(x)
        losses = self.loss(output, y)

        # update accuracy
        self.val_acc.update(output.out, y)

        # log loss
        for name in losses._fields:
            value = getattr(losses, name)
            self.log(f"val_loss_{name}", value, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, logger=True)
        self.val_acc.reset()
