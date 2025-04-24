import torch
from torch import nn
from collections import namedtuple


class CELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    CELossOutput = namedtuple(
        "CELossOutput",
        [
            "out",
        ],
    )

    def forward(self, outputs: torch.Tensor, target: torch.Tensor):
        logits = outputs.out
        loss = self.criterion(logits, target)
        return self.CELossOutput(loss)
