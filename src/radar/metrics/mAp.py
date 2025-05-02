import numpy
from torchmetrics import Metric


class MAp(Metric):
    def __init__(self, num_classes: int, iou_threshold: float = 0.5) -> None:
        super().__init__(compute_on_step=False)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: numpy.ndarray, targets: numpy.ndarray) -> None:
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> float:
        preds = numpy.concatenate(self.preds)
        targets = numpy.concatenate(self.targets)

        # Compute mAP here using preds and targets
        # For example, using a library like pycocotools or your own implementation

        return mAP_value  # Replace with actual mAP value calculation
