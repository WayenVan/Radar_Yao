from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from .torch_dataset import RadarDataset


class RadarDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        shuffle: bool = False,  # lightning will handle the shuffle situation
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers or 1
        self.data_root = cfg.dataset.data_root
        self.shuffle = shuffle

        # setup cats and other things
        self.val_indices = [0, 1, 2]
        self.dataset = RadarDataset(self.data_root)
        self.cats = self.dataset.get_cats()

    def setup(self, stage: str) -> None:
        data_length_total = len(self.dataset)

        total_indices = list(range(data_length_total))
        train_indices = [i for i in total_indices if i not in self.val_indices]

        if stage == "fit" or stage is None:
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, self.val_indices)
        if stage == "test":
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

    def get_cats(self):
        if not hasattr(self, "cats"):
            raise ValueError("Cats not set. Please call setup() first.")
        return self.cats
