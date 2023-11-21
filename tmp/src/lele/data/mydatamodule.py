import pytorch_lightning as pl
import torch
import torch.utils.data as torch_data
from .components.mycomponent import MyComponent


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_percentage=0.8,
        storage_path="data",
        num_workers=4,
        pin_memory=False,
        *,
        custom_component: MyComponent,
    ):
        super().__init__()
        self.save_hyperparameters()

        # save parameters
        self.batch_size = batch_size
        self.train_percentage = train_percentage
        self.storage_path = storage_path

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.custom_component = custom_component
        self.parameters = {}

        self.already_prepare = False

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, save to disk, etc...
        # Do not assign to state here (no self.x = )
        if self.already_prepare:
            return

        self.already_prepare = True

    def setup(self, stage: str):
        # things to do on every process in DDP
        # split, load data, set variables, etc...
        # Can be called once for trainer.fit and again for trainer.test
        if stage == "fit":
            self.train_set = ...
            train_size = int(len(self.train_set) * self.train_percentage)
            val_size = len(self.train_set) - train_size

            train_set, val_set = torch_data.random_split(
                self.train_set,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.train_loader = torch_data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=self.pin_memory,
            )
            self.val_loader = torch_data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=self.pin_memory,
            )

        if stage == "test":
            self.test_set = ...

            self.test_loader = torch_data.DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=self.pin_memory,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def teardown(self, stage: str):
        super().teardown(stage)
        # called on every process in DDP
        # clean up after fit or test
