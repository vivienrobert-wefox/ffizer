import torch
import pytorch_lightning as pl

import torch.nn as nn
import torchmetrics

from .components.mycomponent import MyComponent


class MyModule(pl.LightningModule):
    def __init__(self, nb_classes: int):
        super().__init__()
        self.save_hyperparameters()

        # init model
        self.model = MyComponent()
        self.softmax = nn.Softmax(dim=1)

        # init loss
        self.criterion = ...

        # init metrics
        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=nb_classes)
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=nb_classes)
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=nb_classes)
        self.last_val_acc = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.log("train/loss", loss)
        self.train_acc.update(preds, y)
        self.trainer.progress_bar_metrics["train/acc"] = self.train_acc.compute().item()
        return {"loss": loss}

    def on_train_epoch_end(self):
        self.trainer.progress_bar_metrics.pop("train/acc", None)
        self.log("train/acc", self.train_acc.compute())
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.log("val/loss", loss)
        self.val_acc.update(preds, x)
        self.trainer.progress_bar_metrics[f"val/acc"] = self.val_acc.compute().item()
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        self.trainer.progress_bar_metrics.pop("val/acc", None)
        val_acc = self.val_acc.compute()
        self.log("val/acc", val_acc)
        self.last_val_acc = val_acc.item()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        self.test_acc.update(preds, y)
        test_acc = self.test_acc.compute().item()
        self.log("test/acc", test_acc)
        self.test_acc.reset()
        return {"test/acc": test_acc}

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer

    def forward(self, x):
        x = self.model.forward(x)
        return self.softmax(x)
