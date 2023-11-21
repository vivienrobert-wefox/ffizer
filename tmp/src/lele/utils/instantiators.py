import typing as t

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from wefox_ai_wai_ml.pipelining.hydra import instantiator


@instantiator
def instantiate_model(conf: DictConfig) -> pl.LightningModule:
    return hydra.utils.instantiate(conf.model)


@instantiator()
def instantiate_datamodule(conf: DictConfig) -> pl.LightningDataModule:
    return hydra.utils.instantiate(conf.data)


@instantiator
def instantiate_callbacks(conf: DictConfig) -> t.Optional[dict[str, pl.Callback]]:
    if "callbacks" in conf:
        return {
            cb_name: hydra.utils.instantiate(cb_conf)
            for cb_name, cb_conf in conf.callbacks.items()
            if "_target_" in cb_conf
        }
    return None


@instantiator
def instantiate_trainer(conf: DictConfig, callbacks: dict[str, pl.Callback], logger) -> t.Optional[pl.Trainer]:
    return hydra.utils.instantiate(conf.trainer, callbacks=callbacks and list(callbacks.values()), logger=logger)
