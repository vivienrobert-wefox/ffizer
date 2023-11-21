import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from lele import _CONFIG_DIR, _HYDRA_VERSION
from lele.utils import instantiators
import typing as t
import neptune.new as neptune

from loguru import logger as _logger
from omegaconf import DictConfig

from wefox_ai_wai_ml.pipelining.neptune import hydra_neptune_main


def instantiate_for_train(
    conf: DictConfig,
    run: neptune.Run,
    *,
    model: pl.LightningModule = None,
    datamodule: pl.LightningDataModule = None,
    trainer: pl.Trainer = None,
    callbacks: dict[str, pl.Callback] = None,
    logger: NeptuneLogger = None,
):
    # logger
    logger = logger or (conf.get("logger") and NeptuneLogger(**conf.get("logger"), run=run)) or None

    if conf.seed:
        pl.seed_everything(seed=conf.seed, workers=True)

    datamodule: pl.LightningDataModule = instantiators.instantiate_datamodule(conf, datamodule=datamodule)
    model: pl.LightningModule = instantiators.instantiate_model(conf, model=model)

    callbacks = instantiators.instantiate_callbacks(conf, callbacks=callbacks)

    _logger.info(f"callbacks: {callbacks}")

    trainer = instantiators.instantiate_trainer(conf, callbacks=callbacks, logger=logger, trainer=trainer)

    return model, datamodule, trainer, callbacks, logger


@hydra_neptune_main(config_path=_CONFIG_DIR, config_name="steps/train.yaml", version_base=_HYDRA_VERSION)
def train(
    conf: DictConfig,
    run: neptune.Run,
    *,
    model: pl.LightningModule = None,
    datamodule: pl.LightningDataModule = None,
    trainer: pl.Trainer = None,
    callbacks: dict[str, pl.Callback] = None,
    logger: NeptuneLogger = None,
    **unused_args,
) -> (neptune.Run, dict[str, t.Any]):
    if conf.steps.train.seed:
        pl.seed_everything(seed=conf.steps.train.seed, workers=True)
    model, datamodule, trainer, callbacks, logger = instantiate_for_train(
        conf.steps.train, run, model=model, datamodule=datamodule, trainer=trainer, callbacks=callbacks, logger=logger
    )
    return core_train(
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


def core_train(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: dict[str, pl.Callback],
    logger: t.Optional[NeptuneLogger],
) -> (neptune.Run, dict[str, t.Any]):
    trainer.fit(model=model, datamodule=datamodule, ckpt_path="last")

    if logger is not None and "model_checkpoint" in callbacks:
        logger.run["artifacts/best_model_path"] = callbacks["model_checkpoint"].best_model_path

    return (
        logger.run,
        {
            "model": model,
            "datamodule": datamodule,
            "trainer": trainer,
            "callbacks": callbacks,
            "logger": logger,
        },
    )


if __name__ == "__main__":
    train()
