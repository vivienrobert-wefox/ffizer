import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from blbl import _CONFIG_DIR, _HYDRA_VERSION
from blbl.steps.train import instantiate_for_train
from wefox_ai_wai_ml.pipelining.neptune import hydra_neptune_main
import neptune.new as neptune
from omegaconf import DictConfig
from loguru import logger as _logger
import typing as t


@hydra_neptune_main(config_path=_CONFIG_DIR, config_name="steps/test.yaml", version_base=_HYDRA_VERSION)
def test(
    conf: DictConfig,
    run: neptune.Run,
    *,
    trainer: pl.Trainer = None,
    callbacks: dict[str, pl.Callback] = None,
    datamodule: pl.LightningDataModule = None,
    model: pl.LightningModule = None,
    logger: NeptuneLogger = None,
    **unused_args,
) -> (neptune.Run, dict[str, t.Any]):
    _logger.info("Starting test")
    model, datamodule, trainer, callbacks, logger = instantiate_for_train(
        conf.steps.test, run, model=model, datamodule=datamodule, trainer=trainer, callbacks=callbacks, logger=logger
    )
    return core_test(
        conf=conf,
        model=model,
        datamodule=datamodule,
        callbacks=callbacks,
        trainer=trainer,
        logger=logger,
    )


def core_test(
    conf: DictConfig,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    callbacks: dict[str, pl.Callback],
    model: pl.LightningModule,
    logger: NeptuneLogger,
) -> (neptune.Run, dict[str, t.Any]):
    if conf.steps.test.seed:
        pl.seed_everything(seed=conf.steps.test.seed, workers=True)

    if logger is not None and "model_checkpoint" in callbacks:
        trainer.best_model_path = callbacks["model_checkpoint"].best_model_path = logger.run[
            "artifacts/best_model_path"
        ].fetch()
        ckpt_path = "best"
    else:
        _logger.warning("No best model checkpoint found using last instead.")
        ckpt_path = "last"
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return logger.run, {}


if __name__ == "__main__":
    test()
