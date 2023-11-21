from pathlib import Path
from loguru import logger
from lele import _CONFIG_DIR, _HYDRA_VERSION
from lele.data.mydatamodule import MyDataModule
import hydra
import neptune.new as neptune

import onnx
from wefox_ai_wai_ml.onnx.factory import OnnxFactory
from wefox_ai_wai_ml.pipelining.neptune import hydra_neptune_main
import pytorch_lightning as pl
import omegaconf as oc
import torch


@hydra_neptune_main(config_path=_CONFIG_DIR, config_name="base.yaml", version_base=_HYDRA_VERSION)
def convert(
    conf, run: neptune.Run, *, model: pl.LightningModule = None, datamodule: MyDataModule = None, **unused_args
):
    # initialise objects
    datamodule: pl.LightningDataModule = datamodule or hydra.utils.get_class(
        conf.steps.convert.data._target_
    ).load_from_checkpoint(f"{conf.paths.output_dir}/checkpoints/last.ckpt")
    datamodule.prepare_data()
    datamodule.setup("test")
    model: pl.LightningModule = model or hydra.utils.get_class(conf.steps.convert.model._target_).load_from_checkpoint(
        f"{conf.paths.output_dir}/checkpoints/last.ckpt", map_location=torch.device("cpu")
    )

    return core_convert(
        conf,
        model=model,
        datamodule=datamodule,
        run=run,
    )


def core_convert(conf: oc.DictConfig, run: neptune.Run, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    # get input sample
    input_sample, labels = next(iter(datamodule.test_dataloader()))

    # convert model
    output = OnnxFactory.create_onnx(model, input_sample=input_sample, check_outputs=True)
    output.seek(0)
    model_bytes = output.getvalue()

    if conf.paths.output_dir.startswith("s3:/"):
        import s3fs

        fs = s3fs.S3FileSystem()
        with fs.open(f"{conf.paths.output_dir}/artifacts/model.onnx", "wb") as f:
            f.write(model_bytes)
    else:
        output_path = Path(conf.paths.output_dir).joinpath("artifacts/model.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            f.write(model_bytes)

    if not conf.steps.convert.bentoml.export:
        return run, {}

    import bentoml

    # save to bento
    datamodule.trainer = None
    bento_model = bentoml.onnx.save_model(
        "lele_onnx",
        model=onnx.load_model(output),
        custom_objects={
            "data_parameters": {
                datamodule.parameters,
            }
        },
    )
    s3_path = f"{conf.steps.convert.bentoml.registry_path}/{bento_model.tag.name}/{bento_model.tag.version}"
    bentoml.models.export_model(bento_model.tag, s3_path)
    run["bentoml"] = {
        "tag": bento_model.tag.name,
        "version": bento_model.tag.version,
        "s3_path": s3_path,
    }
    return run, {
        "bento_model": bento_model,
    }


if __name__ == "__main__":
    convert()
