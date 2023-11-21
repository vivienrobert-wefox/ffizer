from collections.abc import Callable

from omegaconf import DictConfig

from loguru import logger
import hydra

from wefox_ai_wai_ml.pipelining.neptune import prepare_run


def run_pipeline(base_conf: DictConfig, _debug_run=None):
    conf, run = prepare_run(base_conf, _debug_run, override_steps=True)  # Change to False if need to customize process.
    pipeline: dict[str, Callable] = {
        key: hydra.utils.get_method(value["_target_"])
        for key, value in conf.steps.items()
        if "_target_" in value
    }
    logger.info(f"Steps to run: {pipeline}")

    args = {}
    for step_name, step_task in pipeline.items():
        logger.info(f"Launching {step_name}")
        run, new_objs = step_task(conf, run, **args)
        run.wait()
        args.update(new_objs)
        logger.debug(f"Available args after {step_name}: {args.keys()}")


if __name__ == "__main__":
    from blbl import _CONFIG_DIR, _HYDRA_VERSION

    @hydra.main(config_path=_CONFIG_DIR, config_name="base.yaml", version_base=_HYDRA_VERSION)
    def main(conf):
        run_pipeline(conf)

    main()
