import hydra
import pytest
from ilil import _CONFIG_DIR, _HYDRA_VERSION
from ilil.utils.testing import KThread
import omegaconf as oc


@pytest.mark.slow
@pytest.mark.parametrize(
    ("base_dir", "train_config", "experiment", "steps"),
    (
        (_CONFIG_DIR, "base.yaml", None, "[train,test]"),
        # could read other files from a folder there to automatically test configs in a given folder
    ),
)
def test_pipeline(base_dir, train_config, experiment, steps, tmp_path):
    with hydra.initialize_config_dir(base_dir, version_base=_HYDRA_VERSION):
        overrides = ["dev=limit", f"steps={steps}"]
        if experiment is not None:
            overrides.append(f"experiment={experiment}")
        cfg = hydra.compose(train_config, overrides=overrides, return_hydra_config=True)
        # workaround from (https://github.com/facebookresearch/hydra/issues/2017#issuecomment-1254220345)
        hydra.utils.HydraConfig.instance().set_config(cfg.copy())
        with oc.open_dict(cfg):
            (tmp_path/"output_dir").mkdir()
            (tmp_path/"bento_registry").mkdir()
            (tmp_path/"bento_registry/model1").mkdir()
            (tmp_path/"bento_registry/model2").mkdir()
            cfg.paths.output_dir = str(tmp_path/"output_dir")

            for step_name, step_items in cfg.steps.items():
                if "bentoml" in step_items:
                    step_items.bentoml.registry_path = str(tmp_path/"bento_registry")
                    step_items.bentoml.export = True
            del cfg["hydra"]  # hydra part of the config is not passed in real behaviour

        from ilil.run_pipeline import run_pipeline
        run_pipeline(cfg)

@pytest.mark.slow
@pytest.mark.parametrize(
    ("base_dir", "train_config", "experiment", "steps"),
    (
        (_CONFIG_DIR, "base.yaml", None, "[train,test,convert]"),
        # could read other files from a folder there to automatically test configs in a given folder
    ),
)
def test_interrupt(base_dir, train_config, experiment, steps, tmp_path):
    with hydra.initialize_config_dir(base_dir, version_base=_HYDRA_VERSION):
        overrides = ["dev=limit", f"steps={steps}"]
        if experiment is not None:
            overrides.append(f"experiment={experiment}")
        cfg = hydra.compose(train_config, overrides=overrides, return_hydra_config=True)
        # workaround from (https://github.com/facebookresearch/hydra/issues/2017#issuecomment-1254220345)
        hydra.utils.HydraConfig.instance().set_config(cfg.copy())
        with oc.open_dict(cfg):
            (tmp_path/"output_dir").mkdir()
            (tmp_path/"bento_registry").mkdir()
            (tmp_path/"bento_registry/model1").mkdir()
            (tmp_path/"bento_registry/model2").mkdir()
            cfg.paths.output_dir = str(tmp_path/"output_dir")

            for step_name, step_items in cfg.steps.items():
                if "bentoml" in step_items:
                    step_items.bentoml.registry_path = str(tmp_path/"bento_registry")
                    step_items.bentoml.export = True
            del cfg["hydra"]  # hydra part of the config is not passed in real behaviour

        print(cfg)

        from ilil.run_pipeline import run_pipeline
        from wefox_ai_wai_ml.pipelining.neptune import prepare_run
        conf, run = prepare_run(cfg)

        p = KThread(target=run_pipeline, args=(conf,), kwargs={"_debug_run": run})
        p.start()

        import time
        print("Waiting until a checkpoint is made.")
        t = 0
        while True:
            assert p.is_alive(), "Thread finished without checkpointing"
            if (tmp_path/"output_dir" / "checkpoints" / "last.ckpt").exists():
                print("Checkpoint found, terminating thread")
                break
            time.sleep(5)
            t += 5
            print(t)
        print("Killing the thread")
        while p.is_alive():
            p.kill()
            time.sleep(2)

        assert not p.is_alive()

        run_pipeline(conf, _debug_run=run)

