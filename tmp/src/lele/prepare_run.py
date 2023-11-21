import hydra

from lele import _CONFIG_DIR, _HYDRA_VERSION
from wefox_ai_wai_ml.pipelining.neptune import prepare_run

if __name__ == "__main__":
    @hydra.main(config_path=_CONFIG_DIR, config_name="base.yaml", version_base=_HYDRA_VERSION)
    def main(conf) -> None:
        conf, run = prepare_run(conf)
        print(run["sys/id"].fetch())
        run.stop()

    main()
