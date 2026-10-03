import os
from datetime import datetime
from typing import List, Optional

from omegaconf import OmegaConf, errors

from ai_fundamental_from_scratch.domain.config import (
    AppConfig,
    DataConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    PersistenceConfig,
    TrainerConfig,
)
from ai_fundamental_from_scratch.ports.outbound.config_port import ConfigRepositoryPort


class OmegaConfAdapter(ConfigRepositoryPort):
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = config_dir

    def load_config(self, overrides: Optional[List[str]] = None) -> AppConfig:
        cli_cfg = OmegaConf.from_cli(overrides or [])
        exp_name = cli_cfg.get("experiment", None)

        # 1. Base default map 로드 (main.yaml)
        main_path = os.path.join(self.config_dir, "main.yaml")
        if not os.path.exists(main_path):
            raise FileNotFoundError(f"Main config file not found: {main_path}")

        base_cfg = OmegaConf.load(main_path)

        defaults_map = {}
        if "defaults" in base_cfg:
            for item in base_cfg["defaults"]:
                for key, val in item.items():
                    defaults_map[key] = val

        # 2. Experiment 파일 존재 여부 검증
        exp_cfg = OmegaConf.create()
        if exp_name:
            exp_path = os.path.join(self.config_dir, "experiment", f"{exp_name}.yaml")
            if not os.path.exists(exp_path):
                raise FileNotFoundError(
                    f"\n[ConfigError] Specified experiment file '{exp_name}.yaml' does not exist in '{os.path.join(self.config_dir, 'experiment')}'."
                    f"\nPlease check the filename syntax or directory contents."
                )

            exp_cfg = OmegaConf.load(exp_path)
            if "defaults" in exp_cfg:
                for item in exp_cfg["defaults"]:
                    for key, val in item.items():
                        defaults_map[key] = val

        # 3. Sub-config 파일 로드
        merged_modules = OmegaConf.create()
        for key, filename in defaults_map.items():
            sub_path = os.path.join(self.config_dir, key, f"{filename}.yaml")
            if not os.path.exists(sub_path):
                raise FileNotFoundError(f"Sub-config file not found: {sub_path}")

            sub_cfg = OmegaConf.load(sub_path)
            sub_cfg["_config_source"] = f"configs/{key}/{filename}.yaml"
            merged_modules[key] = sub_cfg

        base_scalars = {k: v for k, v in base_cfg.items() if k != "defaults"}
        exp_scalars = {k: v for k, v in exp_cfg.items() if k != "defaults"}

        # 4. 베이스 구성 병합
        base_merged_cfg = OmegaConf.merge(merged_modules, base_scalars, exp_scalars)

        # [핵심 1] struct 플래그 활성화 (존재하지 않는 키 오버라이드 금지)
        OmegaConf.set_struct(base_merged_cfg, True)

        # [핵심 2] CLI 인자 중 메타 키인 'experiment'는 하이퍼파라미터 병합 대상에서 제외
        if "experiment" in cli_cfg:
            del cli_cfg["experiment"]

        # 5. CLI 오버라이드 병합 검증
        try:
            final_cfg = OmegaConf.merge(base_merged_cfg, cli_cfg)
        except (
            errors.ConfigAttributeError,
            errors.ConfigKeyError,
            errors.OmegaConfBaseException,
        ) as e:
            raise KeyError(
                f"\n[ConfigError] Invalid parameter override attempt!"
                f"\nYou tried to override a non-existent parameter key or invalid structure."
                f"\nDetails: {e}"
            ) from e

        # 6. 타임스탬프 기반 디렉토리 생성 및 저장
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_run_dir = os.path.join(final_cfg.logger.log_dir, timestamp)
        out_run_dir = os.path.join(final_cfg.persistence.output_dir, timestamp)

        os.makedirs(log_run_dir, exist_ok=True)
        os.makedirs(out_run_dir, exist_ok=True)

        save_path = os.path.join(log_run_dir, "resolved_config.yaml")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(final_cfg))

        # 7. Domain Dataclass로 인스턴스화
        return AppConfig(
            data=DataConfig(
                name=final_cfg.data.name,
                data_dir=final_cfg.data.data_dir,
                source_file=final_cfg.data.get("_config_source", ""),
            ),
            trainer=TrainerConfig(
                name=final_cfg.trainer.name,
                batch_size=final_cfg.trainer.batch_size,
                epochs=final_cfg.trainer.epochs,
                seed=final_cfg.trainer.seed,
                device=final_cfg.trainer.device,
                source_file=final_cfg.trainer.get("_config_source", ""),
            ),
            model=ModelConfig(
                name=final_cfg.model.name,
                input_dim=final_cfg.model.input_dim,
                hidden_dim=final_cfg.model.hidden_dim,
                output_dim=final_cfg.model.output_dim,
                source_file=final_cfg.model.get("_config_source", ""),
            ),
            optimizer=OptimizerConfig(
                name=final_cfg.optimizer.name,
                lr=final_cfg.optimizer.lr,
                source_file=final_cfg.optimizer.get("_config_source", ""),
            ),
            logger=LoggerConfig(
                name=final_cfg.logger.name,
                type=final_cfg.logger.type,
                log_dir=final_cfg.logger.log_dir,
                run_dir=log_run_dir,
                source_file=final_cfg.logger.get("_config_source", ""),
            ),
            persistence=PersistenceConfig(
                name=final_cfg.persistence.name,
                output_dir=final_cfg.persistence.output_dir,
                run_dir=out_run_dir,
                save_top_k=final_cfg.persistence.save_top_k,
                checkpoint_interval=final_cfg.persistence.checkpoint_interval,
                source_file=final_cfg.persistence.get("_config_source", ""),
            ),
        )
