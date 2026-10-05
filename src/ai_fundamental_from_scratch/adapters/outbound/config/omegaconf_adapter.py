import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf, errors

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
        exp_name = cli_cfg.pop("experiment", None)  # 메타 키 제거 및 추출

        # 1. Defaults 맵 구성 (main.yaml + experiment)
        base_cfg, exp_cfg, defaults_map = self._resolve_defaults(exp_name, cli_cfg)

        # 2. Sub-config 로드 및 출처 정보 표기
        merged_modules = self._load_sub_configs(defaults_map)

        # 3. Base/Exp 스칼라 및 CLI 오버라이드 검증 병합
        final_cfg = self._merge_and_validate(base_cfg, exp_cfg, merged_modules, cli_cfg)

        # 4. 실행 디렉토리 생성 및 Final YAML 저장
        log_run_dir, out_run_dir = self._save_resolved_config(final_cfg)

        # 5. Domain Dataclass 매핑 및 반환
        return self._to_domain_config(final_cfg, log_run_dir, out_run_dir)

    # ------------------------------------------------------------------
    # Private Helper Methods (단일 책임 분리)
    # ------------------------------------------------------------------

    def _resolve_defaults(
        self, exp_name: Optional[str], cli_cfg: DictConfig
    ) -> Tuple[DictConfig, DictConfig, Dict[str, str]]:
        main_path = os.path.join(self.config_dir, "main.yaml")
        if not os.path.exists(main_path):
            raise FileNotFoundError(f"Main config file not found: {main_path}")

        base_cfg = OmegaConf.load(main_path)
        defaults_map: Dict[str, str] = {}

        for item in base_cfg.get("defaults", []):
            defaults_map.update(item)

        exp_cfg = OmegaConf.create()
        if exp_name:
            exp_path = os.path.join(self.config_dir, "experiment", f"{exp_name}.yaml")
            if not os.path.exists(exp_path):
                raise FileNotFoundError(
                    f"\n[ConfigError] Specified experiment file '{exp_name}.yaml' does not exist in '{os.path.dirname(exp_path)}'."
                )
            exp_cfg = OmegaConf.load(exp_path)
            for item in exp_cfg.get("defaults", []):
                defaults_map.update(item)

        # [핵심 추가] CLI에서 들어온 모듈 그룹 오버라이드(e.g., data=animals_10)를 defaults_map에 반영
        for group_key in list(defaults_map.keys()):
            if group_key in cli_cfg and isinstance(cli_cfg[group_key], str):
                defaults_map[group_key] = cli_cfg.pop(
                    group_key
                )  # defaults 맵을 바꾸고 cli_cfg에서는 제거

        return base_cfg, exp_cfg, defaults_map

    def _load_sub_configs(self, defaults_map: Dict[str, str]) -> DictConfig:
        """defaults 맵에 명시된 서브 모듈 파일들을 로드하고 출처 트래킹 정보 삽입"""
        merged_modules = OmegaConf.create()

        for key, filename in defaults_map.items():
            sub_path = os.path.join(self.config_dir, key, f"{filename}.yaml")
            if not os.path.exists(sub_path):
                raise FileNotFoundError(f"Sub-config file not found: {sub_path}")

            sub_cfg = OmegaConf.load(sub_path)
            sub_cfg["_config_source"] = f"configs/{key}/{filename}.yaml"
            merged_modules[key] = sub_cfg

        return merged_modules

    def _merge_and_validate(
        self,
        base_cfg: DictConfig,
        exp_cfg: DictConfig,
        merged_modules: DictConfig,
        cli_cfg: DictConfig,
    ) -> DictConfig:
        """설정 병합 및 struct 모드를 통한 무효 키 오버라이드 차단 검증"""
        base_scalars = {k: v for k, v in base_cfg.items() if k != "defaults"}
        exp_scalars = {k: v for k, v in exp_cfg.items() if k != "defaults"}

        base_merged_cfg = OmegaConf.merge(merged_modules, base_scalars, exp_scalars)

        # 존재하지 않는 키 오버라이드 금지
        OmegaConf.set_struct(base_merged_cfg, True)

        try:
            return OmegaConf.merge(base_merged_cfg, cli_cfg)
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

    def _save_resolved_config(self, final_cfg: DictConfig) -> Tuple[str, str]:
        """타임스탬프 실행 폴더 생성 및 최종 확정된 config.yaml 파일 보관"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_run_dir = os.path.join(final_cfg.logger.log_dir, timestamp)
        out_run_dir = os.path.join(final_cfg.persistence.output_dir, timestamp)

        os.makedirs(log_run_dir, exist_ok=True)
        os.makedirs(out_run_dir, exist_ok=True)

        save_path = os.path.join(log_run_dir, "resolved_config.yaml")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(final_cfg))

        return log_run_dir, out_run_dir

    def _to_domain_config(
        self, final_cfg: DictConfig, log_run_dir: str, out_run_dir: str
    ) -> AppConfig:
        """OmegaConf 객체를 pure Python Domain Dataclasses로 매핑"""
        return AppConfig(
            data=DataConfig(
                name=final_cfg.data.name,
                data_dir=final_cfg.data.data_dir,
                modality=final_cfg.data.get("modality", "image"),
                image_shape=(
                    list(final_cfg.data.image_shape)
                    if getattr(final_cfg.data, "image_shape", None)
                    else None
                ),
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
