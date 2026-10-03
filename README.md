## 🚀 Quickstart: Configuration Management & Execution

본 프로젝트는 **Hexagonal Architecture (Ports & Adapters)** 원칙에 따라 설계되어, 도메인 코드 변경 없이 YAML 기반의 설정 조합 및 오버라이드를 지원합니다.

`configs/` 디렉터리 내의 Sub-config 모듈(`data`, `model`, `trainer`, `optimizer`, `logger`, `persistence`)을 부품처럼 조합할 수 있으며, **기본 베이스라인**, **실험(Experiment) 스펙**, **현장 CLI 인자** 세 가지 방식으로 실행이 가능합니다.

---

### 1. 기본 설정 (Baseline) 실행
별도의 옵션 없이 실행하면 `configs/main.yaml`에 정의된 베이스라인 조합으로 구동됩니다.

```bash
uv run python apps/quick_start/main.py
```

### 2. Experiment 파일 오버라이드 실행
configs/experiment/quickstart.yaml에 정의된 실험 단위를 불러와 특정 부품 모듈이나 하이퍼파라미터를 일괄 변경하여 실행합니다.

```bash
uv run python apps/quick_start/main.py experiment=quickstart
```

### 3. CLI 실시간 오버라이드 (Direct Override)
특정 하이퍼파라미터만 터미널에서 즉시 변경하여 실행할 수 있습니다.

```bash
uv run python apps/quick_start/main.py optimizer.lr=0.01 trainer.batch_size=256
```