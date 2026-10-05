from ai_fundamental_from_scratch.adapters.outbound.data.animals_10_adapter import (
    Animals10Adapter,
)
from ai_fundamental_from_scratch.adapters.outbound.data.torchvision_adapter import (
    TorchStandardDataAdapter,
)
from ai_fundamental_from_scratch.ports.outbound.data_port import DataLoaderPort


class DataAdapterFactory:
    @classmethod
    def create_adapter(cls, dataset_name: str) -> DataLoaderPort:
        normalized_name = dataset_name.lower().replace("-", "_")

        # 1. torchvision 내장 데이터셋 동적 분기
        if TorchStandardDataAdapter.is_supported(normalized_name):
            return TorchStandardDataAdapter(dataset_name=normalized_name)

        # 2. Animals10 어댑터 분기
        if normalized_name == "animals_10":
            return Animals10Adapter()

        # 3. 지원하지 않는 데이터셋 예외 처리
        raise ValueError(
            f"[DataError] Unknown or unsupported dataset '{dataset_name}'. "
            f"Custom datasets supported: ['animals_10']"
        )
