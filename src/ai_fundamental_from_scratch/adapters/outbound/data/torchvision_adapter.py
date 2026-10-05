# src/ai_fundamental_from_scratch/adapters/outbound/data/torchvision_adapter.py
import inspect
import re
from typing import Dict, Type

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ai_fundamental_from_scratch.domain.config import DataConfig, TrainerConfig
from ai_fundamental_from_scratch.ports.outbound.data_port import DataLoaderPort


def _get_torchvision_datasets() -> Dict[str, Type[Dataset]]:
    """torchvision.datasets 내 모든 Dataset 클래스를 다각도 키(snake_case, lowercase 등)로 수집"""
    dataset_map = {}
    for name, obj in inspect.getmembers(datasets, inspect.isclass):
        if (
            issubclass(obj, Dataset)
            or "Dataset" in name
            or name in ("MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "ImageNet")
        ):
            # 1. 원본 소문자 (fashionmnist)
            raw_lower = name.lower()
            dataset_map[raw_lower] = obj

            # 2. CamelCase -> snake_case 변환 (FashionMNIST -> fashion_mnist)
            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            dataset_map[snake_case] = obj

            # 3. 언더스코어 제거 버전 (fashionmnist)
            dataset_map[snake_case.replace("_", "")] = obj

    return dataset_map


class TorchStandardDataAdapter(DataLoaderPort):
    _DATASET_MAP: Dict[str, Type[Dataset]] = _get_torchvision_datasets()

    def __init__(self, dataset_name: str):
        normalized_name = dataset_name.lower().replace("-", "_").replace("_", "")

        # 키 탐색
        matched_cls = None
        for key, cls_obj in self._DATASET_MAP.items():
            if key.replace("_", "") == normalized_name:
                matched_cls = cls_obj
                break

        if not matched_cls:
            raise ValueError(
                f"[DataError] Unsupported torchvision dataset: '{dataset_name}'."
            )

        self.dataset_cls: Type[Dataset] = matched_cls

    @classmethod
    def is_supported(cls, dataset_name: str) -> bool:
        normalized_name = dataset_name.lower().replace("-", "_").replace("_", "")
        for key in cls._DATASET_MAP.keys():
            if key.replace("_", "") == normalized_name:
                return True
        return False

    def get_data_loaders(self, data_config: DataConfig, trainer_config: TrainerConfig):
        # image_shape 기반 Transform 구성
        transform_list = []
        if data_config.image_shape and len(data_config.image_shape) == 3:
            h, w, _ = data_config.image_shape
            transform_list.append(transforms.Resize((h, w)))

        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        train_ds = self.dataset_cls(
            root=data_config.data_dir, train=True, download=True, transform=transform
        )
        test_ds = self.dataset_cls(
            root=data_config.data_dir, train=False, download=True, transform=transform
        )

        return (
            DataLoader(train_ds, batch_size=trainer_config.batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=trainer_config.batch_size, shuffle=False),
        )
