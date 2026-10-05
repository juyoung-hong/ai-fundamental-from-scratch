# src/ai_fundamental_from_scratch/adapters/outbound/data/custom_image_adapter.py
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ai_fundamental_from_scratch.domain.config import DataConfig, TrainerConfig
from ai_fundamental_from_scratch.ports.outbound.data_port import DataLoaderPort


class Animals10Adapter(DataLoaderPort):
    def get_data_loaders(self, data_config: DataConfig, trainer_config: TrainerConfig):
        if not os.path.exists(data_config.data_dir):
            raise FileNotFoundError(
                f"[DataError] Image folder directory '{data_config.data_dir}' not found."
            )

        # 1. image_shape: [H, W, C] 파싱
        transform_list = []
        if data_config.image_shape and len(data_config.image_shape) == 3:
            h, w, c = data_config.image_shape
            transform_list.append(transforms.Resize((h, w)))

            # 1채널 흑백 요청 시 Grayscale 변환
            if c == 1:
                transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            transform_list.append(transforms.Resize((28, 28)))

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        [0.485, 0.456, 0.406]
                        if (data_config.image_shape and data_config.image_shape[2] == 3)
                        else [0.5]
                    ),
                    std=(
                        [0.229, 0.224, 0.225]
                        if (data_config.image_shape and data_config.image_shape[2] == 3)
                        else [0.5]
                    ),
                ),
            ]
        )

        transform = transforms.Compose(transform_list)

        # 2. ImageFolder 데이터셋 로드
        full_dataset = datasets.ImageFolder(
            root=data_config.data_dir, transform=transform
        )

        # 3. Train / Test 분할 및 Loader 반환
        train_count = int(len(full_dataset) * 0.8)
        test_count = len(full_dataset) - train_count

        generator = torch.Generator().manual_seed(trainer_config.seed)
        train_ds, test_ds = random_split(
            full_dataset, [train_count, test_count], generator=generator
        )

        return (
            DataLoader(train_ds, batch_size=trainer_config.batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=trainer_config.batch_size, shuffle=False),
        )
