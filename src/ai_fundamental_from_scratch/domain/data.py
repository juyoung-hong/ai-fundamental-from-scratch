from typing import List, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    name: str = Field(..., description="데이터셋 이름")
    data_dir: str = Field(..., description="데이터 저장 경로")
    modality: str = Field(..., description="데이터 모달리티 (예: image)")
    image_shape: Optional[List[int]] = Field(
        ..., description="이미지 크기 [Height, Width, Channels]"
    )
    num_classes: Optional[int] = Field(..., description="클래스 개수")
