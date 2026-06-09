from typing import Optional
from pydantic import BaseModel, Field
import time


class ClipMetadata(BaseModel):
    clip_id: str
    label: str
    label_type: str = Field(..., description="letter, word, or sentence")
    frame_count: int
    fps: float
    recorded_at: float = Field(default_factory=time.time)
    notes: Optional[str] = None


class CollectorConfig(BaseModel):
    countdown_seconds: int = 2
    trim_end_seconds: float = 1.0
    camera_index: int = 0
    fps: int = 30
    data_dir: str = "data"
    prompts_file: str = "prompts.txt"
    reference_image: str = "semaphore_chart.png"
    reference_width: int = 360
