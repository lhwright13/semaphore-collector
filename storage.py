import json
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from models import ClipMetadata, CollectorConfig


class ClipStorage:
    def __init__(self, config: CollectorConfig):
        self.data_dir = Path(config.data_dir)
        self.clips_dir = self.data_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frames: np.ndarray, metadata: ClipMetadata) -> str:
        """Save landmark frames (as float16) and metadata. Returns clip_id."""
        clip_id = metadata.clip_id
        np.save(self.clips_dir / f"{clip_id}.npy", frames.astype(np.float16))
        with open(self.clips_dir / f"{clip_id}.json", "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        return clip_id

    def list_clips(self) -> list[ClipMetadata]:
        clips = []
        for path in sorted(self.clips_dir.glob("*.json")):
            with open(path) as f:
                clips.append(ClipMetadata(**json.load(f)))
        return clips

    def counts_by_label(self) -> Counter:
        """How many clips exist per label, for keeping the dataset balanced."""
        return Counter(c.label for c in self.list_clips())

    def load_frames(self, clip_id: str) -> np.ndarray:
        """Load frames as float32 for downstream math."""
        return np.load(self.clips_dir / f"{clip_id}.npy").astype(np.float32)

    def export_zip(self, out_dir: str = ".") -> Path:
        """Bundle every saved clip into a timestamped zip to send back. Returns
        the zip path, or None if there are no clips."""
        files = sorted(self.clips_dir.glob("*"))
        if not files:
            return None
        out_path = Path(out_dir) / f"semaphore_data_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, arcname=f"clips/{f.name}")
        return out_path

    @staticmethod
    def new_clip_id() -> str:
        return str(uuid.uuid4())[:8]
