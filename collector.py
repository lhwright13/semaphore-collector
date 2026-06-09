import time
import cv2
import numpy as np
from enum import Enum, auto
from models import ClipMetadata, CollectorConfig
from pose_extractor import PoseExtractor
from storage import ClipStorage
from prompts import load_prompts


class State(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    REVIEW = auto()


class Collector:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.pose = PoseExtractor()
        self.storage = ClipStorage(config)
        self.prompts = load_prompts(config.prompts_file)
        self.counts = self.storage.counts_by_label()

        self.state = State.WAITING
        self.prompt_idx = self._least_collected_idx()
        self.countdown_start = 0.0
        self.frames: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.cap = cv2.VideoCapture(config.camera_index)

        self.reference = self._load_reference()
        self.show_reference = self.reference is not None

        self.button_rect = None          # (x1, y1, x2, y2) of the Export button
        self._export_requested = False
        self._export_msg = None          # (text, expires_at)

    def _load_reference(self):
        img = cv2.imread(self.config.reference_image)
        if img is None:
            return None
        scale = self.config.reference_width / img.shape[1]
        return cv2.resize(img, (self.config.reference_width, int(img.shape[0] * scale)))

    def run(self):
        print("Semaphore Collector")
        print("SPACE start | e end | s save | r reject | n skip prompt | h toggle chart | z export zip | q quit")

        cv2.namedWindow("Collector")
        cv2.setMouseCallback("Collector", self._on_mouse)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ONE pose inference per frame, reused for drawing and recording.
            landmarks = self.pose.process(frame, draw=True)

            self._handle_state(landmarks)
            if self._export_requested:
                self._export_requested = False
                self._export()
            self._draw_ui(frame)
            cv2.imshow("Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
            if key == ord("q"):
                break

        self._cleanup()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.button_rect:
            x1, y1, x2, y2 = self.button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._export_requested = True

    def _export(self):
        path = self.storage.export_zip()
        if path is None:
            self._export_msg = ("Nothing to export yet", time.time() + 3)
            print("Nothing to export yet.")
        else:
            n = sum(self.counts.values())
            self._export_msg = (f"Exported {n} clips -> {path.name}", time.time() + 6)
            print(f"Exported {n} clips to {path}")

    def _handle_state(self, landmarks):
        if self.state == State.COUNTDOWN:
            if time.time() - self.countdown_start >= self.config.countdown_seconds:
                self.state = State.RECORDING
                self.frames = []
                self.timestamps = []

        elif self.state == State.RECORDING and landmarks is not None:
            self.frames.append(landmarks)
            self.timestamps.append(time.time())

    def _handle_key(self, key: int):
        if key == ord(" ") and self.state == State.WAITING:
            self.state = State.COUNTDOWN
            self.countdown_start = time.time()
        elif key == ord("e") and self.state == State.RECORDING:
            self._finish_recording()
        elif key == ord("s") and self.state == State.REVIEW:
            self._save_clip()
        elif key == ord("r") and self.state == State.REVIEW:
            print("Rejected.")
            self._reset()
        elif key == ord("n") and self.state != State.RECORDING:
            self.prompt_idx = (self.prompt_idx + 1) % len(self.prompts)
            self.state = State.WAITING
        elif key == ord("h"):
            self.show_reference = not self.show_reference
        elif key == ord("z"):
            self._export_requested = True

    def _finish_recording(self):
        # Trim by real elapsed time, not an assumed fps.
        if self.timestamps:
            cutoff = self.timestamps[-1] - self.config.trim_end_seconds
            keep = [i for i, t in enumerate(self.timestamps) if t <= cutoff]
            self.frames = [self.frames[i] for i in keep]
            self.timestamps = [self.timestamps[i] for i in keep]

        if len(self.frames) < 3:
            print("Too short, discarding.")
            self._reset()
            return
        self.state = State.REVIEW

    def _measured_fps(self) -> float:
        if len(self.timestamps) < 2:
            return float(self.config.fps)
        duration = self.timestamps[-1] - self.timestamps[0]
        return round(len(self.frames) / duration, 1) if duration > 0 else float(self.config.fps)

    def _save_clip(self):
        label, label_type = self.prompts[self.prompt_idx]
        clip_id = ClipStorage.new_clip_id()
        metadata = ClipMetadata(
            clip_id=clip_id,
            label=label,
            label_type=label_type,
            frame_count=len(self.frames),
            fps=self._measured_fps(),
        )
        self.storage.save(np.stack(self.frames), metadata)
        self.counts[label] += 1
        print(f"Saved {clip_id}: '{label}' ({len(self.frames)} frames @ {metadata.fps}fps)")
        self._reset()
        # Auto-advance to whatever you have the fewest examples of.
        self.prompt_idx = self._least_collected_idx()

    def _least_collected_idx(self) -> int:
        return min(range(len(self.prompts)), key=lambda i: self.counts[self.prompts[i][0]])

    def _reset(self):
        self.frames = []
        self.timestamps = []
        self.state = State.WAITING

    def _draw_reference(self, frame: cv2.Mat) -> int:
        """Overlay the semaphore chart in the top-left. Returns the y where
        text below it should start."""
        if not (self.show_reference and self.reference is not None):
            return 40
        rh, rw = self.reference.shape[:2]
        rh = min(rh, frame.shape[0])
        rw = min(rw, frame.shape[1])
        frame[0:rh, 0:rw] = self.reference[0:rh, 0:rw]
        cv2.rectangle(frame, (0, 0), (rw - 1, rh - 1), (255, 255, 255), 1)
        return rh + 30

    def _draw_ui(self, frame: cv2.Mat):
        h, w = frame.shape[:2]
        label, label_type = self.prompts[self.prompt_idx]
        n_have = self.counts[label]

        y = self._draw_reference(frame)

        cv2.putText(frame, f'Sign: "{label}" ({label_type})  [{n_have} saved]', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if self.state == State.WAITING:
            cv2.putText(frame, "SPACE to start", (10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        elif self.state == State.COUNTDOWN:
            n = int(time.time() - self.countdown_start) + 1
            cv2.putText(frame, f"{n} Mississippi...", (10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        elif self.state == State.RECORDING:
            cv2.putText(frame, f"RECORDING  {len(self.frames)} frames", (10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
        elif self.state == State.REVIEW:
            cv2.putText(frame, f"Save (s) or Reject (r)?  {len(self.frames)} frames", (10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        total = sum(self.counts.values())
        cv2.putText(frame, f"Total clips: {total}  |  (n)ext  (h)ide ref  (q)uit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        self._draw_export_button(frame, w, h)

    def _draw_export_button(self, frame: cv2.Mat, w: int, h: int):
        bw, bh = 150, 38
        x1, y1 = w - bw - 10, 10
        x2, y2 = x1 + bw, y1 + bh
        self.button_rect = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 140, 60), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 255, 180), 1)
        cv2.putText(frame, "Export ZIP (z)", (x1 + 10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self._export_msg and time.time() < self._export_msg[1]:
            cv2.putText(frame, self._export_msg[0], (x1 - 4, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

    def _cleanup(self):
        self.pose.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal clips saved: {sum(self.counts.values())}")


if __name__ == "__main__":
    Collector(CollectorConfig()).run()
