"""Turn raw MediaPipe landmarks into a normalized, scale/position-invariant
feature sequence - the universal first step for skeleton-based recognition.

Raw landmarks are in image coordinates, so they change with where you stand,
how big you are in frame, and how far you are from the camera. We fix that by,
per frame:
  - centering on the mid-shoulder point (removes position)
  - dividing by shoulder width (removes body size / camera distance)

We also keep only the upper-body + hand landmarks that semaphore actually uses,
dropping the face mesh and legs. Both real clips and synthetic clips go through
this exact function, so the model never sees a distribution mismatch from
preprocessing.
"""
import numpy as np


# MediaPipe Pose landmark indices we keep (nose, shoulders, elbows, wrists,
# hands, hips). Hips give a stable lower reference; hands matter for your
# small/hand-based style.
SELECTED = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
L_SHOULDER, R_SHOULDER = 11, 12

FEATURE_DIM = len(SELECTED) * 3  # x, y, z per kept landmark = 45


def normalize_sequence(raw: np.ndarray) -> np.ndarray:
    """raw: [T, 132] flat MediaPipe frames -> [T, FEATURE_DIM] normalized."""
    raw = np.asarray(raw, dtype=np.float32)
    T = raw.shape[0]
    pts = raw.reshape(T, 33, 4)[:, :, :3]  # drop visibility -> [T, 33, 3]

    origin = (pts[:, L_SHOULDER] + pts[:, R_SHOULDER]) / 2.0  # [T, 3]
    width = np.linalg.norm(pts[:, L_SHOULDER, :2] - pts[:, R_SHOULDER, :2], axis=1)  # [T]
    width = np.where(width < 1e-6, 1e-6, width)[:, None, None]  # guard, broadcast

    sel = pts[:, SELECTED, :]  # [T, 15, 3]
    normed = (sel - origin[:, None, :]) / width
    return normed.reshape(T, FEATURE_DIM).astype(np.float32)
