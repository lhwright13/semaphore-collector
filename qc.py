"""Data quality + coverage tool for collected clips.

  python qc.py             coverage report + quality warnings
  python qc.py <clip_id>   replay one clip as a skeleton animation

Quality matters: if your wrists leave the frame while signing, MediaPipe guesses
their position with low confidence, and those clips teach the model garbage. This
flags them so you can re-record.
"""
import sys
import glob
import json
from pathlib import Path
from collections import Counter

import numpy as np

from prompts import load_prompts

DATA_DIR = "data/clips"
WRISTS = (15, 16)           # left, right wrist landmark indices
MIN_FRAMES = 5
MIN_WRIST_VIS = 0.5         # mean wrist visibility below this = likely out of frame

# Skeleton edges among the landmarks we care about (MediaPipe indices).
EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # shoulders + arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (11, 23), (12, 24), (23, 24),                          # torso
]


def _clips():
    """Yield (clip_id, metadata_dict, npy_path) for every saved clip."""
    for npy_path in sorted(glob.glob(f"{DATA_DIR}/*.npy")):
        meta_path = npy_path.replace(".npy", ".json")
        if Path(meta_path).exists():
            yield Path(npy_path).stem, json.load(open(meta_path)), npy_path


def report():
    clips = list(_clips())
    if not clips:
        print("No clips yet. Run collector.py to record some.")
        return

    counts = Counter(m["label"] for _, m, _ in clips)
    prompts = [label for label, _ in load_prompts()]

    print(f"\n{len(clips)} clips across {len(counts)} labels\n")
    print("  count  label")
    print("  -----  -----")
    for label in prompts:
        n = counts.get(label, 0)
        bar = "#" * min(n, 30)
        flag = "" if n else "  <- none yet"
        print(f"  {n:5d}  {label:<14} {bar}{flag}")
    # Any labels on disk that aren't in prompts.txt anymore
    for label in sorted(set(counts) - set(prompts)):
        print(f"  {counts[label]:5d}  {label:<14} (not in prompts.txt)")

    # Quality pass
    warnings = []
    for clip_id, meta, npy_path in clips:
        raw = np.load(npy_path).astype(np.float32).reshape(-1, 33, 4)
        if len(raw) < MIN_FRAMES:
            warnings.append(f"  {clip_id} ('{meta['label']}'): only {len(raw)} frames")
        wrist_vis = raw[:, WRISTS, 3].mean()
        if wrist_vis < MIN_WRIST_VIS:
            warnings.append(f"  {clip_id} ('{meta['label']}'): low wrist visibility "
                            f"{wrist_vis:.2f} - hands may have left the frame")

    print(f"\nQuality: {len(warnings)} warning(s)")
    for w in warnings:
        print(w)
    if not warnings:
        print("  all clips look clean")


def replay(clip_id):
    import cv2
    npy_path = f"{DATA_DIR}/{clip_id}.npy"
    if not Path(npy_path).exists():
        print(f"No clip {clip_id}")
        return
    meta = json.load(open(f"{DATA_DIR}/{clip_id}.json"))
    raw = np.load(npy_path).astype(np.float32).reshape(-1, 33, 4)
    W, H = 640, 480
    fps = meta.get("fps", 25)
    print(f"Replaying '{meta['label']}' - {len(raw)} frames @ {fps}fps. Press q to stop.")

    for f in raw:
        canvas = np.full((H, W, 3), 30, np.uint8)
        pts = [(int(x * W), int(y * H)) for x, y, _, _ in f]
        for a, b in EDGES:
            cv2.line(canvas, pts[a], pts[b], (0, 200, 255), 2)
        for idx in (0, 11, 12, 13, 14, 15, 16):
            cv2.circle(canvas, pts[idx], 4, (255, 255, 255), -1)
        cv2.putText(canvas, meta["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("clip", canvas)
        if cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        replay(sys.argv[1])
    else:
        report()
