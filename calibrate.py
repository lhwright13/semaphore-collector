"""Calibrate per-letter pose prototypes from your real letter clips.

A single-letter clip is home -> letter -> home. The held letter pose is the
stillest stretch in the middle (between the home-position transitions). We
average that across takes to get one prototype pose per letter - elbow and wrist
included, exactly as you sign it. synth.py then animates between YOUR real poses
instead of a canonical stand-in, which captures however you bend your arms.

  python calibrate.py        build prototypes.npz from single-letter clips
"""
import glob
import json
from collections import defaultdict

import numpy as np

from features import normalize_sequence

DATA_DIR = "data/clips"
HELD_WINDOW = 5      # frames averaged for the held pose
EDGE_SKIP = 0.15     # ignore first/last 15% of a clip (home transitions)
HOME_FRAMES = 5      # leading frames assumed to be the home position


def held_pose(feat):
    """Average of the stillest HELD_WINDOW frames in the clip's middle."""
    T = len(feat)
    if T <= HELD_WINDOW:
        return feat.mean(0)
    vel = np.linalg.norm(np.diff(feat, axis=0), axis=1)  # per-frame motion
    lo, hi = int(T * EDGE_SKIP), T - int(T * EDGE_SKIP)
    best_i, best_v = lo, np.inf
    for i in range(lo, max(lo + 1, hi - HELD_WINDOW)):
        v = vel[i:i + HELD_WINDOW].mean()
        if v < best_v:
            best_v, best_i = v, i
    return feat[best_i:best_i + HELD_WINDOW].mean(0)


def main():
    by_letter = defaultdict(list)
    homes = []
    for npy in sorted(glob.glob(f"{DATA_DIR}/*.npy")):
        meta = json.load(open(npy.replace(".npy", ".json")))
        feat = normalize_sequence(np.load(npy).astype(np.float32))
        if len(feat) >= HOME_FRAMES:
            homes.append(feat[:HOME_FRAMES].mean(0))
        if len(meta["label"]) == 1:                 # a single-letter clip
            by_letter[meta["label"]].append(held_pose(feat))

    if not by_letter:
        print("No single-letter clips yet. Collect letters with collector.py, then re-run.")
        return

    letters = sorted(by_letter)
    poses = np.stack([np.mean(by_letter[ch], 0) for ch in letters] +
                     [np.mean(homes, 0)])
    names = letters + ["_home"]
    np.savez("prototypes.npz", names=np.array(names), poses=poses)

    print(f"Calibrated {len(letters)} letters from real clips -> prototypes.npz")
    for ch in letters:
        print(f"  {ch}: {len(by_letter[ch])} take(s)")
    missing = sorted(set("abcdefghijklmnopqrstuvwxyz") - set(letters))
    if missing:
        print(f"\nStill need letter clips for: {' '.join(missing)}")


if __name__ == "__main__":
    main()
