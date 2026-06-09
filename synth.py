"""Synthetic semaphore pose generator for pretraining.

Produces fake clips in the SAME raw [T, 132] MediaPipe layout that the collector
saves, so synthetic and real data are interchangeable everywhere downstream.

Important honesty about the angle table below: it is a *structured stand-in*,
not verified canonical semaphore. For pretraining that is fine - the value is
teaching the model the lower-level temporal machinery (how a stream of poses
turns into letters over time) on cheap, unlimited, well-separated examples. Your
real clips, fine-tuned on top, define your actual gesture vocabulary. We also
bias the synthetic motions toward your style (smaller arm extension, hands
returning to the belly on a space) to keep the transfer positive.
"""
import numpy as np

# Arm direction in degrees, measured from straight-down, increasing as the arm
# swings outward to the side (90) and up (180).
POSITIONS = [-45, 0, 45, 90, 135, 180]

# 26 distinct (left_angle, right_angle) pairs, one per letter. (0, 0) = both
# arms home, reserved for space. Deterministic enumeration -> stable labels.
_PAIRS = [(l, r) for l in POSITIONS for r in POSITIONS if not (l == 0 and r == 0)]
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
CANON = {ch: _PAIRS[i] for i, ch in enumerate(ALPHABET)}

# Fixed body layout in image-like (0..1) coordinates, mirrored view to match the
# collector's flipped frames.
MID_X, SHOULDER_Y, HALF_W = 0.5, 0.40, 0.12
NOSE = (0.5, 0.22)
HIPS = {23: (0.40, 0.75), 24: (0.60, 0.75)}
BELLY = (0.5, 0.62)
ARM_LEN = 0.30


def _arm_points(shoulder, side, angle_deg, arm_len, bend=0.0):
    """Return (elbow, wrist) for an arm at the given angle.
    side = -1 for the left arm (outward = image-left), +1 for the right.
    bend offsets the elbow perpendicular to the arm, so arms aren't perfectly
    straight (a proportion that survives normalization)."""
    a = np.radians(angle_deg)
    direction = np.array([side * np.sin(a), np.cos(a)])  # y is down in image coords
    perp = np.array([-direction[1], direction[0]])
    sx = np.array(shoulder)
    return sx + 0.5 * arm_len * direction + bend * perp, sx + arm_len * direction


def _frame(left_angle, right_angle, is_space, arm_len, hip_y=0.75, bend=0.0):
    """Build one raw [132] frame for a target pose."""
    f = np.zeros(33 * 4, dtype=np.float32)

    def put(idx, xy, vis=1.0):
        f[idx * 4 : idx * 4 + 4] = [xy[0], xy[1], 0.0, vis]

    l_sh = (MID_X - HALF_W, SHOULDER_Y)
    r_sh = (MID_X + HALF_W, SHOULDER_Y)
    put(0, NOSE)
    put(11, l_sh)
    put(12, r_sh)
    put(23, (0.40, hip_y))
    put(24, (0.60, hip_y))

    if is_space:  # hands come home to the belly button
        l_el, l_wr = ((np.array(l_sh) + BELLY) / 2, np.array(BELLY))
        r_el, r_wr = ((np.array(r_sh) + BELLY) / 2, np.array(BELLY))
    else:
        l_el, l_wr = _arm_points(l_sh, -1, left_angle, arm_len, bend)
        r_el, r_wr = _arm_points(r_sh, +1, right_angle, arm_len, bend)

    put(13, l_el); put(15, l_wr)
    put(14, r_el); put(16, r_wr)
    # Hand landmarks (pinky/index/thumb) sit at the wrist for simplicity.
    for idx in (17, 19, 21):
        put(idx, l_wr)
    for idx in (18, 20, 22):
        put(idx, r_wr)
    return f


def _targets(text):
    """Yield (left, right, is_space) poses for each character, home->...->home."""
    yield 0.0, 0.0, True  # start at home
    for ch in text.lower():
        if ch == " " or ch not in CANON:
            yield 0.0, 0.0, True
        else:
            l, r = CANON[ch]
            yield float(l), float(r), False
    yield 0.0, 0.0, True  # end at home


def _augment(arr, rng, jitter):
    """Apply augmentations that SURVIVE mid-shoulder/shoulder-width normalization:
    rotation, low-frequency drift, per-joint jitter, and wrist occlusion. (Global
    translation and scale are deliberately omitted - normalization cancels them.)"""
    xyz = arr.reshape(len(arr), 33, 4)
    mask = xyz[..., 3] > 0
    pts = xyz[..., :2]

    # Camera tilt / body lean: rotate every frame about the mid-shoulder point.
    theta = rng.normal(0, np.radians(7))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = np.array([MID_X, SHOULDER_Y], dtype=np.float32)
    pts[:] = (pts - center) @ R.T + center

    # Slow body sway (low-frequency random walk shared by all joints).
    drift = np.cumsum(rng.normal(0, 0.0015, size=(len(arr), 1, 2)), axis=0).astype(np.float32)
    pts += drift

    # Per-joint, per-frame tracking jitter.
    pts += rng.normal(0, jitter, size=pts.shape).astype(np.float32)

    # Occlusion: a wrist+hand drops tracking and freezes for a span (~30% of clips),
    # mimicking MediaPipe losing a hand that leaves the frame.
    if rng.random() < 0.3 and len(arr) > 5:
        group = [(15, 17, 19, 21), (16, 18, 20, 22)][int(rng.integers(2))]
        t0 = int(rng.integers(0, len(arr) - 3))
        t1 = min(len(arr), t0 + int(rng.integers(3, 12)))
        for idx in group:
            pts[t0:t1, idx, :] = pts[t0, idx, :]   # freeze at onset

    xyz[..., :3] *= mask[..., None]         # keep unused joints at zero
    return arr.astype(np.float32)


def generate_clip(text, rng, jitter=0.012):
    """Return (raw [T, 132], text). Interpolates between target poses with random
    speed/hold, per-pose angle noise, body-proportion variation, and augmentation."""
    arm_len = ARM_LEN * rng.uniform(0.40, 1.05)   # compact (you) .. full extension
    hip_y = rng.uniform(0.70, 0.83)
    bend = rng.uniform(0.0, 0.05)
    angle_bias = rng.normal(0, 6, size=2)         # per-clip systematic offset (deg)

    def build(tgt):
        l, r, is_space = tgt
        if not is_space:                          # imprecise human signing
            l += angle_bias[0] + rng.normal(0, 5)
            r += angle_bias[1] + rng.normal(0, 5)
        return _frame(l, r, is_space, arm_len, hip_y, bend)

    targets = list(_targets(text))
    frames = []
    prev = build(targets[0])
    for tgt in targets[1:]:
        cur = build(tgt)
        move = rng.integers(6, 18)                # transition length -> speed
        hold = rng.integers(2, 9)
        for k in range(1, move + 1):
            t = k / move
            t = t * t * (3 - 2 * t)               # smoothstep -> cursive easing
            frames.append((1 - t) * prev + t * cur)
        frames.extend([cur] * hold)
        prev = cur

    return _augment(np.stack(frames), rng, jitter), text


def stream(labels, rng, jitter=0.005):
    """Infinite generator of (raw_clip, text), labels chosen uniformly at random."""
    labels = list(labels)
    while True:
        text = labels[rng.integers(len(labels))]
        yield generate_clip(text, rng, jitter)
