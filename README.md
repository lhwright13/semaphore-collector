# Semaphore Data Collector

A small webcam tool for recording semaphore (flag-signaling) arm movements. It
tracks your arms with computer vision and saves the motion as compact data files
(no video, just joint positions). Recordings are used to train a model that reads
semaphore from a camera.

If a friend sent you here: thank you for helping. Setup takes a few minutes, then
you sign the prompts shown on screen and send back one zip file.

## What you need

- A webcam
- Python 3.10 specifically. The pose-tracking library (mediapipe) does not yet
  have working installers for Python 3.13 or 3.14, so 3.10 is required.

## Setup

Clone the repo, then run the setup script for your system.

macOS / Linux:

```bash
git clone https://github.com/lhwright13/semaphore-collector.git
cd semaphore-collector
./setup.sh
```

Windows:

```bat
git clone https://github.com/lhwright13/semaphore-collector.git
cd semaphore-collector
setup.bat
```

This creates a local `venv` and installs the four dependencies. Nothing is
installed globally.

## Recording

Start the collector:

```bash
source venv/bin/activate      # Windows: venv\Scripts\activate.bat
python collector.py
```

A camera window opens with a semaphore chart in the corner for reference. For
each prompt shown ("hello", the letter "r", etc.):

1. Press `SPACE`. A short countdown runs ("1 Mississippi, 2 Mississippi").
2. Sign the prompt, matching the chart. Hold the final position briefly.
3. Press `e` to end. The last second is trimmed automatically.
4. Press `s` to save, or `r` to reject and redo.

It automatically moves to whichever prompt has the fewest recordings, so the data
stays balanced. A few takes of each prompt, at slightly different speeds, is ideal.

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | start recording (after countdown) |
| `e` | end recording |
| `s` | save the take |
| `r` | reject and redo |
| `n` | skip to the next prompt |
| `h` | hide/show the reference chart |
| `z` | export all recordings to a zip |
| `q` | quit |

Keep your hands inside the frame while signing - if they leave the view, the
tracking guesses and the data is less useful.

### Hands-free voice control (optional)

To avoid reaching for the keyboard between every take, you can cue collection by
voice. Install the extra dependencies once:

```bash
pip install -r requirements-voice.txt
```

The first run downloads a small offline speech model (~40 MB). After that, just
say **"next"** to save the current take, advance to the next prompt, and start
the next countdown automatically - a continuous loop. All keys still work too.
(On macOS the terminal will ask for microphone permission the first time.)

## Sending your data back

Click the green **Export ZIP** button (or press `z`). This writes a file named
`semaphore_data_<date>.zip` into the project folder. Send that one file back.

Send the zip to: **(add your contact here)**

## For the maintainer

The training pipeline (synthetic data generation, model, calibration) lives in
the same repo. Install its extra dependency with:

```bash
pip install -r requirements-training.txt
```

- `synth.py` - synthetic pose generator (see `data_synthesis.html` for how it works)
- `features.py` - landmark normalization
- `model.py` / `train.py` - CNN + transformer recognizer
- `calibrate.py` - build per-letter prototypes from real clips
- `qc.py` - coverage report and clip replay
