"""Offline voice command listener for hands-free collection.

Uses Vosk (a small, local speech model - no internet once the model is fetched)
to spot a short list of command words from the microphone. Restricting the
grammar to just the commands makes recognition fast and reliable.

Optional: if vosk/sounddevice are not installed, the collector falls back to
keyboard-only. Install with `pip install -r requirements-voice.txt`.
"""
import json
import shutil
import ssl
import threading
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = Path("models/vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000


class VoiceListener:
    def __init__(self, commands, on_command):
        self.commands = [c.lower() for c in commands]
        self.on_command = on_command
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def available() -> bool:
        try:
            import vosk          # noqa: F401
            import sounddevice   # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_model(self) -> bool:
        if MODEL_DIR.exists():
            return True
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        zip_path = MODEL_DIR.parent / "_vosk_model.zip"
        print("Downloading voice model (~40 MB, one time)...")
        # macOS python.org builds ship without root certs, so verify via certifi.
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MODEL_DIR.parent)
        zip_path.unlink(missing_ok=True)
        return MODEL_DIR.exists()

    def start(self) -> bool:
        """Begin listening in a background thread. Returns False if unavailable."""
        if not self.available():
            return False
        try:
            if not self._ensure_model():
                return False
        except Exception as e:
            print(f"Voice model download failed: {e}")
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        import queue
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer

        model = Model(str(MODEL_DIR))
        grammar = json.dumps(self.commands + ["[unk]"])
        rec = KaldiRecognizer(model, SAMPLE_RATE, grammar)

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16",
                               channels=1, callback=callback):
            while not self._stop.is_set():
                data = audio_q.get()
                if rec.AcceptWaveform(data):
                    words = json.loads(rec.Result()).get("text", "").split()
                    for w in words:
                        if w in self.commands:
                            self.on_command(w)

    def stop(self):
        self._stop.set()
