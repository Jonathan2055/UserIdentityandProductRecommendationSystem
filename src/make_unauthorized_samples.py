"""Create fallback 'stranger' samples for the unauthorized-attempt demo.

Best practice: put a REAL non-member photo/recording (a friend who agreed) into
    data/images/unauthorized/   and   data/audio/unauthorized/
This script only fills those folders if they are empty:
  * image: the public-domain NASA astronaut portrait that ships with scikit-image
  * audio: a synthetic voice-like clip (pitch ~140 Hz with harmonics and syllable rhythm)

    python src/make_unauthorized_samples.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from skimage import data

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def synthetic_voice(seconds: float = 1.6, f0: float = 140.0, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    pitch = f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))                  # natural vibrato
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 12))                  # harmonics
    y *= 0.5 * (1 - np.cos(2 * np.pi * 3 * t))                             # ~3 syllables per second
    y += 0.02 * np.random.default_rng(0).normal(size=t.size)               # breath noise
    return (0.9 * y / np.abs(y).max()).astype(np.float32)


def main():
    C.IMG_UNAUTH_DIR.mkdir(parents=True, exist_ok=True)
    C.AUDIO_UNAUTH_DIR.mkdir(parents=True, exist_ok=True)

    if not any(p.suffix.lower() in C.IMAGE_EXTS for p in C.IMG_UNAUTH_DIR.iterdir()):
        out = C.IMG_UNAUTH_DIR / "stranger_astronaut.jpg"
        Image.fromarray(data.astronaut()).save(out, quality=95)
        print("created", out)
    else:
        print("images/unauthorized already has a sample - left untouched")

    if not any(p.suffix.lower() in C.AUDIO_EXTS for p in C.AUDIO_UNAUTH_DIR.iterdir()):
        out = C.AUDIO_UNAUTH_DIR / "stranger_synthetic.wav"
        sf.write(out, synthetic_voice(), C.SAMPLE_RATE)
        print("created", out)
    else:
        print("audio/unauthorized already has a sample - left untouched")


if __name__ == "__main__":
    main()
