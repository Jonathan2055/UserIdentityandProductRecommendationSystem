
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "audio" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "audio" / "processed"
AUGMENTED_DIR = PROJECT_DIR / "data" / "audio" / "augmented"

SAMPLE_RATE = 16000
RANDOM_SEED = 42
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")   # librosa/soundfile can read all of these


def normalize_audio(audio):
    """Normalize audio to a peak amplitude of 0.95."""
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.95

    return audio


def load_audio(input_path):
    """Load as 16 kHz mono, trim silence and normalize (no file is written)."""
    audio, sr = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) == 0:
        raise ValueError(f"No usable audio found in {input_path}")

    return normalize_audio(audio)


def preprocess_audio(input_path, output_path):
    """Load, trim silence, normalize, and save an audio recording."""
    audio = load_audio(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE)

    return audio


def augment_audio(audio):
    """Create five augmented versions of an audio recording."""
    return {
        "pitch_up": librosa.effects.pitch_shift(
            y=audio, sr=SAMPLE_RATE, n_steps=1
        ),
        "pitch_down": librosa.effects.pitch_shift(
            y=audio, sr=SAMPLE_RATE, n_steps=-1
        ),
        "faster": librosa.effects.time_stretch(
            y=audio, rate=1.05
        ),
        "slower": librosa.effects.time_stretch(
            y=audio, rate=0.95
        ),
        "background_noise": audio + np.random.default_rng(
            RANDOM_SEED
        ).normal(0, 0.005, len(audio))
    }


def extract_features(audio):
    """Extract MFCC, spectral, energy, and zero-crossing features."""
    features = {}

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = float(
            np.mean(mfcc[i])
        )
        features[f"mfcc_{i + 1}_std"] = float(
            np.std(mfcc[i])
        )

    feature_groups = {
        "rolloff": librosa.feature.spectral_rolloff(
            y=audio, sr=SAMPLE_RATE
        ),
        "rms": librosa.feature.rms(y=audio),
        "centroid": librosa.feature.spectral_centroid(
            y=audio, sr=SAMPLE_RATE
        ),
        "zcr": librosa.feature.zero_crossing_rate(y=audio)
    }

    for name, values in feature_groups.items():
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    return features


def get_speaker_and_utterance(file_path):
    """Read speaker and utterance information from the filename."""
    speaker = file_path.stem.split("_")[0]

    utterance = file_path.stem

    prefix = speaker + "_"
    if utterance.startswith(prefix):
        utterance = utterance[len(prefix):]

    suffixes = [
        "_clean",
        "_pitch_up",
        "_pitch_down",
        "_faster",
        "_slower",
        "_background_noise"
    ]

    for suffix in suffixes:
        if utterance.endswith(suffix):
            utterance = utterance[:-len(suffix)]
            break

    return speaker, utterance


def run_pipeline():
    """Process all raw speaker recordings and extract features."""
    np.random.seed(RANDOM_SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []

    raw_files = sorted(
        p for p in RAW_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )

    if not raw_files:
        print(f"No audio recordings found in {RAW_DIR}")
        return

    for input_path in raw_files:
        speaker = input_path.parent.name.lower()
        original_name = input_path.stem

        clean_path = PROCESSED_DIR / f"{original_name}_clean.wav"

        print(f"Processing: {input_path.name}")

        clean_audio = preprocess_audio(input_path, clean_path)

        recordings = {
            "clean": clean_audio
        }

        augmented = augment_audio(clean_audio)

        for variant, audio in augmented.items():
            audio = normalize_audio(audio)

            output_path = (
                AUGMENTED_DIR /
                f"{original_name}_{variant}.wav"
            )

            sf.write(output_path, audio, SAMPLE_RATE)
            recordings[variant] = audio

        for variant, audio in recordings.items():
            features = extract_features(audio)

            feature_rows.append({
                "speaker": speaker,
                "utterance": original_name,
                "variant": variant,
                "filename": (
                    f"{original_name}_clean.wav"
                    if variant == "clean"
                    else f"{original_name}_{variant}.wav"
                ),
                **features
            })

    features_df = pd.DataFrame(feature_rows)

    csv_path = PROCESSED_DIR / "audio_features.csv"
    features_df.to_csv(csv_path, index=False)

    print("\nAudio pipeline completed!")
    print(f"Processed recordings: {len(raw_files)}")
    print(f"Feature dataset shape: {features_df.shape}")
    print(f"Saved features to: {csv_path}")


# ------------------------------------------------------------------
# Helpers used by the command-line app (app/cli.py).
# They reuse the functions above, so a login clip is processed exactly
# like the training clips.
# ------------------------------------------------------------------
def list_raw_audio():
    """One row per raw recording: member (folder name), phrase, path."""
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.suffix.lower() in AUDIO_EXTS:
            member = path.parent.name.lower()
            phrase = path.stem.lower().replace(" ", "")
            phrase = phrase[len(member) + 1:] if phrase.startswith(member + "_") else phrase
            rows.append({"member": member, "phrase": phrase, "path": path})
    return pd.DataFrame(rows, columns=["member", "phrase", "path"])


def audio_to_features(path):
    """One audio file -> one-row feature table (same columns as training)."""
    return pd.DataFrame([extract_features(load_audio(path))])


if __name__ == "__main__":
    run_pipeline()
