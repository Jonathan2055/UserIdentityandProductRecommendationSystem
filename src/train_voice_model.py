"""Task 4 - Voiceprint Verification Model.

Reads data/audio/processed/audio_features.csv (made by src/audio_pipeline.py),
trains and evaluates the voice model, saves models/voice_model.joblib.

    python src/train_voice_model.py            # use the existing features file
    python src/train_voice_model.py --rebuild  # re-run the audio pipeline first
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_pipeline import AUDIO_EXTS, audio_to_features, run_pipeline  # noqa: E402
from identity_model import identify, save_bundle, train_identity_model  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = (
    PROJECT_DIR / "data" / "audio" / "processed" / "audio_features.csv"
)
MODEL_DIR = PROJECT_DIR / "models"


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    data = pd.read_csv(FEATURES_PATH)

    print("VOICE MODEL TRAINING")
    print("--------------------")
    print(f"Dataset shape: {data.shape}")

    if "speaker" not in data.columns:
        raise ValueError("The dataset must contain a speaker column.")

    print("\nRecordings per speaker:")
    print(data.groupby("speaker")["filename"].nunique())

    speakers = data["speaker"].nunique()

    if speakers < 2:
        print("\nTraining paused.")
        print("Only one speaker is currently available.")
        print(
            "Please collect recordings from other consenting "
            "speakers before training a verification model."
        )
        print(
            "Augmented recordings from one person cannot replace "
            "independent speakers."
        )
        return

    print("\nMultiple speakers found.")
    print("Training with grouped cross-validation (by original recording).\n")

    # Same training/evaluation recipe as the face model (src/identity_model.py).
    # Rename this file's columns to the shared names:
    #   speaker -> member, utterance (original recording) -> source_file,
    #   variant -> augmentation. 'filename' is only bookkeeping, so it is dropped.
    shared = data.rename(columns={
        "speaker": "member",
        "utterance": "source_file",
        "variant": "augmentation",
    }).drop(columns=["filename"])

    bundle = train_identity_model(shared, group_col="source_file")
    save_bundle(bundle, MODEL_DIR / "voice_model.joblib")
    print(f"\nSaved -> {MODEL_DIR / 'voice_model.joblib'}")
    check_unauthorized(bundle)


def check_unauthorized(bundle):
    """Show that non-member recordings in data/audio/unauthorized/ are rejected."""
    folder = PROJECT_DIR / "data" / "audio" / "unauthorized"
    files = [p for p in sorted(folder.glob("*")) if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        print("\n(no recordings in data/audio/unauthorized/ - add one to test rejection)")
        return
    print("\nUnauthorized recordings:")
    for p in files:
        r = identify(bundle, audio_to_features(p))
        verdict = "ACCEPTED as " + r["member"] if r["accepted"] else "REJECTED"
        print(f"  {p.name:<30} -> {verdict} ({r['reason']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the voiceprint verification model")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run the audio pipeline (audio_pipeline.py) first")
    if ap.parse_args().rebuild:
        run_pipeline()
    main()
