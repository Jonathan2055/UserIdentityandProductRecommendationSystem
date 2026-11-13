
from pathlib import Path

import joblib
import pandas as pd

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
    print(
        "Next, implement speaker-independent training and evaluation "
        "before saving a model."
    )


if __name__ == "__main__":
    main()
