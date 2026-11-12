"""Task 4 - Facial Recognition Model.

Reads data/cleaned_data/image_features.csv (from src/image_pipeline.py), trains and
evaluates the face model, saves models/face_model.joblib.

    python src/train_face_model.py            # use the existing features file
    python src/train_face_model.py --rebuild  # re-run the image pipeline first
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from identity_model import identify, save_bundle, train_identity_model  # noqa: E402


def check_unauthorized(bundle) -> None:
    """Show that stranger images in data/images/unauthorized/ are rejected."""
    from image_pipeline import image_to_features
    files = [p for p in sorted(C.IMG_UNAUTH_DIR.glob("*")) if p.suffix.lower() in C.IMAGE_EXTS]
    if not files:
        print("\n(no images in data/images/unauthorized/ - add one to test rejection)")
        return
    print("\nUnauthorized images:")
    for p in files:
        X, _ = image_to_features(p)
        r = identify(bundle, X)
        print(f"  {p.name:<30} -> {'ACCEPTED as ' + r['member'] if r['accepted'] else 'REJECTED'} ({r['reason']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-run image_pipeline first")
    args = ap.parse_args()

    if args.rebuild or not C.IMAGE_FEATURES_CSV.exists():
        from image_pipeline import build_image_features
        build_image_features()

    df = pd.read_csv(C.IMAGE_FEATURES_CSV)
    print("=== Facial Recognition Model ===")
    bundle = train_identity_model(df, group_col="source_image")
    save_bundle(bundle, C.FACE_MODEL_PATH)
    print(f"\nSaved -> {C.FACE_MODEL_PATH}")
    check_unauthorized(bundle)


if __name__ == "__main__":
    main()
