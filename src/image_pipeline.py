"""Task 2 - Image data collection and processing.

Steps for every member photo in data/images/raw/<member>/<member>_<expression>.jpg:
    1. load (respecting the phone's EXIF rotation)
    2. find the face and crop it (OpenCV Haar cascade), resize to 128x128
    3. make 4 augmented copies: rotation, horizontal flip, grayscale, brightness
    4. extract features from the original + every copy
       - HOG       : face shape / edges         (324 values)
       - LBP hist  : skin & texture pattern      (10 values)
       - colour    : 8-bin histogram per channel (24 values)
    5. save all rows to data/cleaned_data/image_features.csv

Run directly:  python src/image_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

AUGMENTATIONS = ["rotate_15", "flip", "grayscale", "brightness"]


# ------------------------------------------------------------------ discovery
def parse_image_name(path: Path, member: str) -> str:
    """'peggy_smiling.jpg' -> 'smiling'  ('peggy_smiling_2.jpg' -> 'smiling')."""
    stem = path.stem.lower()
    for expr in C.EXPRESSIONS:
        if expr in stem:
            return expr
    return stem.replace(f"{member}_", "")  # unknown expression name: keep it as is


def list_raw_images() -> pd.DataFrame:
    """One row per raw photo: member, expression, path."""
    rows = []
    for member in C.MEMBERS:
        folder = C.IMG_RAW_DIR / member
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in C.IMAGE_EXTS:
                rows.append({"member": member, "expression": parse_image_name(p, member), "path": p})
    return pd.DataFrame(rows, columns=["member", "expression", "path"])


def missing_images_report() -> list[str]:
    """Which required files (member x expression) are still missing."""
    have = list_raw_images()
    missing = []
    for m in C.MEMBERS:
        for e in C.EXPRESSIONS:
            if have[(have.member == m) & (have.expression == e)].empty:
                missing.append(f"{m}/{m}_{e}.jpg")
    return missing


# ------------------------------------------------------------------ loading / face crop
def load_image(path: str | Path) -> np.ndarray:
    """Load as RGB uint8. exif_transpose fixes phone photos that are stored sideways."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img)


def detect_and_crop_face(rgb: np.ndarray, size: int = C.IMG_SIZE) -> tuple[np.ndarray, bool]:
    """Return (face crop resized to size x size, face_found).

    The biggest detected face is used, with a 20% margin so the chin/forehead are kept.
    If no face is found we fall back to a centre square crop, so the pipeline never breaks.
    """
    h, w = rgb.shape[:2]
    scale = 800 / max(h, w) if max(h, w) > 800 else 1.0          # faster detection on big photos
    small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = [int(v / scale) for v in (x, y, fw, fh)]
        m = int(0.2 * max(fw, fh))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(w, x + fw + m), min(h, y + fh + m)
        crop, found = rgb[y0:y1, x0:x1], True
    else:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop, found = rgb[y0:y0 + s, x0:x0 + s], False

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), found


# ------------------------------------------------------------------ augmentation
def augment(face: np.ndarray) -> dict[str, np.ndarray]:
    """4 augmented copies of one face crop (all stay RGB 128x128)."""
    h, w = face.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return {
        # head slightly tilted
        "rotate_15": cv2.warpAffine(face, rot, (w, h), borderMode=cv2.BORDER_REFLECT),
        # mirror image (faces are roughly symmetric, so this is still the same person)
        "flip": cv2.flip(face, 1),
        # colour removed: forces the model not to rely only on skin/background colour
        "grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        # brighter + more contrast: simulates different lighting
        "brightness": cv2.convertScaleAbs(face, alpha=1.2, beta=25),
    }


# ------------------------------------------------------------------ features
def extract_features(face: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

    # HOG: 32x32-pixel cells -> 4x4 grid, 2x2-cell blocks, 9 orientations = 3*3*4*9 = 324 values.
    # Large cells keep the vector small (we only have tens of images).
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)

    # LBP (uniform, 8 neighbours, radius 1) -> 10-bin normalised histogram of texture codes
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(11), density=True)

    # Colour: 8-bin normalised histogram per RGB channel
    color = []
    for ch in range(3):
        hist, _ = np.histogram(face[..., ch], bins=8, range=(0, 256))
        color.extend(hist / hist.sum())

    feats = {f"hog_{i}": float(v) for i, v in enumerate(hog_vec)}
    feats.update({f"lbp_{i}": float(v) for i, v in enumerate(lbp_hist)})
    feats.update({f"color_{c}_{i}": float(color[k * 8 + i]) for k, c in enumerate("rgb") for i in range(8)})
    return feats


def image_to_features(path: str | Path) -> tuple[pd.DataFrame, bool]:
    """Used by the CLI: one image file -> one-row feature table (+ was a face found)."""
    face, found = detect_and_crop_face(load_image(path))
    return pd.DataFrame([extract_features(face)]), found


# ------------------------------------------------------------------ full run
def build_image_features(save_augmented: bool = True, verbose: bool = True) -> pd.DataFrame:
    raw = list_raw_images()
    if raw.empty:
        raise FileNotFoundError(f"No images found under {C.IMG_RAW_DIR}. See README 'How to add your data'.")

    rows = []
    for r in raw.itertuples():
        face, found = detect_and_crop_face(load_image(r.path))
        if verbose and not found:
            print(f"  ! no face detected in {r.path.name} -> used centre crop")
        variants = {"original": face, **augment(face)}
        source = f"{r.member}/{r.path.name}"
        for aug_name, img in variants.items():
            if save_augmented and aug_name != "original":
                out = C.IMG_AUG_DIR / r.member / f"{r.path.stem}_{aug_name}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(out, quality=95)
            rows.append({"member": r.member, "expression": r.expression, "augmentation": aug_name,
                         "source_image": source, "face_detected": int(found), **extract_features(img)})

    df = pd.DataFrame(rows)
    C.IMAGE_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.IMAGE_FEATURES_CSV, index=False)
    if verbose:
        print(f"{len(raw)} photos x {1 + len(AUGMENTATIONS)} versions = {len(df)} rows, "
              f"{df.shape[1] - 5} features -> {C.IMAGE_FEATURES_CSV}")
    return df


if __name__ == "__main__":
    miss = missing_images_report()
    if miss:
        print("Still missing (pipeline will run on what exists):\n  " + "\n  ".join(miss))
    build_image_features()
