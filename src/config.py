"""Shared settings for the whole pipeline (paths, team members, thresholds).

Every script and notebook imports from here, so a path or a name only has to be
changed in one place.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned_data"
MODELS_DIR = ROOT / "models"

# Optional: set UIPR_WORK_DIR to run the image/audio pipeline on a copy of the
# media somewhere else (used for testing with dummy data so nothing fake ends up
# in the repo). Normally it is unset and everything lives in data/ and models/.
_WORK = Path(os.environ["UIPR_WORK_DIR"]) if os.environ.get("UIPR_WORK_DIR") else None
MEDIA_DIR = _WORK if _WORK else DATA_DIR
FEATURES_DIR = _WORK if _WORK else CLEAN_DIR
TRAINED_DIR = _WORK if _WORK else MODELS_DIR

IMG_RAW_DIR = MEDIA_DIR / "images" / "raw"
IMG_AUG_DIR = MEDIA_DIR / "images" / "augmented"
IMG_UNAUTH_DIR = MEDIA_DIR / "images" / "unauthorized"

AUDIO_RAW_DIR = MEDIA_DIR / "audio" / "raw"
AUDIO_AUG_DIR = MEDIA_DIR / "audio" / "augmented"
AUDIO_UNAUTH_DIR = MEDIA_DIR / "audio" / "unauthorized"

IMAGE_FEATURES_CSV = FEATURES_DIR / "image_features.csv"
AUDIO_FEATURES_CSV = MEDIA_DIR / "audio" / "processed" / "audio_features.csv"   # written by audio_pipeline.py (Santhiana)
MERGED_ENCODED_CSV = CLEAN_DIR / "merged_dataset_encoded.csv"   # Task 1 output (Jonathan)

FACE_MODEL_PATH = TRAINED_DIR / "face_model.joblib"
VOICE_MODEL_PATH = TRAINED_DIR / "voice_model.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_model.joblib"         # Jonathan's model

# ---------------------------------------------------------------- team
MEMBERS = ["peggy", "jonathan", "santhiana", "shema"]

# Required file names per member (see README "How to add your data")
EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = ["yes_approve", "confirm_transaction"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

# Team members are not customers in the transactions file, so each member is
# linked to one real customer_id. After face + voice pass, the product model
# predicts for that customer. Chosen: customers WITH social data and the most
# purchases (4-5 each), each on a different main platform.
MEMBER_TO_CUSTOMER = {
    "peggy": 187,      # 5 purchases, LinkedIn
    "jonathan": 114,   # 4 purchases, Facebook
    "santhiana": 101,  # 4 purchases, Twitter
    "shema": 188,      # 4 purchases, TikTok
}

# ---------------------------------------------------------------- processing
IMG_SIZE = 128          # face crops are resized to IMG_SIZE x IMG_SIZE
SAMPLE_RATE = 16000     # all audio resampled to 16 kHz mono (enough for speech)
N_MFCC = 13
RANDOM_STATE = 42

# Fallback confidence thresholds if a model file does not store its own.
# The training scripts pick the real threshold from cross-validation.
DEFAULT_THRESHOLD = 0.60
