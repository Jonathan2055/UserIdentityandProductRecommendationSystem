"""Web interface for the User Identity and Product Recommendation System.

Same models and the same decision logic as app/cli.py, in the browser:

    streamlit run app/web_app.py

Flow: face image -> face model (unknown = ACCESS DENIED)
      -> product model runs for the recognised member (result hidden)
      -> voice sample -> voice model (must be the SAME member)
      -> recommendation is shown.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import config as C  # noqa: E402
from audio_pipeline import audio_to_features, list_raw_audio  # noqa: E402
from identity_model import identify  # noqa: E402
from image_pipeline import image_to_features, list_raw_images  # noqa: E402
from product_recommender import load_product_model, recommend  # noqa: E402

st.set_page_config(page_title="Identity Check", layout="centered")


# ------------------------------------------------------------------ models and samples
@st.cache_resource
def load_models():
    missing = [p.name for p in (C.FACE_MODEL_PATH, C.VOICE_MODEL_PATH, C.PRODUCT_MODEL_PATH) if not p.exists()]
    if missing:
        return None, missing
    return (joblib.load(C.FACE_MODEL_PATH), joblib.load(C.VOICE_MODEL_PATH), load_product_model()), []


def sample_files(kind: str) -> dict[str, Path]:
    """Label -> path for every member file plus the unauthorized samples."""
    if kind == "image":
        files = list(list_raw_images()["path"])
        files += [p for p in sorted(C.IMG_UNAUTH_DIR.glob("*")) if p.suffix.lower() in C.IMAGE_EXTS]
    else:
        files = list(list_raw_audio()["path"])
        files += [p for p in sorted(C.AUDIO_UNAUTH_DIR.glob("*")) if p.suffix.lower() in C.AUDIO_EXTS]
    return {f"{Path(p).parent.name}/{Path(p).name}": Path(p) for p in files}


def pick_file(kind: str, key: str) -> Path | None:
    """Let the user choose a dataset sample or upload their own file."""
    source = st.radio("Source", ["Choose a sample", "Upload a file"], horizontal=True,
                      key=f"{key}-src", label_visibility="collapsed")
    if source == "Choose a sample":
        options = sample_files(kind)
        label = st.selectbox("Sample", list(options), index=None, placeholder="Choose a file…", key=f"{key}-pick")
        return options.get(label) if label else None
    types = ["jpg", "jpeg", "png"] if kind == "image" else ["wav", "mp3", "flac", "ogg"]
    up = st.file_uploader("Upload", type=types, key=f"{key}-up", label_visibility="collapsed")
    if up is None:
        return None
    tmp = Path(tempfile.gettempdir()) / f"identity-check-{key}{Path(up.name).suffix.lower()}"
    tmp.write_bytes(up.getvalue())
    return tmp


def show_check(result: dict, threshold: float) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Best match", result["best_guess"].title())
    c2.metric("Confidence", f"{result['confidence']:.0%}")
    c3.metric("Needs at least", f"{threshold:.0%}")
    st.caption("Novelty check: " + ("looks like no enrolled team member" if result["novel"]
                                    else "looks like an enrolled team member"))


def reset():
    st.session_state.attempt = st.session_state.get("attempt", 0) + 1
    for k in ("face", "voice", "rec"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------ page
st.title("Identity Check")
st.write("Face check → hidden product prediction → voice check. "
         "A product recommendation is shown only to a verified team member.")

models, missing = load_models()
if models is None:
    st.error("Missing model file(s): " + ", ".join(missing) +
             ". Train them first: `python src/train_face_model.py --rebuild` and "
             "`python src/train_voice_model.py --rebuild`.")
    st.stop()
face_model, voice_model, product_model = models

if "attempt" not in st.session_state:
    st.session_state.attempt = 0
n = st.session_state.attempt
st.button("Start a new attempt", on_click=reset)

# ---- step 1: face
st.header("1. Face check")
face_path = pick_file("image", f"face{n}")
if face_path:
    st.image(str(face_path), width=260)
    if st.button("Verify face", type="primary", key=f"vf{n}"):
        X, found = image_to_features(face_path)
        st.session_state.face = {"result": identify(face_model, X), "found": found, "file": face_path.name}
        st.session_state.pop("voice", None)
        st.session_state.pop("rec", None)

face = st.session_state.get("face")
if not face:
    st.stop()
show_check(face["result"], face_model["threshold"])
if not face["found"]:
    st.caption("No face was detected in the picture, so the centre of the picture was used.")
if not face["result"]["accepted"]:
    st.error(f"**ACCESS DENIED**: face not recognised ({face['result']['reason']}).")
    st.stop()
member = face["result"]["member"]
st.success(f"Face recognised: **{member.title()}**. Product model unlocked.")

# ---- step 2: hidden product prediction
st.header("2. Product prediction")
if "rec" not in st.session_state:
    st.session_state.rec = recommend(member, product_model)
rec = st.session_state.rec
st.info(f"Prediction computed for customer #{rec['customer_id']}. It stays hidden until the voice check passes.")

# ---- step 3: voice
st.header("3. Voice check")
voice_path = pick_file("audio", f"voice{n}")
if voice_path:
    st.audio(str(voice_path))
    if st.button("Verify voice", type="primary", key=f"vv{n}"):
        st.session_state.voice = {"result": identify(voice_model, audio_to_features(voice_path)), "file": voice_path.name}

voice = st.session_state.get("voice")
if not voice:
    st.stop()
show_check(voice["result"], voice_model["threshold"])
if not voice["result"]["accepted"]:
    st.error(f"**ACCESS DENIED**: voice not verified ({voice['result']['reason']}).")
    st.stop()
if voice["result"]["member"] != member:
    st.error(f"**ACCESS DENIED**: the voice belongs to {voice['result']['member'].title()}, "
             f"but the face belongs to {member.title()}.")
    st.stop()

# ---- result
st.success(f"**ACCESS GRANTED**: voice verified as {member.title()}. Transaction approved.")
st.header(f"Recommended product: {rec['top']}")
p = rec["profile"]
c1, c2, c3 = st.columns(3)
c1.metric("Purchases", p["purchases"])
c2.metric("Average spend", f"{p['avg_spend']:.0f}")
c3.metric("Purchase interest", f"{p['avg_purchase_interest']:.1f}")
chart = pd.DataFrame(rec["ranking"], columns=["Category", "Probability"]).set_index("Category")
st.bar_chart(chart, horizontal=True)
