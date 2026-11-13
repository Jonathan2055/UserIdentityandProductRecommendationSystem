"""Task 6 - System demonstration (command-line app).

Flow (matches the assignment diagram):
    face image ──> Face model ──(unknown)──> ACCESS DENIED
                       │ recognised member X
                       ▼
            Product model runs for X's linked customer (result kept hidden)
                       │
    voice sample ─> Voice model ──(unknown, or not X)──> ACCESS DENIED
                       │ voice confirmed as X
                       ▼
            Show the product recommendation

Usage
    python app/cli.py                                  # interactive: asks for file paths
    python app/cli.py --face F.jpg --voice V.wav       # one transaction with your own files
    python app/cli.py --demo full                      # valid transaction (auto-picks files)
    python app/cli.py --demo unauthorized-face         # stranger's face
    python app/cli.py --demo unauthorized-voice        # right face, stranger's voice
    python app/cli.py --demo wrong-member-voice        # right face, ANOTHER member's voice
    python app/cli.py --demo all                       # all of the above, one after another
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import config as C  # noqa: E402
from audio_pipeline import audio_to_features, list_raw_audio  # noqa: E402
from identity_model import identify  # noqa: E402
from image_pipeline import image_to_features, list_raw_images  # noqa: E402
from product_recommender import load_product_model, recommend  # noqa: E402

LINE = "-" * 62
PAUSES = True   # short pauses between steps make the demo video easier to follow


def say(msg: str = "", pause: float = 0.0) -> None:
    print(msg, flush=True)
    if pause and PAUSES:
        time.sleep(pause)


def load_models():
    missing = [p for p in (C.FACE_MODEL_PATH, C.VOICE_MODEL_PATH, C.PRODUCT_MODEL_PATH) if not p.exists()]
    if missing:
        sys.exit("Missing model file(s): " + ", ".join(str(m) for m in missing)
                 + "\nTrain them first: python src/train_face_model.py --rebuild && python src/train_voice_model.py --rebuild")
    return joblib.load(C.FACE_MODEL_PATH), joblib.load(C.VOICE_MODEL_PATH), load_product_model()


def denied(why: str) -> bool:
    say(f"\n  ACCESS DENIED - {why}")
    say(LINE)
    return False


def run_transaction(face_path: Path, voice_path: Path, models, pause: float = 0.4) -> bool:
    face_model, voice_model, product_model = models
    say(LINE)

    # ---- step 1: face
    say(f"[1/3] Face check   : {face_path.name}", pause)
    X_face, found = image_to_features(face_path)
    if not found:
        say("      (no face detected in the image - using the centre of the picture)")
    face = identify(face_model, X_face)
    say(f"      best match {face['best_guess']} ({face['confidence']:.0%} confident, threshold {face_model['threshold']:.0%})")
    if not face["accepted"]:
        return denied(f"face not recognised: {face['reason']}.")
    member = face["member"]
    say(f"      Face recognised -> {member}. Product model unlocked.", pause)

    # ---- step 2: product prediction (hidden until voice confirms)
    rec = recommend(member, product_model)
    say(f"[2/3] Prediction   : computed for customer #{rec['customer_id']} (hidden until voice approval)", pause)

    # ---- step 3: voice
    say(f"[3/3] Voice check  : {voice_path.name}", pause)
    voice = identify(voice_model, audio_to_features(voice_path))
    say(f"      best match {voice['best_guess']} ({voice['confidence']:.0%} confident, threshold {voice_model['threshold']:.0%})")
    if not voice["accepted"]:
        return denied(f"voice not verified: {voice['reason']}.")
    if voice["member"] != member:
        return denied(f"voice belongs to {voice['member']}, but the face belongs to {member}.")
    say(f"      Voice verified -> {member}. Transaction approved.", pause)

    # ---- result
    p = rec["profile"]
    say(f"\n  ACCESS GRANTED - welcome {member.title()}")
    say(f"  Customer #{rec['customer_id']}: {p['purchases']} purchases, avg spend {p['avg_spend']:.0f}, "
        f"engagement {p['avg_engagement']:.0f}, purchase interest {p['avg_purchase_interest']:.1f}")
    say(f"  Recommended product category: >>> {rec['top'].upper()} <<<")
    for cat, prob in rec["ranking"]:
        say(f"      {cat:<12} {'#' * int(round(prob * 40)):<20} {prob:.0%}")
    say(LINE)
    return True


# ------------------------------------------------------------------ demo file picking
def _first(paths):
    paths = [p for p in paths if p]
    return paths[0] if paths else None


def pick_member_files(member: str | None):
    imgs, auds = list_raw_images(), list_raw_audio()
    members = [member] if member else [m for m in C.MEMBERS if m in set(imgs.member) & set(auds.member)]
    if not members:
        sys.exit("No member has both a photo and a recording in data/images/raw and data/audio/raw.")
    m = members[0]
    img = imgs[imgs.member == m].sort_values("expression")["path"]
    aud = auds[auds.member == m].sort_values("phrase", ascending=False)["path"]   # 'yes_approve' first
    return m, img.iloc[0], aud.iloc[0]


def stranger(folder: Path, exts):
    return _first(sorted(p for p in folder.glob("*") if p.suffix.lower() in exts))


def run_demo(kind: str, member: str | None, models) -> None:
    m, face, voice = pick_member_files(member)
    if kind in ("full", "all"):
        say(f"\n=== DEMO: valid transaction by {m} ===")
        run_transaction(face, voice, models)
    if kind in ("unauthorized-face", "all"):
        s = stranger(C.IMG_UNAUTH_DIR, C.IMAGE_EXTS)
        say("\n=== DEMO: unauthorized face ===")
        if s:
            run_transaction(s, voice, models)
        else:
            say("No image in data/images/unauthorized/ - add a non-member photo (or run src/make_unauthorized_samples.py).")
    if kind in ("unauthorized-voice", "all"):
        s = stranger(C.AUDIO_UNAUTH_DIR, C.AUDIO_EXTS)
        say(f"\n=== DEMO: {m}'s face + a stranger's voice ===")
        if s:
            run_transaction(face, s, models)
        else:
            say("No recording in data/audio/unauthorized/ - add a non-member recording (or run src/make_unauthorized_samples.py).")
    if kind in ("wrong-member-voice", "all"):
        auds = list_raw_audio()
        other = auds[auds.member != m]
        say(f"\n=== DEMO: {m}'s face + ANOTHER member's voice ===")
        if other.empty:
            say("Need recordings from at least two members for this demo.")
        else:
            run_transaction(face, other.iloc[0]["path"], models)


def interactive(models) -> None:
    say("User Identity & Product Recommendation System  (blank line to quit)")
    while True:
        face = input("\nPath to face image : ").strip().strip('"')
        if not face:
            break
        voice = input("Path to voice clip : ").strip().strip('"')
        if not voice:
            break
        if not Path(face).exists() or not Path(voice).exists():
            say("  file not found, try again")
            continue
        run_transaction(Path(face), Path(voice), models)


def main():
    ap = argparse.ArgumentParser(description="Face + voice authenticated product recommendation")
    ap.add_argument("--face", type=Path)
    ap.add_argument("--voice", type=Path)
    ap.add_argument("--demo", choices=["full", "unauthorized-face", "unauthorized-voice", "wrong-member-voice", "all"])
    ap.add_argument("--member", choices=C.MEMBERS, help="which member to use in --demo")
    ap.add_argument("--fast", action="store_true", help="no pauses between steps")
    args = ap.parse_args()

    global PAUSES
    PAUSES = not args.fast
    models = load_models()

    if args.demo:
        run_demo(args.demo, args.member, models)
    elif args.face and args.voice:
        sys.exit(0 if run_transaction(args.face, args.voice, models) else 1)
    else:
        interactive(models)


if __name__ == "__main__":
    main()
