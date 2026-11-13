# User Identity and Product Recommendation System

Formative 2: Multimodal Data Preprocessing. A user is **recognised by face**, the request is **confirmed by voice**, and only then does the system show a **product recommendation**.

```
face image ─▶ Face model ──unknown──▶ ACCESS DENIED
                 │ member X
                 ▼
     Product model runs for X (hidden)
                 │
voice clip ─▶ Voice model ──unknown / not X──▶ ACCESS DENIED
                 │ X confirmed
                 ▼
     Recommended product category is displayed
```

## Repository layout
| Path | What | Owner |
|---|---|---|
| `data/raw_data/*.xlsx` | provided customer datasets | jonathan |
| `EDA/data_merging.ipynb` | Task 1: cleaning, merge, EDA, feature engineering | jonathan |
| `data/cleaned_data/merged_dataset*.csv` | merged dataset (readable + encoded) | jonathan |
| `training_models/product_model_training.ipynb` → `models/product_model.joblib` | Product Recommendation Model | jonathan |
| `data/images/raw/<member>/` | 3 photos per member | everyone |
| `src/image_pipeline.py`, `EDA/image_processing.ipynb` | Task 2: face crop, augmentation, features → `image_features.csv` | shema |
| `src/train_face_model.py`, `training_models/face_model_training.ipynb` → `models/face_model.joblib` | Facial Recognition Model | shema |
| `src/identity_model.py` | shared train/evaluate/reject logic for face + voice | shema |
| `data/audio/raw/<member>/` | 2 recordings per member | everyone |
| `src/audio_pipeline.py`, `EDA/audio_processing.ipynb` | Task 3: plots, augmentation, features → `data/audio/processed/audio_features.csv` | santhiana |
| `src/train_voice_model.py`, `training_models/voice_model_training.ipynb` → `models/voice_model.joblib` | Voiceprint Verification Model | santhiana |
| `app/cli.py`, `src/product_recommender.py`, `src/make_unauthorized_samples.py` | Task 6: command-line system simulation | peggy |
| `src/config.py` | paths, member list, member → customer link | peggy |
| `REPORT.md`, `docs/DECISIONS.md`, `WORK_SPLIT.md` | report and documentation | peggy (all contribute) |

## Setup
```bash
pip install -r requirements.txt
```

## How to add your data (every member)
**Photos**: put them in `data/images/raw/<your-name>/`, named in lower case:
```
peggy_neutral.jpg   peggy_smiling.jpg   peggy_surprised.jpg
```
- Use a front-facing camera at roughly arm's length, with your face clearly visible and good light.
- `.jpg` and `.png` are both fine. Extra photos such as `peggy_smiling_2.jpg` are welcome; more photos give a better model.

**Recordings**: put them in `data/audio/raw/<your-name>/`:
```
peggy_yes_approve.wav   peggy_confirm_transaction.wav
```
- Record in a quiet room, saying the phrase once, clearly, in a 1–3 second clip.
- Extra takes such as `peggy_yes_approve_2.wav` improve the voice model.
- `.wav` and `.mp3` both work. Phone voice memos (`.m4a`) and WhatsApp voice notes (`.opus`) must be converted to `.wav` first, for example with an online converter.

**Unauthorized samples**: put a photo and a recording of a **non-member** (someone who agreed) in `data/images/unauthorized/` and `data/audio/unauthorized/`. If you have none, `python src/make_unauthorized_samples.py` creates fallback samples.

## Run the pipeline
```bash
python src/image_pipeline.py          # -> data/cleaned_data/image_features.csv
python src/audio_pipeline.py          # -> data/audio/processed/audio_features.csv
python src/train_face_model.py        # -> models/face_model.joblib
python src/train_voice_model.py       # -> models/voice_model.joblib
```
You can also run the notebooks in `EDA/` and then `training_models/`, which do the same steps with plots and explanations.

## Run the system simulation
```bash
python app/cli.py --demo all                                   # every scenario, one after another
python app/cli.py --demo full --member peggy                   # valid transaction
python app/cli.py --demo unauthorized-face                     # stranger's face  -> denied
python app/cli.py --demo unauthorized-voice                    # stranger's voice -> denied
python app/cli.py --demo wrong-member-voice                    # face of A + voice of B -> denied
python app/cli.py --face path/to/face.jpg --voice path/to/voice.wav
python app/cli.py                                              # interactive (asks for paths)
```
