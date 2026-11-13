# Work split and how each member commits their part

Nothing in this repo has been committed for you. Each member commits **their own files from their own git account**, so the GitHub history shows everyone's contribution. Run the commands from the repo root after `git pull`.

| Member | Responsibility | Status |
|---|---|---|
| **jonathan** | Task 1 data merge, cleaning, EDA; Product Recommendation Model | done (commit `e723634`) |
| **shema** | Task 2 image pipeline and the Facial Recognition Model (+ shared identity-model logic) | code pushed; run the notebooks once all photos are in and commit `image_features.csv` + `face_model.joblib` |
| **santhiana** | Task 3 audio pipeline and the Voiceprint Verification Model | audio pipeline pushed; voice-model training finished by peggy (see below) |
| **peggy** | Task 6 CLI system simulation, unauthorized demo, report and docs (took over from shema) | done; record the video |
| **everyone** | 3 photos (neutral, smiling, surprised) and 2 recordings ("Yes, approve", "Confirm transaction") | ✅ peggy, santhiana, shema · ❌ **jonathan still missing** |

## Suggested order
1. **Everyone**: add your own photos and recordings (see README, "How to add your data") and commit them:
   ```bash
   git add data/images/raw/<you>/ data/audio/raw/<you>/
   git commit -m "Add <you>'s face images and voice samples"
   git push
   ```
2. **peggy** commits the shared setup first, because every other file imports `src/config.py`:
   ```bash
   git add src/config.py requirements.txt .gitignore README.md data/images/unauthorized/.gitkeep data/audio/unauthorized/.gitkeep
   git commit -m "Add project config, folder structure and dependencies"
   ```
3. **shema** (after everyone's photos are in):
   ```bash
   python src/image_pipeline.py && python src/train_face_model.py
   # or run EDA/image_processing.ipynb then training_models/face_model_training.ipynb (keep the outputs)
   git add src/image_pipeline.py src/identity_model.py src/train_face_model.py \
           EDA/image_processing.ipynb training_models/face_model_training.ipynb \
           data/images/augmented/ data/cleaned_data/image_features.csv models/face_model.joblib
   git commit -m "Image processing, image_features.csv and facial recognition model"
   ```
4. **santhiana** (after everyone's recordings are in, and after shema has pushed `identity_model.py`):
   ```bash
   python src/audio_pipeline.py && python src/train_voice_model.py
   git add src/audio_pipeline.py src/train_voice_model.py \
           EDA/audio_processing.ipynb training_models/voice_model_training.ipynb \
           data/audio/processed/audio_features.csv models/voice_model.joblib
   git commit -m "Audio processing, audio_features.csv and voiceprint verification model"
   ```
5. **peggy** (once both models exist):
   ```bash
   python src/make_unauthorized_samples.py      # only if no real stranger samples were added
   python app/cli.py --demo all                  # record this for the video
   git add app/cli.py src/product_recommender.py src/make_unauthorized_samples.py \
           data/images/unauthorized/ data/audio/unauthorized/ REPORT.md docs/DECISIONS.md WORK_SPLIT.md
   git commit -m "CLI system simulation, unauthorized demo, report and docs"
   ```

**If the files are on one laptop:** copy them to the owner (share the folder or zip it), or commit with the owner's identity, e.g. `git commit --author="Name <their-github-email>"`. The `--author` flag credits them on GitHub; agree on this as a team first.

## Contribution sheet (fill in)
| Member | What I did | Files | Hours |
|---|---|---|---|
| jonathan | | | |
| peggy | CLI app (`app/cli.py`), product link (`src/product_recommender.py`), config, unauthorized samples, voice-model training on top of santhiana's audio pipeline (mp3 support, CLI helpers), report and docs | | |
| santhiana | Audio preprocessing and EDA pipeline, own recordings and photos | | |
| shema | Face-model files, own photos and recordings | | |
