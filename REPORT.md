# Formative 2 – User Identity and Product Recommendation System

**Team:** Jonathan, Peggy, Santhiana, Shema  
**GitHub:** _<repo link>_  
**Demo video:** _<video link>_

> Numbers marked **[fill in]** come from running the notebooks on the team's real photos and recordings. Copy them from the printed tables.

## 1. Overview
The system protects a product-recommendation model behind two biometric checks:
1. **Face recognition** decides *who* is trying to use the system. An unknown face is denied immediately.
2. For a recognised member, the **product model** predicts the product category for their linked customer profile. The result stays hidden.
3. **Voiceprint verification** must confirm the **same** member, who says "Yes, approve" or "Confirm transaction". Only then is the recommendation shown.

Any failed step prints `ACCESS DENIED` with the reason. Three models are used: face (images), voice (audio) and product (tabular data).

## 2. Data merge and product model (Task 1)
**Cleaning:**
- `customer_id_new` (`A178`) was converted to the numeric format of `customer_id_legacy` (`178`).
- 5 exact duplicate social rows were removed.
- Customers with several social rows (41 of 84) were collapsed into one row each: mean engagement, interest and sentiment, the majority platform, and a profile count. Without this, the join would duplicate purchases.
- 10 missing ratings were filled with the overall median (3.0), and a `rating_was_missing` flag was added. A per-product median would leak the target.
- Data types were checked, and no outliers were found under the IQR rule.

**Merge:**
- A **left join** of transactions → social on `customer_id`, because every purchase (the target) must be kept.
- Validation: 150 rows in and 150 out, with no duplicated `transaction_id`. 117 purchases (78%) matched a social profile, and the other 33 were flagged with `has_social_data = 0` and given neutral values.

**Feature engineering:**
- Customer spend, frequency and average rating.
- Purchase size relative to that customer's usual amount.
- Month, weekday and weekend.
- Engagement × interest.
- Platform one-hot encoding.

**EDA (5 labelled plots):**
- The target is balanced (25–35 purchases per category).
- There are no outliers.
- Purchase amounts overlap heavily between categories.
- Social signals barely differ by product.
- The only strong correlations are between engineered features and their sources.

**Model:** a Random Forest, with the split and cross-validation done **by customer**.
- It was grown from a 1-tree model (underfit) to an unlimited one (overfit, with train accuracy near 1.00 and validation accuracy around 0.2), then constrained to `max_depth=3, min_samples_leaf=20`.
- Locked test set: **accuracy 0.20, macro-F1 0.11, log loss 1.69**. The baseline loss is 1.66, and random guessing gives an accuracy of 0.20 and a loss of 1.61.
- A shuffled-label check scored almost the same as the real labels (0.13 vs 0.17 accuracy). **The dataset carries very little signal about which product is bought.** We report this honestly: the product model works mechanically in the pipeline, but its recommendations are close to chance.

## 3. Images (Task 2)
- **Collection:** each member provided neutral, smiling and surprised photos, named `<member>_<expression>.jpg` in `data/images/raw/<member>/`. So far: Peggy, Santhiana and Shema (9 photos, `.jpg`/`.jpeg`). **[fill in: Jonathan's photos]**
- **Preprocessing:** EXIF rotation is applied, then the Haar-cascade face detector crops the face with a 20% margin, and the crop is resized to 128×128. Faces were detected in 8 of 9 photos; `santhiana_surprised.jpeg` fell back to the centre crop.
- **Augmentation (4 per image):**
  - 15° rotation
  - horizontal flip
  - grayscale
  - brightness/contrast increase
  This gives 5 rows per photo.
- **Features (358):**
  - HOG with 9 orientations, 32-pixel cells and 2×2 blocks (324 values, for shape)
  - a uniform LBP histogram (10 values, for texture)
  - an 8-bin RGB histogram (24 values, for colour)
- **Output:** saved in `data/cleaned_data/image_features.csv`, with the columns `member, expression, augmentation, source_image`.
- **Why not deep embeddings:** deep frameworks are heavy and unreliable on our Python version. Histograms and embeddings are both allowed, and these features are explainable.

## 4. Audio (Task 3)
- **Collection:** 2 phrases per member, as `.wav` or `.mp3` files in `data/audio/raw/<member>/`. So far there are 6 clips (Peggy, Santhiana, Shema; 2.0–3.9 s each). **[fill in: Jonathan's clips]**
- **Preprocessing:** 16 kHz mono, trimming silence quieter than 25 dB below the peak, then peak normalisation to 0.95 so that microphone distance is removed as a cue. The cleaned clips are saved in `data/audio/processed/`.
- **Visualisation:** a waveform and a mel-spectrogram for every member and phrase. Syllables show up as bursts in the waveform. In the spectrogram, harmonic stripes reflect the speaker's pitch, formant bands reflect the vocal tract (which is what identifies a voice), and high-frequency smears are fricatives ("s", "f"). **[fill in: one observation comparing two members]**
- **Augmentation (5 per clip):**
  - pitch ±1 semitone (kept small so the speaker's identity is preserved)
  - time stretch ×1.05 (faster) and ×0.95 (slower)
  - Gaussian background noise (σ = 0.005)
  This gives 6 rows per clip.
- **Features (34):**
  - 13 MFCCs (mean and standard deviation)
  - spectral roll-off, RMS energy, spectral centroid and zero-crossing rate (mean and standard deviation of each)
- **Output:** saved in `data/audio/processed/audio_features.csv`, with the columns `speaker, utterance, variant, filename`.

## 5. Face and voice models (Task 4)
**Shared recipe** (`src/identity_model.py`):
- Logistic Regression (scaled) and Random Forest are compared, and the best by cross-validated macro-F1 is chosen.
- `StratifiedGroupKFold` groups rows by **original file**, so augmented copies of one photo or clip never appear in both train and test.

**Open-set rejection.** A classifier always names *some* member, so two checks let the system say "unknown":
1. **Confidence threshold:** the 10th percentile of confidence on correct CV predictions, clipped to [0.40, 0.85].
2. **Local Outlier Factor novelty detector:** this flags inputs unlike any enrolled member.

| Model | Chosen algorithm | CV accuracy | CV F1 | CV loss | Train loss | Threshold | Genuine accepted |
|---|---|---|---|---|---|---|---|
| Face* | Logistic Regression | 0.93 | 0.93 | 0.15 | 0.00 | 0.66 | 76% |
| Voice* | Random Forest | 0.56 | 0.56 | 0.84 | 0.12 | 0.41 | 22% |
| Product | Random Forest | 0.20 (test) | 0.11 (test) | 1.69 (test) | | – | – |

\* *Interim results* with 3 of 4 members (Jonathan's photos and recordings are still missing). These will be re-run once all data is in.

- **The face model** generalises well to unseen photos (0.93 CV accuracy, with no photo accepted as the wrong member).
- **The novelty check matters.** In the demo, the stranger's photo was classified as "peggy" with 98% confidence, and only the Local Outlier Factor check rejected it.
- **The voice model is weak.** With only 2 clips per person, the 2-fold CV trains on one phrase and tests on the *other* phrase, which is a hard test. CV accuracy is 0.56 across 3 speakers, against 0.33 for chance. The large train-vs-CV gap (loss 0.12 vs 0.84) shows the model memorises.
- **Recording more takes per person** is the most effective improvement for the voice model. **[fill in: confusion-matrix observations]**

## 6. Multimodal decision logic and system demo (Task 6)
`app/cli.py` implements this flow:
1. **Face:** the face model runs. If it returns unknown (low confidence or a novelty), the attempt is **denied**.
2. **Product:** the product model runs for the member's linked customer (`MEMBER_TO_CUSTOMER` in `src/config.py`). Team members aren't in the transactions data, so each is linked to a real customer with social data and 4–5 purchases. The result is kept hidden.
3. **Voice:** the voice model runs. It must be accepted **and** predict the same member as the face. Otherwise the attempt is **denied**.
4. **Result:** the recommendation is shown, with the probability for each category.

**Demonstrated scenarios** (`python app/cli.py --demo all`):

| Scenario | Expected | Result |
|---|---|---|
| Member's face + member's voice | granted, recommendation shown | **[fill in]** |
| Stranger's face | denied at the face step | **[fill in]** |
| Member's face + stranger's voice | denied at the voice step | **[fill in]** |
| Member A's face + member B's voice | denied (voice/face mismatch) | **[fill in]** |

## 7. Limitations and possible improvements
- **Small biometric data:** with only 3 photos and 2 clips per person, more takes would make both models more reliable. A pretrained face embedding such as FaceNet and a speaker embedding such as ECAPA would generalise better to new photos.
- **Weak product signal:** the product data barely predicts the category (see Section 2). More behavioural features, such as browsing history or past categories (with care to avoid leakage), would be needed.
- **No liveness detection:** the face step accepts a printed photo. A real system would add liveness detection and a spoken-phrase check (speech recognition), not only speaker identity.

## 8. Contributions
| Member | Contribution |
|---|---|
| Jonathan | Data cleaning, merge and validation, EDA, feature engineering, Product Recommendation Model |
| Shema | Image pipeline (face crop, augmentation, features), `image_features.csv`, Facial Recognition Model, shared identity-model logic |
| Santhiana | Audio pipeline (preprocessing, visualisation, augmentation, features), `audio_features.csv` |
| Peggy | CLI system simulation, unauthorized-attempt demo, voice-model training on top of the audio pipeline, report and documentation |
| All | Own face images and voice recordings, demo video |
