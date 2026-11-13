# Design decisions (and why)

Every non-obvious choice in the project, in one place. The notebooks explain the same choices next to the code.

## Task 1: Data merge and product model (jonathan)
| Decision | Why |
|---|---|
| Strip the `A` from `customer_id_new` (`A178` → `178`) | The two files use different ID formats for the same customer. |
| Drop 5 exact duplicate social rows | They are repeated entries, not new information. |
| Collapse social profiles to **one row per customer** (mean scores, majority platform, profile count) | 41 customers have several social rows. Joining without this would copy a purchase several times and inflate the data. |
| Fill 10 missing ratings with the **overall** median and add a `rating_was_missing` flag | A per-product median would leak the target. The flag lets the model know which values were imputed. |
| **LEFT join**, with transactions on the left | The target (`product_category`) lives in transactions, so all 150 purchases must stay. An inner join would drop 33. |
| Post-merge checks: row count unchanged (150), no duplicate `transaction_id` | Proves the join didn't create or lose rows. |
| No features built from past product categories | Avoids leaking the answer. |
| Train/test split **by customer** (`GroupShuffleSplit`, `GroupKFold`) | Stops the model from scoring well just by remembering people. |
| Grow from a tiny model to a big one, then constrain (`max_depth=3`, `min_samples_leaf=20`) | Shows underfitting, then overfitting, then the balance point. |
| Shuffled-label check | Showed the features carry very little signal about the product. This is reported honestly rather than tuned away. |

## Shared settings (`src/config.py`)
| Decision | Why |
|---|---|
| One config file for paths, members, thresholds and links | Everything is changed in one place, and the notebooks, scripts and app stay in sync. |
| `MEMBER_TO_CUSTOMER`: peggy→187, jonathan→114, santhiana→101, shema→188 | Team members are not customers in the data, so each one is linked to a real customer. The chosen customers have social data, the most purchases (4–5) and different platforms. |
| Processing code in `src/`, with notebooks importing it | The rubric wants both scripts and a notebook. One shared implementation means the app processes a login photo **exactly** like the training photos. |

## Task 2: Images (shema)
| Decision | Why |
|---|---|
| Naming: `raw/<member>/<member>_<expression>.jpg` | The label comes from the folder, so there's no manual labelling, and a missing photo is spotted automatically. |
| Apply EXIF rotation when loading | Phone photos are often stored sideways. |
| Haar-cascade face crop with a 20% margin, resized to 128×128, falling back to a centre crop | Removes the background and keeps every vector the same size. The pipeline never crashes on a hard photo. |
| `opencv-python<5` | OpenCV 5 no longer ships the Haar cascade files. 4.x still does and installs on Python 3.14. |
| 4 augmentations: rotate 15°, horizontal flip, grayscale, brightness/contrast | Realistic variations of a login photo (tilt, mirrored selfie, colour or lighting). This multiplies 3 photos into 15 rows per member. |
| Features: HOG (324) + LBP histogram (10) + colour histogram (24) = 358 | Shape, texture and colour complement each other. They're explainable and need no deep-learning install. HOG uses large cells to keep the vector small for a small dataset. |

## Task 3: Audio (santhiana; voice-model training finished by peggy)
| Decision | Why |
|---|---|
| `.wav` or `.mp3`, loaded as 16 kHz mono | Speech sits below 8 kHz and 16 kHz is the speech standard. Every clip gets the same format. `.mp3` support was added because Shema's clips are MP3. |
| Trim silence (25 dB) and peak-normalise to 0.95 | Pauses and phone distance say nothing about who is speaking. |
| 5 augmentations: pitch ±1 semitone, speed ×1.05 and ×0.95, Gaussian noise (σ 0.005) | Natural variation: mood, speaking rate, a noisy room. Pitch is kept to ±1 so a voice doesn't turn into another member's. |
| Features: 13 MFCCs (mean/std), plus roll-off, RMS energy, centroid and ZCR (mean/std) = 34 | MFCCs are the classic voiceprint feature. Roll-off and energy were requested. Averaging over time gives clips of any length the same vector size. |
| Folder name gives the speaker, lower-cased | `Shema/` and `shema/` are the same person, and lower-casing prevents a duplicate class on case-sensitive systems (Mac/Linux). |
| Voice training reuses the face model's recipe (`identity_model.py`) by renaming `speaker→member`, `utterance→source_file`, `variant→augmentation` | Santhiana's CSV format stays unchanged, and both identity models are trained and evaluated the same way. |
| Raw recordings and `models/*.joblib` are committed (removed from `.gitignore`) | The assignment asks for the datasets, and the CLI needs the trained models. Generated audio (processed and augmented) stays ignored, except `audio_features.csv`. |

## Task 4: Face and voice models (shared `src/identity_model.py`)
| Decision | Why |
|---|---|
| Same recipe for face and voice | Both answer "which member is this, and are we sure?", and one implementation is easier to explain and test. |
| `StratifiedGroupKFold` grouped by **original file** | Augmented copies of one photo/recording never sit in both train and test, which would inflate the scores. This is the same idea as splitting by customer. |
| Number of folds = the fewest original files any member has (2–5) | Each fold tests one unseen file per member. |
| Candidates: Logistic Regression (scaled) and Random Forest; pick the best CV macro-F1, then the lowest loss | One linear and one non-linear model. The assignment lists both. |
| Metrics: accuracy, macro-F1, log loss and a confusion matrix, for train vs CV | The metrics the rubric asks for. The train-vs-CV gap shows any overfitting. |
| **Confidence threshold** = the 10th percentile of confidence on correct CV guesses, clipped to [0.40, 0.85] | Accepts about 90% of genuine attempts and rejects anything less sure. 0.25 is a pure guess with 4 members. |
| **Novelty check** (Local Outlier Factor) | A classifier always names *some* member. LOF recognises inputs unlike anyone in training (strangers), which makes "unauthorized" detectable. |

## Task 6: CLI (peggy)
| Decision | Why |
|---|---|
| Order: face → product prediction (hidden) → voice → display | Follows the assignment diagram. The prediction can only be *executed* (shown) after voice approval. |
| Voice must match the **same** member as the face | Stops person A's face combined with person B's voice. |
| Demo modes: `full`, `unauthorized-face`, `unauthorized-voice`, `wrong-member-voice`, `all` | Covers every accept and deny path for the video, with no hard-coded outcomes because the real models decide each time. |
| Fallback stranger samples (scikit-image's public-domain astronaut photo, a synthetic voice) | The demo still works if nobody outside the team records. A real non-member sample is preferred. |
