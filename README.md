# TNBC Classification from H&E Whole-Slide Images (UNI + CLAM)

A portfolio project that classifies triple-negative breast cancer (TNBC) vs. non-TNBC directly from H&E-stained whole-slide images (WSIs), using a pathology foundation model for patch-level feature extraction and weakly-supervised multiple instance learning (MIL) for slide-level classification.

**Test AUC: 0.9115** (CLAM-SB, Google Colab T4)

This repository is built on top of the official [CLAM](https://github.com/mahmoodlab/CLAM) implementation (Lu et al., *Nature Biomedical Engineering*, 2021) and is distributed under the same GPL-3.0 license (see `LICENSE.md`). The original CLAM codebase (`models/`, `wsi_core/`, `main.py`, `create_patches_fp.py`, etc.) is unmodified; this project's original contributions are the two notebooks under `1_feature_extraction/` and `2_classification/`.

---

## Motivation

During my Master's research at KAIST, I predicted TNBC patients' response to neoadjuvant chemotherapy (NAC) from transcriptomic data using transfer learning (TRANSACT + ssGSEA), improving specificity by 23 percentage points over baseline. That work relied on transcriptomic data alone and lacked any pathological/morphological signal.

An interview for a computational pathology internship at LG AI Research surfaced this gap directly — a lack of whole-slide image (WSI) and multimodal experience. This project was built to close that gap: a pathology foundation model (UNI) combined with weakly-supervised MIL (CLAM) to classify TNBC directly from H&E slides, with no transcriptomic input.

---

## Pipeline

```
[Data collection]        [Feature extraction]        [Classification]        [Evaluation]
─────────────────        ────────────────────        ─────────────────       ────────────
TCGA-BRCA WSI      →     UNI (ViT-L/16)         →     CLAM-SB           →    Test AUC
(GDC, open access)       patch embeddings              Gated Attention         0.9115
                         (n_patches × 1024)             MIL

cBioPortal         →     saved as .pt files
ER/PR/HER2 labels
```

---

## Data

| Item | Detail |
|---|---|
| WSI source | GDC Portal, TCGA-BRCA, open access (no controlled-access token required) |
| Label source | cBioPortal API — ER / PR / HER2 status |
| TNBC definition | ER-negative + PR-negative + HER2-negative |
| Final cohort | 116 TNBC + 214 non-TNBC = **330 patients** |
| Sampling | Stratified random sampling on ER/PR/HER2 combination (`random_state=42`) |

All data used is from TCGA-BRCA's open-access tier (no protected health information / controlled-access genomic data).

---

## 1. Feature extraction — `1_feature_extraction/TNBCslide_usingUNImodel.ipynb`

Run on Google Colab (T4 GPU).

| Item | Detail |
|---|---|
| Model | [UNI](https://huggingface.co/MahmoodLab/UNI) (MahmoodLab, ViT-L/16 + DINOv2, pretrained on Mass-100K: 100K+ WSIs across 20 organ types) |
| Patch size | 256×256 px |
| Tissue filter | mean pixel value < 230, std > 10 (discard background) |
| Patch budget | dynamically computed per slide from its pixel dimensions (hard cap: 8,000 patches) |
| Extraction | streaming — 32 patches embedded at a time to keep memory usage fixed (~50MB), avoiding OOM on large SVS files (1–2GB) |
| Output | one `.pt` file per slide: `features` (n_patches × 1024), `coords`, `label`, `slide_size` |
| Result | 330 `.pt` files, avg. 5,577 patches/slide, avg. slide size 95,044 × 45,112 px |

> **Note:** the Hugging Face token required to load UNI is read via Colab Secrets / an interactive prompt at runtime — it is intentionally **not** stored in this notebook. See "Running this project" below.

## 2. Classification — `2_classification/CLAMLearning_edit.ipynb`

Run on Google Colab (T4 GPU); a local Apple Silicon (MPS) run was also performed for comparison.

| Item | Detail |
|---|---|
| Model | CLAM-SB (single-branch, gated-attention MIL) |
| Data split | Train 231 / Val 49 / Test 50 (70:15:15, stratified, `random_state=42`) |
| Loss | Weighted cross-entropy (TNBC class weight ≈ 1.84) + instance-level auxiliary loss |
| Instance loss | Pseudo-labels auto-generated from attention-ranked top-k / bottom-k patches |
| Hyperparameters | LR = 2e-4, weight decay = 1e-5, bag loss weight = 0.7, 30 epochs |
| Scheduler | CosineAnnealingLR (Colab) / ReduceLROnPlateau (local) |

---

## Results

| Metric | Colab (best) | Local (Apple M-series, MPS) |
|---|---|---|
| Test AUC | **0.9115** | 0.8420 |
| Balanced Accuracy | 0.8108 | — |
| F1 Score | 0.7568 | — |
| Best Val AUC | — | 0.8511 |

The gap between the two runs is attributable to CUDA vs. MPS RNG differences, minor config differences between the two notebook versions (e.g., weighted loss, LR scheduler), and split variance on a relatively small cohort (n=330).

### Attention heatmaps

![Attention heatmaps](assets/attention_heatmaps.png)

- TNBC slides: attention concentrates on specific regions, consistent with dense tumor cell areas.
- non-TNBC slides: attention is more diffusely spread across the slide.
- Of 4 example TNBC slides, 3 were correctly classified (predicted TNBC probability 0.854–0.994); 1 was misclassified (0.007).

![Confusion matrix (Colab)](assets/confusion_matrix.png)
![Training curves (Colab)](assets/training_curves.png)

---

## Key technical decisions

| Decision | Rationale |
|---|---|
| GDC open-access tier | No access token needed; 3,112 slides available under this tier |
| Streaming feature extraction | Avoids OOM when processing large (1–2GB) SVS files |
| Dynamic per-slide patch budget | Keeps tissue coverage proportional across slides of very different sizes |
| Stratified sampling (`random_state=42`) | Mitigates class imbalance and guarantees a reproducible cohort across sessions |
| Weighted cross-entropy | Corrects for TNBC (116) vs. non-TNBC (214) class imbalance |
| Instance-level auxiliary loss | Core CLAM mechanism — generates patch-level supervision from slide-level labels only |
| `batch_size=1` | Required since patch count varies per slide |
| AUC + Balanced Accuracy | More reliable than raw accuracy under class imbalance |

---

## References

| Component | Paper |
|---|---|
| CLAM | Lu, M.Y. et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." *Nature Biomedical Engineering*, 2021. |
| UNI | Chen, R.J. et al. "Towards a general-purpose foundation model for computational pathology." *Nature Medicine*, 2024. |
| Attention-based MIL | Ilse, M. et al. "Attention-based Deep Multiple Instance Learning." *ICML*, 2018. |

---

## Environment

| Stage | Environment |
|---|---|
| Feature extraction | Google Colab, T4 GPU |
| Classification (best result) | Google Colab, T4 GPU |
| Classification (local) | Apple M-series MacBook, MPS backend, macOS |
| Local setup | Python 3.10, Miniforge, conda env `clam` |

---

## Connection to my Master's research

```
Master's research (transcriptomics)
  GDSC cell lines → patient-level transfer learning (TRANSACT)
  ssGSEA pathway vectors → multi-output SVR
  → predicts TNBC NAC response (pCR / RD)
  → +23pp specificity vs. baseline
          ↓ extending into pathology imaging
This project (WSI-based)
  TCGA-BRCA WSI → UNI feature extraction
  Patch embeddings → CLAM gated-attention MIL
  → TNBC vs. non-TNBC classification
  → Test AUC 0.9115
          ↓ planned next
Multimodal extension
  WSI slide embeddings + ssGSEA pathway vectors, combined
  → pCR vs. RD prediction (PROACTING dataset)
  → integrating morphological + molecular information
```

---

## Status

| Item | Status |
|---|---|
| UNI feature extraction (330 patients) | ✅ Done |
| CLAM training (Colab, AUC 0.9115) | ✅ Done |
| CLAM training (local, AUC 0.8420) | ✅ Done |
| Attention heatmap visualization | ✅ Done |
| 5-fold cross-validation | 🔄 Planned |
| Multimodal extension (WSI + ssGSEA) | 🔄 Planned |
| pCR vs. RD task (PROACTING dataset) | 🔄 Planned |

---

## Running this project

1. Clone this repo and install dependencies per the upstream CLAM instructions (`env.yml`, `docs/INSTALLATION.md`).
2. Open `1_feature_extraction/TNBCslide_usingUNImodel.ipynb` in Google Colab. When prompted for a Hugging Face token (required for `MahmoodLab/UNI`, request access [here](https://huggingface.co/MahmoodLab/UNI)), either:
   - add it to Colab Secrets (🔑 icon in the left sidebar) under the name `HF_TOKEN`, or
   - enter it interactively when prompted — it is never written to the notebook file.
3. Run `2_classification/CLAMLearning_edit.ipynb` on the `.pt` feature files produced in step 2.

**Do not commit a Hugging Face token, Google Drive path containing personal identifiers, or any other credential to this repository.**

---

## License

This repository is a derivative of [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) and is distributed under the GNU General Public License v3.0 — see `LICENSE.md`.
