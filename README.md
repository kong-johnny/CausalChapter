# CausalChapter

> **Paper information**  
> **Title:** CausalChapter: Improving Long-Form Video Chaptering with Interventional Dependency Modeling  
> **Conference:** EMNLP 2026

The source code is released under the MIT License. Dataset files and the
Qwen2.5 base model remain subject to their respective upstream licenses.

This audited release is currently on **`codex/reproducible-release`**, not
`main`. It contains recovered Qwen checkpoints and a newly trained LCDS
reconstruction. The exact historical causal checkpoint was not recovered;
the new weights are not a claim of identical historical paper reproduction.

### Checkpoint scope: AVL / AVLecture

The released segmentation and generation weights are trained/fine-tuned for
**AVL (AVLecture)**. They are not advertised as domain-general or as
VidChapters-trained checkpoints. For a different dataset or domain, retrain
or fine-tune **both the segmenter and generator** using that domain's training
data, and select its own configuration on a held-out development split.
Downloading another dataset does not adapt these AVL weights automatically.

## Environment setup

### Tested environment

The 2026-11-02 evaluation was run in an isolated environment on the author's
PPU server:

```text
Python       3.12.3
PyTorch      2.9.0 (PPU runtime)
Transformers 4.57.1
PEFT         0.15.2
Accelerate   1.0.1
NumPy        1.26.4
Safetensors  0.7.0
pycocoevalcap 1.2
Java         OpenJDK 8
```

### Install

```bash
git lfs install
git clone --branch codex/reproducible-release https://github.com/kong-johnny/CausalChapter.git
cd /path/to/CausalChapter
git lfs pull
conda env create -f environment.yml
conda activate causalchapter
bash scripts/verify_artifacts.sh
```

The public environment file uses upstream PyTorch; PPU users must retain the
framework supplied by their PPU SDK. CUDA and CPU clean-install acceptance is
pending. Cross-device bitwise equality after retraining has not been
established. See [the audit](docs/REPRODUCIBILITY_AUDIT.md) for measured results
and unresolved paper-table discrepancies.

## Prepare external model and data

### Released checkpoints

The release packaging uses Git LFS for these checkpoint paths:

```text
checkpoints/segment_qwen2p5_3b_endpos_z035/
checkpoints/qwen2p5_3b_crossattn_clip_best/shared_lora/adapter_model.safetensors
checkpoints/lcds_longformer_reconstruction/
```

Their expected SHA-256 checksums and all released evaluation artifacts are in
`MANIFEST.sha256`. `scripts/verify_artifacts.sh` checks every byte before an
evaluation starts.

### 1. Base model

Download **Qwen2.5-3B-Instruct** from its official source and expose it as:

```text
CausalChapter/models/Qwen2.5-3B-Instruct/
```

For example, if it already exists elsewhere:

```bash
cd /path/to/CausalChapter
ln -s /absolute/path/to/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
```

The fixed release uses the 3B model.  It is **not** a 7B checkpoint.

### 2. Datasets

Download the source data from the official repositories:

- [AVLectures](https://github.com/Darshansingh11/AVLectures)
- [VidChapters-7M](https://github.com/antoyang/VidChapters)

Raw videos are not redistributed in this repository. Follow the download
instructions and terms of use provided by each dataset.

#### AVLecture

The exact processed train/dev/test JSONL files are included under
`data/avlecture/`, with provenance in `data/PROVENANCE.md`. Prepare the released
evaluation cache with:

```bash
python scripts/prepare_eval_features.py
```

The evaluation inputs are:

```text
data/avlecture/test.jsonl
data/avlecture/test_subheading_abstract.jsonl
data/avlecture/vis2d_cache/avlecture/
```

`test.jsonl` stores the utterance-level transcript and timestamps;
`test_subheading_abstract.jsonl` stores the corresponding chapter segments and
descriptions. The historical release cache contains visual features for only
2 of 70 test videos; the historical loader supplies zeros for the rest. This
records the behavior of the existing checkpoint and does not establish full
multimodal coverage. A separate complete cache recovered from backup is used
for the new LCDS reconstruction.

Use a fresh clone for the historical release profile. Mixing extra visual
features into that cache changes the inputs and is rejected by the preparation
script.

#### VidChapters-7M

Follow the official VidChapters-7M repository to download its annotations and
ASR transcripts and run the supplied preprocessing scripts. Retain, for each
video, the ordered transcript, chapter timestamps, and chapter titles, then
extract the visual features and map the processed records to the same fields
used above. Raw YouTube videos should remain outside this repository.

## Quick start

### New LCDS reconstruction

The checkpoint includes its Longformer encoder, boundary/dependency heads,
tokenizer, and frozen dev-only decoder. An additional base Longformer download
is not required for inference. Extract its complete feature cache separately
from the historical two-video cache:

```bash
python scripts/prepare_eval_features.py \
  --archive data/avlecture/lcds_visual_features.tar.gz --destination data/lcds
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 python -m segment.infer_lcds \
  --checkpoint checkpoints/lcds_longformer_reconstruction \
  --text data/avlecture/test.jsonl \
  --visual-cache data/lcds/vis2d_cache/avlecture \
  --output outputs/lcds_inference
```

This inference entry point does not load chapter annotations. Custom inputs
need ordered `example_id`, `text`, and `stet` JSONL fields plus matching 512D
per-utterance visual features; raw video/ASR/feature extraction is not performed
by this command. It exports boundary labels, actual contiguous chapter times,
and probabilities. Fresh-clone acceptance of this new entry point is pending.

The frozen 70-video test has internal boundary P/R/F1 of
63.71/55.18/59.14%, and time-domain F1 at ±5/10/30 seconds of
47.31/63.80/69.18%. Matched error at 30 seconds is 4.60s. These are
new reconstruction results, not the historical model's metric row.

The actual new-LCDS predicted-boundary → generation evaluation completed
all 70 AVL videos with 329 nonempty titles. Under the separately documented
strict protocol: CIDEr **89.25**, METEOR **19.29**, SODA_c **14.99**.
CIDEr/METEOR are temporal-IoU-matched caption scores; SODA evaluates all
events against real reference times. See `outputs/lcds_generate_strict/`.
One-video annotation-free CPU inference also passed with exactly the same
decoded boundaries as the frozen dev reference; full fresh-clone download
acceptance is tracked in the audit.

### Historical Qwen profile (legacy evaluation)

The commands below preserve the historical evaluation protocol. It filters
predicted segments using annotations before generation and uses predicted
times for reference events; its SODA score must not be presented as strict
end-to-end timing quality. The separate strict generation path retains all
chapters and true reference times; see the audit for details.

```bash
cd /path/to/CausalChapter
bash scripts/verify_artifacts.sh
bash scripts/reproduce_all.sh
```

By default, the reproduction output is written to:

```text
outputs/reproduce_best/
```

This runs the released segmenter first and feeds its predicted boundaries to
the released title/summary generator. Segmentation-only and generation-only
runs are also available:

```bash
bash scripts/reproduce_segmentation.sh
bash scripts/reproduce_best.sh
```

The scripts set all released hyperparameters internally. You can override
paths when needed:

```bash
CC_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct \
CC_VIS_CACHE_DIR=/path/to/vis2d_cache/avlecture \
CC_EVAL_SAVE_DIR=outputs/my_reproduce_run \
bash scripts/reproduce_best.sh
```

### Released segmentation configuration

The released segmenter classifies the final token of each utterance, so the
causal Qwen backbone has observed the full utterance before making a boundary
decision. It uses LoRA rank 64, class weight 0.35, probability threshold 0.35,
and a minimum gap of 6 utterances. The complete machine-readable configuration
is `checkpoints/segment_qwen2p5_3b_endpos_z035/release_config.json`.

To evaluate the committed boundary artifact without a GPU:

```bash
python -m segment.evaluate_boundaries \
  --boundaries boundaries/boundaries_12_thr0.35_gap6.jsonl \
  --text data/avlecture/test.jsonl \
  --ground-truth data/avlecture/test_subheading_abstract.jsonl
```

### Reproducibility contract

- The historical checkpoint selection protocol is being audited. The legacy
  training scripts evaluate the test split during training; they must not be
  described as development-only model selection.
- All release weights and reference outputs are checksum-verified.
- `environment.yml` is a candidate pinned environment pending clean-install
  validation. Every runnable
  script accepts path overrides instead of depending on server paths.
- The historical generator reference is CIDEr 111.19, METEOR 20.18, SODA-c
  21.66. Re-evaluation on the current PPU produced 108.89, 19.89, 21.82 using
  the same fixed historical predicted boundaries. These are distinct measured
  runs; the latter does not establish exact reproduction of the former.

### Validate decoding without a GPU

```bash
python -m segment.replay_scores \
  --scores outputs/segment_reference/scores.jsonl \
  --output outputs/replayed_boundaries.jsonl \
  --expected boundaries/boundaries_12_thr0.35_gap6.jsonl
```

This reproduces all 70 historical boundary records byte for byte. It validates
decoding, not neural inference or retraining.

### New manuscript LCDS reconstruction

`segment/lcds.py` implements zero-mask interventions, cosine changes between
predicted following windows, local dependency drops, and a learnable
boundary-class logit bias. The Longformer + CLIP implementation and training
entry point are a new reconstruction, not the missing historical F1=56.72
checkpoint. Core tests cover masking, alignment, gradients, and independence
from ground-truth labels during prediction.

The new run uses original train/dev splits, pins the initial Longformer
revision, selects decoding parameters on dev only, and saves only the best
checkpoint. Full visual coverage is required; the historical 2-video cache is
insufficient for training this model.

```bash
CC_TRAIN_VIS_CACHE=/path/to/complete/visual/cache \
  bash scripts/train_lcds_reconstruction.sh

# A new run: freeze its own dev configuration before a final test.
# The released frozen configuration is in outputs/lcds_reconstruction/.
python scripts/evaluate_lcds_release.py \
  --output outputs/lcds_reconstruction \
  --data-dir data/avlecture \
  --visual-cache /path/to/complete/visual/cache \
  --decoding-config outputs/lcds_reconstruction/dev_decoding_tuning.json
```

The gate defaults to dev-only verification. `--final-test` explicitly enables
one test evaluation after it passes and creates an audit marker that prevents
silent repeated test runs. For an independently retrained checkpoint, first
collect its dev scores and run `scripts/tune_lcds_dev.py`; do not assume a
decoder selected for different weights is optimal.
