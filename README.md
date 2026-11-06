# CausalChapter

**CausalChapter: Improving Long-Form Video Chaptering with Interventional Dependency Modeling** — EMNLP 2026

LCDS predicts chapter boundaries; two-pass KL-CSSE generates chapter titles.
These weights target **AVL / AVLecture**. Fine-tune both modules for other domains.

## Installation

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/reproducible-release https://github.com/kong-johnny/CausalChapter.git
cd CausalChapter
git lfs pull --include="data/avlecture/lcds_visual_features.tar.gz"
conda env create -f environment.yml
conda activate causalchapter
```

The environment uses upstream PyTorch. PPU users should retain their SDK's
PyTorch runtime. Java is required for METEOR. Development uses Python3.12,
PPU PyTorch2.9.0, Transformers4.57.1, PEFT0.15.2 and NumPy1.26.4.

## Download weights

Download [CausalChapter-AVL-epoch1-20261103.tar.gz](https://pan.bnu.edu.cn/l/N1dKPT)
from BNU Cloud (1.76 GB / 1.64 GiB), and place it at the repository root.
The archive checksum is included in `WEIGHTS_ARCHIVE.sha256`.

Verify the archive and install its checkpoint directories at the repository root:

```bash
shasum -a 256 -c WEIGHTS_ARCHIVE.sha256
CC_WEIGHT_TEMP="$(mktemp -d)"
tar -xzf CausalChapter-AVL-epoch1-20261103.tar.gz -C "$CC_WEIGHT_TEMP"
mkdir -p checkpoints
cp -R "$CC_WEIGHT_TEMP/checkpoints/." checkpoints/
python scripts/verify_artifacts.py --profile clean
```

The archive includes:
```text
checkpoints/lcds_longformer_reconstruction/   # full segmenter and frozen decoder
checkpoints/qwen2p5_3b_clean_epoch1/          # generator LoRA, tokenizer, configuration
```

Keep the repository scripts when installing weights; the archive also contains
handoff documentation. Download the official base model separately:
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

## Run the complete pipeline

```bash
export CC_BASE_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct
bash scripts/reproduce_clean_epoch1.sh
```

This verifies artifacts, prepares the complete visual cache, predicts LCDS
boundaries and real times, and runs generation and evaluation. All predicted
chapters are retained. Reference annotations are scoring-only.
Outputs default to `outputs/clean_epoch1_reproduced/`; override
`CC_EVAL_SAVE_DIR` and `CC_SEG_OUTPUT_DIR` to choose different output directories.

Settings: text-only Qwen generation, rank32 LoRA, beam3, CSSE top3, seed42,
strict real timelines and lexical caption matching. LCDS uses the frozen
threshold0.75/gap8/middle decoder. `reproduce_best.sh`, `reproduce_all.sh` and
`reproduce_lcds.sh` are aliases for this pipeline.

### Segmentation only

```bash
python scripts/prepare_eval_features.py \
  --archive data/avlecture/lcds_visual_features.tar.gz --destination data/lcds
python -m segment.infer_lcds \
  --checkpoint checkpoints/lcds_longformer_reconstruction \
  --text data/avlecture/test.jsonl \
  --visual-cache data/lcds/vis2d_cache/avlecture \
  --output outputs/lcds_inference
```

Custom inputs require ordered `example_id`, `text`, `stet` fields and matching
512D per-utterance visual features. Raw video, ASR and feature extraction are
upstream steps. Chapter annotations are not required for segmentation inference.

## Training

Use `generate/train_clean.py` via `scripts/run_clean_generator.sh`:
fresh official base plus new LoRA, train-only gradients, predicted-dev selection.
See [training instructions](docs/CLEAN_GENERATOR_TRAINING.md) for required inputs.
LCDS training uses `scripts/train_lcds_reconstruction.sh`.

Original `generate/train.py` and `scripts/reproduce_legacy_*.sh` retain the
historical workflow; they are not the entry points for these weights.

## Data and licenses

Processed AVL splits: [data/PROVENANCE.md](data/PROVENANCE.md).
Source datasets: [AVLectures](https://github.com/Darshansingh11/AVLectures),
[VidChapters-7M](https://github.com/antoyang/VidChapters).

Source code is MIT. **Built with Qwen:** model weights/tokenizers retain
**Qwen Research License** terms for non-commercial research/evaluation;
commercial use requires an upstream license. See
`checkpoints/qwen2p5_3b_clean_epoch1/LICENSE-Qwen` and `NOTICE`.
Longformer retains Apache-2.0 terms. Dataset/derived feature rights remain
subject to upstream terms. Raw video/audio is not included.
