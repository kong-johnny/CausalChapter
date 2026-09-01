# CausalChapter

> **Paper information**  
> **Title:** CausalChapter: Improving Long-Form Video Chaptering with Interventional Dependency Modeling  
> **Conference:** EMNLP 2026

## Environment setup

### Tested environment

The best run was produced with the following environment:

```text
Python       3.10.13
PyTorch      2.9.1+cu128
Transformers 4.57.1
PEFT         0.15.2
Accelerate   1.0.1
NumPy        1.26.4
Safetensors  0.7.0
GPU          NVIDIA A100-SXM4-80GB
Driver       470.82.01
```

### Install

```bash
cd /path/to/CausalChapter
conda create -n causalchapter python=3.10 -y
conda activate causalchapter
pip install -r requirements.txt
```

If you want the closest environment to the released run, pin the core packages manually after creating the environment:

```bash
pip install torch==2.9.1 transformers==4.57.1 peft==0.15.2 accelerate==1.0.1 numpy==1.26.4 safetensors==0.7.0
pip install -r requirements.txt
```

Depending on your CUDA runtime and cluster image, you may need to install the matching PyTorch wheel from the official PyTorch index instead of the generic command above.

## Prepare external model and data

### Released CausalChapter checkpoint

The best Qwen2.5-3B CausalChapter checkpoint is distributed separately because
the LoRA weight file exceeds GitHub's single-file limit. Place the downloaded
file at:

```text
checkpoints/qwen2p5_3b_crossattn_clip_best/shared_lora/adapter_model.safetensors
```

Download link: **to be added**.

The expected SHA-256 checksum is recorded in `MANIFEST.sha256`.

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

### 2. Dataset and visual cache

Prepare AVLecture-style files under:

```text
CausalChapter/data/avlecture/
```

Required for reproducing the released evaluation:

```text
data/avlecture/test.jsonl
data/avlecture/test_subheading_abstract.jsonl
data/avlecture/vis2d_cache/avlecture/
```

Optional training / validation files used by the training scripts include:

```text
data/avlecture/train_subheading_abstract_mixed_youhuahou.jsonl
data/avlecture/dev_subheading_abstract_mixed_youhuahou.jsonl
data/avlecture/train_mixed_youhuahou.jsonl
data/avlecture/dev_mixed_youhuahou.jsonl
```

If your dataset is stored elsewhere:

```bash
cd /path/to/CausalChapter
ln -s /absolute/path/to/avlecture data/avlecture
```

## Quick start

```bash
cd /path/to/CausalChapter
bash scripts/verify_artifacts.sh
bash scripts/reproduce_best.sh
```

By default, the reproduction output is written to:

```text
outputs/reproduce_best/
```

The script sets all key environment variables internally.  You can override paths when needed:

```bash
CC_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct \
CC_VIS_CACHE_DIR=/path/to/vis2d_cache/avlecture \
CC_EVAL_SAVE_DIR=outputs/my_reproduce_run \
bash scripts/reproduce_best.sh
```
