# AVL generation checkpoint

This recovered Qwen2.5-3B LoRA and visual Q-Former checkpoint is an
**AVL / AVLecture-domain** generation model. Its domain attribution follows
the author's supplied experiment provenance. It is not a VidChapters-trained
or domain-general checkpoint. Original base-model pretraining is separate
from this domain fine-tuning.

For a different dataset or domain, retrain/fine-tune both the segmenter and
generator and select new settings on the target domain's held-out dev data.
Do not promise equivalent accuracy by merely replacing the input video.

Use the pinned dependencies, tokenizer, shared LoRA, Q-Former, base model, and
feature profile together. Legacy generation evaluation has known temporal
protocol limitations; see `docs/REPRODUCIBILITY_AUDIT.md`. The strict
predicted-boundary pipeline is evaluated separately.
