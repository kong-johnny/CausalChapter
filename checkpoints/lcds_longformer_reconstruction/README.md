# New LCDS reconstruction

Domain: **AVL / AVLecture**. These are AVL fine-tuned weights, not a
cross-domain performance guarantee. When moving to a different dataset or
domain, retrain/fine-tune both segmentation and generation components and
select domain-specific settings on that domain's dev split.

This is a newly trained Longformer + CLIP-feature boundary model, not the
unrecovered historical causal checkpoint. The full encoder/head state is
included in `best.pt`; no external base weights are required for inference.

Initialization: [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096),
revision `301e6a42cb0d9976a6d6a26a079fef81c18aa895`.
The upstream encoder/tokenizer is Apache-2.0 licensed; see
`LICENSE-Longformer`. CausalChapter's added implementation is MIT licensed.
The model was fine-tuned on the released AVLecture train split for ten epochs;
epoch and decoding selection used dev only. Dataset/media rights are not
granted by the code license. See `data/PROVENANCE.md`.

Use `python -m segment.infer_lcds --help`. Inference needs utterance text,
timestamps, and matching visual features, but no chapter annotation file.
The exact frozen decoder is `dev_decoding_tuning.json`; `best_config.json`
retains the original run configuration for provenance. Its server-local base
initialization path is not accessed by the packaged inference command.

Frozen test: internal P/R/F1 63.71/55.18/59.14%; ±5/10/30-second F1
47.31/63.80/69.18%; matched MAE@30s 4.60 seconds. These are not the
historical manuscript metric row. Do not select new parameters using this
final test. Cross-hardware bitwise equality has not been established.
