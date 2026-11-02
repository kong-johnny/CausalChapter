# Development-only visual alignment ablation

## Material Passport

- Type: code-experiment plan and execution record.
- Authority: user permits architecture and hyperparameter adaptation while
  retaining the paper's LCDS and CSSE mechanisms.
- Status: prepared; no improvement claim before measured development results.
- Inputs: the same original train/dev split and frozen first-run initialization.
- Selection: train/dev only; the already exposed final test is not an input.

## Observations motivating this change

The frozen dev decoder has 40 missed and 39 extra internal boundaries. There
are no adjacent gold internal boundaries less than eight utterances apart,
so direct suppression of two adjacent gold chapters is not the sole cause.
High-confidence extras include a lecturer saying “move on to a more
interesting problem” within a longer annotated chapter, and a boundary one
utterance after the annotated transition. This motivates checking whether
visual changes remain temporally aligned rather than using speech cues alone.

The original fusion queries the entire video's visual vectors without visual
position embeddings or an alignment mask. A unit test confirms it is invariant
to permuting those visual vectors. This establishes an architectural property,
not evidence that it causes every observed error.

## Controlled comparison

Both variants start from the same frozen first-run weights, use seed 42,
three further epochs, learning rate 1e-5, gradient accumulation four, identical
train/dev inputs, and midpoint dev selection. `global` is the continuation
control; `local2` changes only the visual attention mask to ±2 utterances.
The original zero-mask LCDS predictor, cosine dependency shift, local drop,
and learnable boundary bias remain unchanged. CSSE generation is unchanged.

`scripts/run_lcds_alignment_ablation.sh` runs both sequentially and then
performs the same development decoding search for each. It never reads test.
These are new fine-tuning experiments with new optimizer/scheduler states,
not exact resumed optimization. Only one best checkpoint per variant is kept.
Do not promote the local variant merely because its rationale is attractive;
retain the original if it does not improve the recorded dev objectives.

The experiment follows the experiment skill's evidence/monitoring discipline;
the user's explicit autonomous implementation permission supplies authority
for authored code and bounded new runs. No external model services are used.
