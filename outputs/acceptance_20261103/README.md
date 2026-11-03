# Partial acceptance evidence — not a final benchmark

`acceptance.json` and `fresh_clone_acceptance.log` verify GitHub-only download,
artifact/feature hashes, and annotation-free CPU inference for the new LCDS
model at source commit f31587f. They do not certify a complete generator run.

`results.json` and `search_plan.json` preserve the completed eight-video dev
sweep. **These are quarantined diagnostics, not independent holdout scores or
evidence of outperforming the paper.** Historical generator training can mix
dev examples into gradients, and exact inherited-weight provenance is not
certified. See `docs/REPRODUCIBILITY_AUDIT.md` before interpreting these files.

`clean_generator/` records the fresh-base experiment's initial launch,
train/dev input and base hashes, and successful one-video two-pass GPU smoke.
The initial launcher stopped at a blank predicted-chapter timeline mismatch;
that pre-model error was fixed without rerunning the prepared segmenter.
Formal generator training then started from fresh base weights, not smoke
weights or an inherited adapter. This directory currently contains no final
generator metrics and does not establish release acceptance.
