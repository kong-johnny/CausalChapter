# Resource-bounded prediction-pipeline release plan

Latest user decision, 2026-11-03: epoch3 was stopped after epoch2 dev regression.
Epoch1 full70 test is 101.83/21.20/16.00. The user accepted this usable version
and will manually upload a verified weight archive to Google Drive. Further
optimization and automatic publication are superseded by packaging/handoff;
see `DRIVE_RELEASE_HANDOFF.md`. The remainder records the historical plan.

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-03
- Verification Status: UNVERIFIED (future experiments and acceptance)
- Version Label: resource_plan_v1
- Authority: user requests autonomous legitimate optimization and release;
  time-limited AIS access overrides any assumption of unlimited compute.

## Objective and comparison gate

Reach all applicable metrics for the same model scale, dataset and evaluation
definition, using LCDS predictions followed by two-pass KL-CSSE generation.
Never supply GT segmentation or GT-filtered chapters to generation. Reference
titles may supervise training losses; references remain scoring-only during
dev/test inference. Keep metrics and failed candidates, not only a winning row.

The 3B target is CIDEr/METEOR/SODA_c=104.59/17.87/12.73 and the reported
segmentation row F1=56.72/tIoU=69.95. The 7B main-table target is
CIDEr/SODA_c=116.88/13.80 and reported F1@30=72.97/tIoU=69.95. Do not mix
the rows or infer comparability from a numerically larger result. Reconcile
paper/old-notes column discrepancies, time-vs-utterance tolerance, terminal
endpoint treatment and caption matching before a paper-win claim.

The old GT-prefiltered generation protocol is not a permitted fallback.
If a historical row cannot be validly compared, preserve that unresolved status
and evaluate a corrected baseline and candidate with one fixed protocol;
such a comparison alone does not establish exceeding the published row.

## Current evidence and immutable run

- Segmenter: ten-epoch best, frozen threshold .75/gap8/middle, test internal
  F1 59.1398 and time F1@30s 69.1756. Keep its original frozen record unchanged.
- Fresh generator PID 70027, original AIS 61820589 through prun port 55623.
  Root `/mnt/nas2/duanmo/data/CausalChapter_generator_clean_20261103`;
  latest log comes from `run.logpath`. Do not restart or change its loaded code.
- Fresh base + new LoRA, train245 only, dev35 selection. Three epochs,
  rank32/lr2e-5/accum4, two gradient passes per epoch. Train chapters are
  predicted, not OOF; this limitation stays disclosed.
- Epoch 1 dev: 99.6889/20.8240/18.0033, 169 nonempty predicted titles,
  170 references, 154 lexical caption pairs; real-time all-event SODA.
  `outputs/clean_generator_20261103/epoch1_case_review.md` records cases.
- New LCDS GitHub-only download/CPU smoke already passed; it does not verify
  the fresh generator or a complete fresh-clone chain.

## Ordered execution and compute budget

1. **Finish the existing three-epoch run.** No parallel GPU training or duplicate
   smoke. Retrieve every epoch's metrics and output cases. Preserve the selected
   best checkpoint and hashes on persistent storage; never rely only on `/tmp`.
   At the last measured rate a complete epoch costs roughly three hours, with
   about 74 minutes of KL-CSSE construction; estimates are not reservations.
2. **CPU-first diagnosis and protocol reconciliation while GPU is occupied.**
   Reuse saved probabilities for threshold/gap analysis and saved titles for
   scoring audits. Do not regenerate captions to answer a scoring-only question.
   Do not launch another final-test parameter sweep. Train/dev remain the
   selection path in this plan.
3. **One bounded decoding comparison after training.** Use the selected fresh
   checkpoint, fixed predicted dev inputs and fixed metric protocol. At most
   three new settings on the same declared eight-video screening subset.
   Baseline plus three settings is a screening result, not benchmark evidence.
   Beam/CSSE settings must follow inspected errors; no Cartesian grid. Record
   runtime. A two-hour GPU ceiling is a planning budget: check progress and
   checkpoint safety before starting another candidate, not an automatic kill
   of a healthy in-flight process. Only the top one or two candidates advance
   to all 35 dev videos.
4. **Boundary candidate only if cases support it.** Fine dev-only threshold
   scan at gap8/middle found .74 F1@30s=78.5185 versus .75's 78.0669. This is
   a candidate, not a new locked model. Reuse probabilities, export real times,
   and inspect merged/extra chapters. Generation evaluation is needed before
   replacing .75. Never silently alter the current trainer's inputs.
5. **At most one additional training experiment in this allocation before
   reassessment.** Only if full-dev evidence indicates a remaining mechanism
   failure that decoding cannot address. Choose one hypothesis (e.g. local
   visual fusion or title-coverage training), not simultaneous changes. First
   perform a short smoke and a measured throughput/memory check, then decide
   whether sufficient allocation time remains for training and verification.
   Preserve LCDS local intervention and KL-CSSE. No blind 7B training merely
   to chase the main-table score; 3B adapters cannot initialize a 7B model.
6. **Freeze, score and independently reproduce.** Freeze the chosen checkpoint,
   decoder and generation configuration before final benchmark evaluation.
   Benchmark has previously been exposed; disclose reuse. No retrospective
   threshold/checkpoint selection from final scores under this plan. If the
   comparison fails, record failure and budget remaining rather than cycling
   through full-test runs until a lucky result passes.

## Case-driven decisions and controls

- Merged chapters (mechanics lectures 13/19): inspect missed local boundary
  scores and omitted topics; lowering a threshold is not assumed beneficial.
- Extra split (ODE lecture): inspect repeated titles and genuine topic changes.
- Correctly segmented DAG lecture: use it as a title-specificity control.
- Long chapters: measure actual token truncation before changing context size.
- CSSE errors: support IDs/KL are logged; use fixed-input ablation before
  attributing an error to retrieval. Do not delete KL-CSSE to save runtime and
  present the resulting model as the same mechanism.
- Freeze preprocessing, seeds, split IDs, cache bytes and scoring rules across
  each comparison. Compare CIDEr, METEOR and SODA jointly, including regressions.

## Allocation deadline and durable handoff

The user confirmed 25 hours remain on 2026-11-03 around 16:04 Asia/Shanghai.
Use 2026-11-04 17:00 Asia/Shanghai as the conservative working deadline, with
the last six hours (from 11:00) reserved for evaluation, transfer and handoff.
Back up the current useful best before further experiments. Avoid open-ended
follow-on jobs. Before new expensive work,
require estimated runtime plus at least six hours of margin for evaluation,
transfer and clone acceptance. If less time remains, preserve artifacts and
handoff state instead of starting a run likely to be evicted.

Keep log/provenance and the best useful checkpoint; do not retain all epoch
checkpoints. Do not delete a sole unknown historical checkpoint to reclaim space.
Only remove exact verified redundant task artifacts, with a recovery statement.

## Release acceptance, not just an upload

- Publish exact best weights, tokenizer/configuration, frozen decoder, dependency
  versions, model/data licenses and provenance, all required features and hashes.
- New GitHub clone downloads artifacts without borrowing NAS weights/features;
  verify hashes, no-label segmentation, real continuous chapter times and the
  complete predicted-boundary generation chain. Record hardware and tolerances.
- Include one-command train/infer/evaluate instructions, complete metrics and
  representative successes and failures. Weights are AVL-domain, not universal.
- Merge reviewed candidate branch only when comparable metric and reproducibility
  gates both pass, preserving unrelated user changes. No guarantee of universal
  bitwise equality or predetermined metric success.

## Monitoring

Use the existing automation only: exact PID/run.logpath, alive/progress/error and
completed epoch metrics. No new scheduler or duplicate task. Stay quiet for
healthy progress; notify on meaningful metric/quality changes, failure,
completion or required user action, jointly as segmenter/generator/end-to-end.
