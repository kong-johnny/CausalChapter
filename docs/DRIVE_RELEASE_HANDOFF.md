# First-epoch AVL weights: upload handoff

The user accepted this usable version on 2026-11-03 and will upload the weights
to BNU Cloud. This supersedes the earlier requirement to exceed every paper
metric before publishing. No further training or automatic main merge is queued.

## Upload

Shared archive: [BNU Cloud download](https://pan.bnu.edu.cn/l/N1dKPT).
The anonymous share page was accessible on 2026-11-06 and showed the expected
archive filename. `WEIGHTS_ARCHIVE.sha256` in the repository records its expected
SHA-256. Full remote archive download verification remains pending.
Do not replace the file with another checkpoint under the same version name.

Extract into a temporary directory and copy only `checkpoints/` into the repository
(as shown in README), keeping the repository scripts. The archive contains both checkpoint directories,
their tokenizers/configs, component licenses, checksums, and measured test results.
The Qwen base model and visual feature archive are separate dependencies.

## Exact scope and results

Built with Qwen. Generator: fresh Qwen2.5-3B-Instruct base plus rank32 LoRA,
text-only generation, two-pass KL-CSSE, selected by dev after epoch1.
Train245 only; dev35 selection. Train predicted segmentation is not OOF.
Epoch2 dev regressed; epoch3 was stopped by user request. The released adapter
is exactly the preserved epoch1 bytes, not the partially trained epoch3 state.

Frozen LCDS: new Longformer reconstruction, epoch10, threshold .75/gap8/middle.
Full70-video test, all329 predicted chapters, strict real timelines and lexical
caption matching: CIDEr **101.83**, METEOR **21.20**, SODA_c **16.00**.
Paper3B: 104.59/17.87/12.73, numerical differences -2.76/+3.33/+3.27.
Historical paper-protocol comparability is not fully resolved; do not claim
all paper metrics were exceeded. Weights are AVL-domain; retrain both modules
for a new domain. Caption scores use matched pairs; SODA uses full events.

## User-operated reproduction

1. Clone `https://github.com/kong-johnny/CausalChapter`, branch
   `codex/reproducible-release`; apply the supplied code changes if not pushed.
2. Verify the archive with the repository
   `WEIGHTS_ARCHIVE.sha256` file, extract to a temporary
   directory, and copy only its checkpoint directories as shown in README.
   Run `python scripts/verify_artifacts.py --profile clean` after installing features.
3. Obtain official `Qwen/Qwen2.5-3B-Instruct` separately. Its model license is
   **Qwen Research**, not the source repository's MIT license: non-commercial
   research/evaluation use; commercial use requires the upstream permission.
   See the enclosed `LICENSE-Qwen` and `NOTICE`.
4. `git lfs pull --include="data/avlecture/lcds_visual_features.tar.gz"` fetches
   the complete released feature archive; retain its `.files.json` manifest.
5. Install the repository environment and Java for METEOR. Tested runtime:
   PPU Torch2.9.0, Transformers4.57.1, PEFT0.15.2, NumPy1.26.4. PPU users
   source their SDK first. Generic CUDA clean-install validation is pending.
6. Set `CC_BASE_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct` and run
   `bash scripts/reproduce_clean_epoch1.sh`. This recomputes predicted LCDS
   boundaries and runs generation without GT segmentation or GT prefiltering.
7. Check `outputs/clean_epoch1_reproduced/eval_summary.json`, retain full logs,
   and compare all three metrics. The new combined fresh-download chain has
   not yet been accepted; the earlier GitHub download acceptance covers LCDS.

Publish code, results, licenses and download instructions together. Source code
is MIT; model/data components retain upstream terms. Existing historical
checkpoints remain diagnostic alternatives, not the default for this version.
