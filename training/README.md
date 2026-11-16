# SPPG — Human-reviewed MAX30102 fine-tune experiment v1

A reproducible **research-only** comparison of the working v15-derived **lambda=0.5** PeakNet and an experimental real-data-adapted PeakNet. The working **v17 80 ms temporal decoder is unchanged**. This repository is **not** a validated clinical landmark detector.

## Input provenance

- `data/PPG00_original.zip`: all **23 original raw-RED/IR MAX30102 recordings**, 100 Hz, 1001 samples each, selected when signal appeared visually good. The ZIP SHA-256 matches `source_input_sha256` in the user's review JSON.
- `data/manual_review.json`: the uploaded human review, **preserved byte-for-byte**, with all 23 files and all annotation decisions.
- `data/auto_session.json`: original non-PeakNet automatic S landmark proposal positions (not confirmed by manual editing). Mirrors `tables/auto_session.json` solely to support the earlier exporter.
- `data/landmarks_reviewed.csv`: a convenient table of the **262 beats**, including per-beat status, S/D/DN positions and validation flags.
- `data/real_256_windows_reviewed.csv`: the **161** window export (256 samples, starts 0,128,256,384,512,640,745). Input is inverted **raw IR, per-window z-score**, not causal LCD-filtered IR.

### Human-annotation audit

- **72** distinct second peaks (D), **184** shoulders (explicitly no distinct D), **3** absent D, **3** unspecified D status.
- **257** recorded DN positions, but these do **not** contribute to this two-output (S,D) model's training objective.
- **2** recording-level quality fields are `good`; **21** remain `not_reviewed` (this is an unset field, *not* evidence that signal quality is bad).
- The hand/finger metadata fields are blank across all files; do not infer which recording came from which hand or finger.
- **One DN location appears assigned to the wrong beat:** `PPG0021_b05` has S=363, next S=447, but DN=461. The DN is flagged as invalid in the beat table and excluded from training anyway. The source JSON is not modified.
- The original reviewer's S peak positions are mostly **algorithmic provisional labels**, not manually adjudicated S. See `s_source` and `s_auto_ok` in the beat table.
- Three beats have no D status: `PPG0002_b10`, `PPG0009_b10`, `PPG0013_b12`; their D labels are masked where needed.

### Masked supervised windows

- `s_supervised=1`: **151/161** windows. S locations are provisional except where explicitly reviewed.
- `d_supervised=1`: **135/161** windows; **61** of those include at least one reviewed distinct D. All 26 other windows have unknown-D losses **masked**, not converted into negatives.
- `dn_supervised=0`: all windows. The only output classes remain **S and D**.

## What was trained

- Baseline model from `models/baseline_lambda050/` (the firmware lambda=0.5 model before v17 temporal postprocessing).
- Synthetic: original v15 four-case negative-aware generator (dual, weak, systolic-only, noise-only), 1800 training windows, fixed seed 1007, 600 synthetic validation windows, fixed seed 92926.
- Real: 18 full recordings train, **5 full recordings held out** (no overlapping windows split across train/validation), 126 real train windows and 35 held-out real windows. Training repeats the real windows four times per epoch to give them about 22% representation; all per-channel losses obey the supervision masks.
- QAT fine-tuning for three epochs, learning rate 1e-5, lambda=0.5 exclusion penalty, guard ±8 samples. Original activation quantization scales are **frozen** to make the weight-change comparison interpretable.
- Both INT8 models have **10,304 parameter bytes** and **10,496 activation arena bytes** in the C++ host Noodle2 graph.

## Reproduce

```bash
python -m pip install -r requirements.txt
python tools/export_reviewed_training.py --package . --review data/manual_review.json --out data/real_256_windows_reviewed.csv
python tools/finetune_real.py --epochs 3
cd cpp && make test all
cd ..
python tools/evaluate_real_int8.py
```

Original headers and checkpoints are in `models/baseline_lambda050/`; the experimental candidate is in `models/real_finetuned/`. The C++ Noodle2 direct-vs-tiled exactness and exported golden tests pass. The `results/` directory contains logs, held-out metrics and event-level predictions. The provided C++ host code is for Linux: physical ESP32 deployment remains **untested** for the new candidate.

## Interpretation

**Do not deploy the fine-tuned candidate as an upgrade** on the basis of this result: real held-out BCE improved but D event F1 with the v17 temporal decoder **decreased** on the same held-out recordings. Preserve lambda=0.5 + v17 as the reference firmware. Only seven manually marked D peaks fell within the held-out scoring area, so the evaluation has very wide uncertainty. For a publishable result, recruit another annotator and additional independent subjects/recording conditions.
