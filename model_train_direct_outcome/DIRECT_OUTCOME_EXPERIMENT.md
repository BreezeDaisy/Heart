# Direct Outcome Multitask Experiment

This copy keeps the original two-stage HMS/outcome code and adds a new
patient-level single-stage experiment.

## Goal

Predict `Outcome` directly from four-position heart sounds and clinical
features. Murmur is an auxiliary training target and is not required as an
inference input.

## New Files

- `src/dataset_direct_outcome.py`
  - One item is one patient.
  - Loads available `AV/MV/PV/TV` wavs.
  - Cuts recordings into 3-second segments.
  - Builds three log-mel scales per segment.
  - Adds clinical features:
    - Age
    - Sex
    - Height
    - Weight
    - Pregnancy status
    - Missing-value indicators

- `src/model_direct_outcome.py`
  - Reuses the original `ScaleEncoder`.
  - Encodes segment-level audio.
  - Uses attention and max pooling to build patient-level audio features.
  - Fuses audio features with clinical features.
  - Outputs:
    - `Outcome`: Normal / Abnormal
    - `Murmur`: Absent / Present / Unknown
    - `timing / grade / shape` auxiliary heads

- `src/train_direct_outcome.py`
  - Main direct-outcome training script.
  - Uses outcome as the primary loss.
  - Adds higher Abnormal class weight.
  - Adds a soft false-negative penalty for true Abnormal patients.
  - Searches validation threshold with recall priority.

- `train_direct_outcome_final.py`
  - One-command launcher for the first experiment preset.

## Backup

The copied original model file is backed up at:

`src/model_hms.py.bak`

## Run Smoke Test

```powershell
python .\src\train_direct_outcome.py --device cpu --smoke-test
```

## Run Training

Copy wav files into:

`data/circor/training_data/`

Then run:

```powershell
python train_direct_outcome_final.py
```

If no GPU is available:

```powershell
python train_direct_outcome_final.py --device cpu
```

## Metric Priority

This experiment should not be judged by plain accuracy first. Use:

1. Abnormal recall
2. False negatives
3. Specificity
4. Accuracy

The default threshold search targets:

`Abnormal Recall >= 0.95`

