# HMS Repro Package

This folder is a handoff package for reproducing the final HMS front-end training line and the downstream binary outcome stage used by the best final pipeline.

It covers:

- retrain `final_hms_single_v1`
- regenerate `hms_features_final_single_v1.csv`
- retrain the final binary outcome model on the regenerated HMS features
- generate the final binary outcome output CSV

It does not focus on the 3-class display report. The reproduction target here is the binary outcome stage.

## What is included

- `train_hms_final.py`: one-command launcher for the final HMS retraining setup
- `generate_hms_features_final.py`: regenerate the final HMS feature table from the retrained checkpoint
- `check_repro_ready.py`: quick layout check before training
- `reproduce_hms.py`: one-command check + retrain + feature regeneration
- `reproduce_full_outcome.py`: one-command check + HMS retrain + feature regeneration + outcome retrain + binary outcome inference
- `requirements.txt`: Python dependencies
- `src/`: the actual training and feature-generation source files
- `data/circor/training_data.csv`: copied training metadata table

## What another person still needs

The full training wav set is required and is not duplicated here by default.

Put the training wav files here:

`model_train/data/circor/training_data/`

Expected naming format for the raw training directory:

- `12345_AV.wav`
- `12345_MV.wav`
- `12345_PV.wav`
- `12345_TV.wav`

## Recommended layout

```text
model_train/
  README.md
  requirements.txt
  check_repro_ready.py
  train_hms_final.py
  generate_hms_features_final.py
  checkpoints/
  data/
    circor/
      training_data.csv
      training_data/
        *.wav
  src/
    ...
```

## Setup

From inside `model_train/`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python check_repro_ready.py
```

If `check_repro_ready.py` reports missing wav files, copy the full training wav directory into `data/circor/training_data/` and rerun the check.

## What the HMS stage uses

The HMS training code directly reads the raw wavs under:

`model_train/data/circor/training_data/`

It does not use `training_data_clean/` or `training_data_clean_light/` for this final line.

The `dataset_hms.py` file is required because it is the loader that:

- reads the raw wav files
- slices them into 3-second segments
- normalizes them
- converts them into log-mel inputs on the fly for training

## Retrain the final HMS model

Run:

```powershell
python train_hms_final.py
```

Or run the full reproduction flow in one command:

```powershell
python reproduce_hms.py
```

## Retrain the final binary outcome stage

After the HMS feature CSV is regenerated, run:

```powershell
python train_outcome_hms_final.py
```

This uses:

- feature CSV: `model_train/data/circor/hms_features_final_single_v1.csv`
- feature set: `base_timing_grade_shape_position_persistent_no_embedding`
- model preset: `single_final_v1`
- binary threshold: `0.54`

Outputs:

- `model_train/results/outcome_hms_final_single_persistent_no_embedding_v1_results.csv`
- `model_train/results/outcome_hms_final_single_persistent_no_embedding_v1_summary.csv`

## Generate the final binary outcome output

Run:

```powershell
python run_outcome_final_inference.py
```

This writes:

- detailed output: `model_train/results/pseudo_test_predictions_hms_position.csv`
- binary-only output: `model_train/results/pseudo_test_predictions_binary.csv`

The binary-only CSV is the one to hand off when you only want:

- `Patient ID`
- `abnormal_prob`
- `binary_pred`
- `true_label`

## End-to-end reproduction

To run the whole chain in one command:

```powershell
python reproduce_full_outcome.py
```

This will run:

1. `check_repro_ready.py`
2. `train_hms_final.py`
3. `generate_hms_features_final.py`
4. `train_outcome_hms_final.py`
5. `run_outcome_final_inference.py`

Default behavior:

- seed: `0`
- device: `cuda`
- epochs: `40`
- patience: `10`
- batch size: `32`
- output checkpoint: `model_train/checkpoints/final_hms_single_v1.pth`
- output metadata: `model_train/checkpoints/final_hms_single_v1.json`

If GPU is unavailable:

```powershell
python train_hms_final.py --device cpu
```

## Regenerate the final HMS feature table

After training finishes:

```powershell
python generate_hms_features_final.py
```

Outputs:

- checkpoint: `model_train/checkpoints/final_hms_single_v1.pth`
- decision metadata: `model_train/checkpoints/final_hms_single_v1.json`
- features: `model_train/data/circor/hms_features_final_single_v1.csv`
- outcome train summary: `model_train/results/outcome_hms_final_single_persistent_no_embedding_v1_summary.csv`
- outcome binary output: `model_train/results/pseudo_test_predictions_binary.csv`

## Override paths if needed

The package defaults point to the local `model_train/data/...` and `model_train/checkpoints/...` layout.

If someone keeps the dataset elsewhere, they can override paths explicitly:

```powershell
python .\src\train_hms.py --csv-path D:\circor\training_data.csv --wav-dir D:\circor\training_data --checkpoint-path .\checkpoints\final_hms_single_v1.pth
```

```powershell
python .\src\generate_hms_features.py --csv-path D:\circor\training_data.csv --wav-dir D:\circor\training_data --checkpoint-path .\checkpoints\final_hms_single_v1.pth --output-path .\data\circor\hms_features_final_single_v1.csv
```
