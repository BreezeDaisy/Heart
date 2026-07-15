# Final 4-Position Pipeline

This folder contains the final inference pipeline for self-recorded samples.

It is fixed to:

- use all 4 auscultation positions: `AV`, `MV`, `PV`, `TV`
- run a quality check before inference
- apply conservative position-wise cleaning only when needed
- generate HMS features with the current best front-end model
- run outcome with `base_timing_grade_shape_position_persistent_no_embedding`
- build the final `green / yellow / red` screening report
- prompt for each patient's age group and sex in the terminal

## Expected raw sample naming

Put raw wav files in `sample/` using either of these formats:

- `01-AV.wav`
- `01_AV.wav`

Each patient must have all 4 files:

- `AV`
- `MV`
- `PV`
- `TV`

## One-click run

```powershell
.\.venv\Scripts\python.exe final\run_final_pipeline.py
```

Outputs are written under:

`results/final_pipeline/latest/`

During the run, the terminal will prompt for each patient's:

- age group
- sex

If you want one shared value for every patient, use non-interactive mode:

```powershell
.\.venv\Scripts\python.exe final\run_final_pipeline.py --non-interactive-metadata --age "Young Adult" --sex "Female"
```

Key files:

- `prepared_sample/`: normalized 4-point wav files
- `quality/sample_quality_check.csv`: raw quality judgement
- `cleaning/cleaning_decisions.csv`: whether each point used raw or light cleaning
- `features/hms_features_sample_final.csv`: 4-point HMS features
- `predictions/external_test_predictions.csv`: final abnormal probability
- `report/final_screening_report.csv`: final 3-class report

## Script roles

- `prepare_four_position_sample.py`: normalize raw sample names to `patient_position.wav`
- `build_sample_metadata.py`: generate metadata for the outcome stage
- `run_quality_check.py`: decide whether cleaning is needed
- `clean_four_positions.py`: clean only the positions that need light cleaning
- `generate_features.py`: run the HMS front-end and aggregate patient-level features
- `run_outcome.py`: run the persistent no-embedding outcome model
- `build_three_class_report.py`: convert outcome probability into final triage output
- `run_final_pipeline.py`: run the whole pipeline end to end
