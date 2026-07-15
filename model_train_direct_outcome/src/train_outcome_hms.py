import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from hms_outcome_utils import (
    evaluate_predictions,
    fit_pipeline,
    fit_pipeline_ensemble,
    get_outcome_ensemble_specs,
    load_merged,
    predict_positive_proba,
    prepare_feature_columns,
)
from paths import FINAL_FEATURE_CSV, RESULTS_ROOT, TRAIN_CSV

TRAIN_CSV = str(TRAIN_CSV)
TRAIN_FEATURES_CSV = str(FINAL_FEATURE_CSV)

SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2
FEATURE_SET = "base_timing_grade_position"
MODEL_PRESET = "stacked_v2"
PCA_DIM = 16
C_VALUE = 1.0
CLASS_WEIGHT = None
THRESHOLD = 0.50
MAX_ITER = 4000
SOLVER = "liblinear"
USE_XGB = True
LR_WEIGHT = 2.0
XGB_WEIGHT = 1.0
XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 4.0,
}

OUT_DIR = RESULTS_ROOT
OUT_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train outcome model from HMS features.")
    parser.add_argument("--train-csv", type=str, default=TRAIN_CSV)
    parser.add_argument("--feature-csv", type=str, default=TRAIN_FEATURES_CSV)
    parser.add_argument("--feature-set", type=str, default=FEATURE_SET)
    parser.add_argument("--model-preset", type=str, default=MODEL_PRESET)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outcome_hms_multiseed",
        help="Prefix for saved results files under results/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_csv = Path(args.train_csv).resolve()
    feature_csv = Path(args.feature_csv).resolve()
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not feature_csv.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {feature_csv}\n"
            "Run `python generate_hms_features_final.py` first."
        )

    merged = load_merged(args.train_csv, args.feature_csv, with_label=True)
    feature_set = args.feature_set
    model_preset = args.model_preset
    threshold = args.threshold
    column_info = prepare_feature_columns(merged, feature_set=feature_set)

    print("=== HMS outcome model ===")
    print("Feature set:", feature_set)
    print("Model preset:", model_preset)
    if model_preset is None:
        print("PCA dims:", PCA_DIM)
        print("C:", C_VALUE)
    print("class_weight:", CLASS_WEIGHT)
    print("Threshold:", threshold)
    if model_preset is None:
        print("Use XGBoost:", USE_XGB)
        print("Blend weights:", (LR_WEIGHT, XGB_WEIGHT))
    else:
        print("Ensemble members:", [spec["name"] for spec in get_outcome_ensemble_specs(model_preset)])
    print("Feature CSV:", args.feature_csv)
    print("Base columns:", column_info["used_base_cols"])
    print("Timing columns:", column_info["timing_cols"])
    print("Grade columns:", column_info["grade_cols"])
    print("Position columns:", column_info["position_cols"])
    print("Num embedding features:", len(column_info["embedding_cols"]))

    per_seed_rows = []
    best_seed_payload = None

    for seed in SEEDS:
        train_df, val_df = train_test_split(
            merged,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=merged["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if model_preset is None:
            pipeline = fit_pipeline(
                train_df,
                feature_set=feature_set,
                pca_dim=PCA_DIM,
                c_value=C_VALUE,
                class_weight=CLASS_WEIGHT,
                max_iter=MAX_ITER,
                solver=SOLVER,
                use_xgb=USE_XGB,
                xgb_params=XGB_PARAMS,
                lr_weight=LR_WEIGHT,
                xgb_weight=XGB_WEIGHT,
                random_state=seed,
            )
            pca_dim_used = pipeline["pca_dim"]
        else:
            pipeline = fit_pipeline_ensemble(
                train_df,
                feature_set=feature_set,
                ensemble_specs=get_outcome_ensemble_specs(model_preset),
                class_weight=CLASS_WEIGHT,
                max_iter=MAX_ITER,
                solver=SOLVER,
                random_state=seed,
            )
            pca_dim_used = "ensemble"

        probs = predict_positive_proba(val_df, pipeline)
        preds = (probs >= threshold).astype(int)
        metrics = evaluate_predictions(val_df["label"].values, preds)

        per_seed_rows.append(
            {
                "seed": seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "accuracy": metrics["accuracy"],
                "abnormal_precision": metrics["abnormal_precision"],
                "abnormal_recall": metrics["abnormal_recall"],
                "abnormal_f1": metrics["abnormal_f1"],
                "specificity": metrics["specificity"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "threshold": threshold,
                "model_preset": model_preset or "single_model",
                "feature_set": feature_set,
                "pca_dim": pca_dim_used,
                "C": C_VALUE,
                "class_weight": CLASS_WEIGHT,
            }
        )

        print(
            f"seed={seed} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1={metrics['abnormal_f1']:.4f} | "
            f"recall={metrics['abnormal_recall']:.4f} | "
            f"specificity={metrics['specificity']:.4f}"
        )

        if best_seed_payload is None or metrics["accuracy"] > best_seed_payload["accuracy"]:
            best_seed_payload = {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "y_true": val_df["label"].values,
                "preds": preds,
            }

    results_df = pd.DataFrame(per_seed_rows)
    summary_df = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "model_preset": model_preset or "single_model",
                "pca_dim": "ensemble" if model_preset else PCA_DIM,
                "C": C_VALUE,
                "class_weight": CLASS_WEIGHT,
                "threshold": threshold,
                "mean_accuracy": results_df["accuracy"].mean(),
                "std_accuracy": results_df["accuracy"].std(ddof=0),
                "mean_abnormal_precision": results_df["abnormal_precision"].mean(),
                "mean_abnormal_recall": results_df["abnormal_recall"].mean(),
                "mean_abnormal_f1": results_df["abnormal_f1"].mean(),
                "mean_specificity": results_df["specificity"].mean(),
            }
        ]
    )

    results_path = OUT_DIR / f"{args.output_prefix}_results.csv"
    summary_path = OUT_DIR / f"{args.output_prefix}_summary.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== Multi-seed summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved per-seed results to: {results_path}")
    print(f"Saved summary to: {summary_path}")

    if best_seed_payload is not None:
        print(f"\nBest seed by accuracy: {best_seed_payload['seed']}")
        print("Confusion Matrix:")
        print(confusion_matrix(best_seed_payload["y_true"], best_seed_payload["preds"], labels=[0, 1]))
        print("Classification Report:")
        print(
            classification_report(
                best_seed_payload["y_true"],
                best_seed_payload["preds"],
                target_names=["Normal", "Abnormal"],
                digits=4,
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()
