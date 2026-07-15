import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from hms_outcome_utils import (
    build_output_df,
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

SEED = 3
TEST_SIZE = 0.2
FEATURE_SET = "base_timing_grade_position"
MODEL_PRESET = "stacked_v2"
PCA_DIM = 16
C_VALUE = 1.0
CLASS_WEIGHT = None
BINARY_THRESHOLD = 0.50
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

# Leave these as None to use the held-out validation split as a pseudo-test set.
EXTERNAL_TEST_CSV = None
EXTERNAL_TEST_FEATURES_CSV = None

OUT_DIR = RESULTS_ROOT
OUT_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HMS outcome inference.")
    parser.add_argument("--train-csv", type=str, default=TRAIN_CSV)
    parser.add_argument("--feature-csv", type=str, default=TRAIN_FEATURES_CSV)
    parser.add_argument("--external-test-csv", type=str, default=EXTERNAL_TEST_CSV)
    parser.add_argument("--external-test-feature-csv", type=str, default=EXTERNAL_TEST_FEATURES_CSV)
    parser.add_argument("--feature-set", type=str, default=FEATURE_SET)
    parser.add_argument("--model-preset", type=str, default=MODEL_PRESET)
    parser.add_argument("--threshold", type=float, default=BINARY_THRESHOLD)
    parser.add_argument("--seed", type=int, default=SEED)
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

    train_all_df = load_merged(args.train_csv, args.feature_csv, with_label=True)
    feature_set = args.feature_set
    model_preset = args.model_preset
    binary_threshold = args.threshold
    seed = args.seed
    column_info = prepare_feature_columns(train_all_df, feature_set=feature_set)

    print("=== HMS outcome inference ===")
    print("Feature set:", feature_set)
    print("Model preset:", model_preset)
    if model_preset is None:
        print("PCA dims:", PCA_DIM)
        print("C:", C_VALUE)
    print("class_weight:", CLASS_WEIGHT)
    print("Binary threshold:", binary_threshold)
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

    if args.external_test_csv and args.external_test_feature_csv:
        print("\nMode: external test inference")
        if model_preset is None:
            pipeline = fit_pipeline(
                train_all_df,
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
        else:
            pipeline = fit_pipeline_ensemble(
                train_all_df,
                feature_set=feature_set,
                ensemble_specs=get_outcome_ensemble_specs(model_preset),
                class_weight=CLASS_WEIGHT,
                max_iter=MAX_ITER,
                solver=SOLVER,
                random_state=seed,
            )

        test_df = load_merged(args.external_test_csv, args.external_test_feature_csv, with_label=False)
        prepare_feature_columns(test_df, feature_set=feature_set)
        probs = predict_positive_proba(test_df, pipeline)
        preds = (probs >= binary_threshold).astype(int)

        out_df = build_output_df(test_df, probs, preds)
        out_path = OUT_DIR / "external_test_predictions_hms_position.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        binary_out_path = OUT_DIR / "external_test_predictions_binary.csv"
        out_df[["Patient ID", "abnormal_prob", "binary_pred"]].to_csv(binary_out_path, index=False, encoding="utf-8-sig")
        print(f"Saved external test predictions to: {out_path}")
        print(f"Saved binary-only predictions to: {binary_out_path}")
        print(out_df.head(10).to_string(index=False))
        return

    print("\nMode: held-out validation split as pseudo-test")
    train_df, val_df = train_test_split(
        train_all_df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=train_all_df["label"],
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

    probs = predict_positive_proba(val_df, pipeline)
    preds = (probs >= binary_threshold).astype(int)
    metrics = evaluate_predictions(val_df["label"].values, preds)

    print(f"Pseudo-test accuracy@{binary_threshold:.2f}: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(val_df["label"].values, preds, labels=[0, 1]))
    print("Classification Report:")
    print(
        classification_report(
            val_df["label"].values,
            preds,
            target_names=["Normal", "Abnormal"],
            digits=4,
            zero_division=0,
        )
    )

    out_df = build_output_df(val_df, probs, preds)
    out_path = OUT_DIR / "pseudo_test_predictions_hms_position.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    binary_out_path = OUT_DIR / "pseudo_test_predictions_binary.csv"
    out_df[["Patient ID", "abnormal_prob", "binary_pred", "true_label"]].to_csv(
        binary_out_path, index=False, encoding="utf-8-sig"
    )
    print(f"Saved pseudo-test predictions to: {out_path}")
    print(f"Saved binary-only predictions to: {binary_out_path}")

    print("\nTop 10 highest-risk samples:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
