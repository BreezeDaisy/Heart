from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from paths import POSITION_FALLBACK_CSV

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

POSITION_FEATURE_CANDIDATES = [
    "murmur_prob_overall",
    "murmur_prob_AV",
    "murmur_prob_MV",
    "murmur_prob_PV",
    "murmur_prob_TV",
]
ENGINEERED_PREFIXES = [
    "std_prob_",
    "segment_",
    "position_present_max_",
    "position_present_mean_",
    "position_present_std_",
    "position_present_top2_mean_",
    "timing_max_",
    "grade_max_",
    "shape_max_",
    "engineered_",
]
OUTCOME_ENSEMBLE_PRESETS = {
    "single_final_v1": [
        {
            "name": "lr_pca8_c0.5",
            "pca_dim": 8,
            "c_value": 0.5,
            "use_xgb": False,
            "weight": 1.0,
        },
    ],
    "single_stable_v1": [
        {
            "name": "et_pca8",
            "backend": "extra_trees",
            "pca_dim": 8,
            "weight": 1.0,
            "extra_trees_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
        },
    ],
    "single_peak_v1": [
        {
            "name": "lr_pca8_c0.5",
            "pca_dim": 8,
            "c_value": 0.5,
            "use_xgb": False,
            "weight": 1.0,
        },
    ],
    "stable_v1": [
        {
            "name": "et_raw128",
            "backend": "extra_trees",
            "pca_dim": 0,
            "weight": 2.0,
            "extra_trees_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
        },
        {
            "name": "lr_pca8_c0.25",
            "pca_dim": 8,
            "c_value": 0.25,
            "use_xgb": False,
            "weight": 1.0,
        },
        {
            "name": "lr_pca16_c8.0",
            "pca_dim": 16,
            "c_value": 8.0,
            "use_xgb": False,
            "weight": 2.0,
        },
    ],
    "stable_v2": [
        {
            "name": "et_raw128",
            "backend": "extra_trees",
            "pca_dim": 0,
            "weight": 2.0,
            "extra_trees_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
        },
        {
            "name": "lr_pca8_c4.0",
            "pca_dim": 8,
            "c_value": 4.0,
            "use_xgb": False,
            "weight": 1.0,
        },
        {
            "name": "lr_pca16_c4.0",
            "pca_dim": 16,
            "c_value": 4.0,
            "use_xgb": False,
            "weight": 1.0,
        },
    ],
    "peak_v1": [
        {
            "name": "lr_pca8_c0.25",
            "pca_dim": 8,
            "c_value": 0.25,
            "use_xgb": False,
            "weight": 2.0,
        },
        {
            "name": "lr_pca8_c1.0",
            "pca_dim": 8,
            "c_value": 1.0,
            "use_xgb": False,
            "weight": 1.0,
        },
    ],
    "stacked_v2": [
        {
            "name": "et_raw128",
            "backend": "extra_trees",
            "pca_dim": 0,
            "weight": 2.0,
            "extra_trees_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
        },
        {
            "name": "lr_pca8_c0.25",
            "pca_dim": 8,
            "c_value": 0.25,
            "use_xgb": False,
            "weight": 1.0,
        },
        {
            "name": "lr_pca8_c0.5",
            "pca_dim": 8,
            "c_value": 0.5,
            "use_xgb": False,
            "weight": 1.0,
        },
    ],
    "stacked_v1": [
        {
            "name": "lr_raw128_c0.25",
            "pca_dim": 0,
            "c_value": 0.25,
            "use_xgb": False,
            "weight": 1.0,
        },
        {
            "name": "lr_pca16_c4.0",
            "pca_dim": 16,
            "c_value": 4.0,
            "use_xgb": False,
            "weight": 1.0,
        },
        {
            "name": "xgb_pca16",
            "pca_dim": 16,
            "c_value": 1.0,
            "use_xgb": True,
            "lr_weight": 0.0,
            "xgb_weight": 1.0,
            "weight": 1.0,
            "xgb_params": {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_lambda": 4.0,
            },
        },
    ],
}


@dataclass
class ScreeningDecision:
    risk_level: str
    title: str
    recommendation: str


def decision_from_prob(prob, low_th=0.40, high_th=0.62):
    if prob < low_th:
        return ScreeningDecision(
            risk_level="low_risk",
            title="当前未见明显异常",
            recommendation="建议日常观察；若后续出现不适或复测偏高，请再次检测或就医。",
        )
    if prob < high_th:
        return ScreeningDecision(
            risk_level="review",
            title="建议复测",
            recommendation="建议在更安静环境下重复检测 2~3 次；若风险仍偏高，建议进一步检查。",
        )
    return ScreeningDecision(
        risk_level="high_risk",
        title="建议尽快进一步检查",
        recommendation="当前结果偏高风险，建议尽快到正规医疗机构进一步检查。",
    )


def _score_from_row(row, column_name):
    if row is None or column_name not in row:
        return 0.0
    value = pd.to_numeric(row[column_name], errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def murmur_triage_scores(row):
    prob_present = _score_from_row(row, "prob_present")
    murmur_overall = _score_from_row(row, "murmur_prob_overall")
    max_present = _score_from_row(row, "max_prob_present")
    present_signal = float(np.clip(max(prob_present, murmur_overall, max_present), 0.0, 1.0))

    holosystolic = _score_from_row(row, "timing_prob_holosystolic")
    grade_iii = _score_from_row(row, "grade_prob_iii_vi")
    pathologic_like = float(
        np.clip(
            0.45 * present_signal
            + 0.25 * present_signal * holosystolic
            + 0.20 * present_signal * grade_iii
            + 0.10 * max_present,
            0.0,
            1.0,
        )
    )
    return {
        "murmur_present_signal": present_signal,
        "pathologic_like_murmur_score": pathologic_like,
        "holosystolic_murmur_score": float(np.clip(present_signal * holosystolic, 0.0, 1.0)),
        "high_grade_murmur_score": float(np.clip(present_signal * grade_iii, 0.0, 1.0)),
    }


def has_murmur_review_cue(row):
    if row is not None and "pathologic_like_murmur_score" in row:
        scores = {
            "murmur_present_signal": _score_from_row(row, "murmur_present_signal"),
            "pathologic_like_murmur_score": _score_from_row(row, "pathologic_like_murmur_score"),
            "holosystolic_murmur_score": _score_from_row(row, "holosystolic_murmur_score"),
            "high_grade_murmur_score": _score_from_row(row, "high_grade_murmur_score"),
        }
    else:
        scores = murmur_triage_scores(row)
    return (
        scores["pathologic_like_murmur_score"] >= 0.55
        or scores["holosystolic_murmur_score"] >= 0.40
        or scores["high_grade_murmur_score"] >= 0.25
        or scores["murmur_present_signal"] >= 0.90
    )


def decision_from_prob(prob, low_th=0.14, high_th=0.54, row=None):
    if prob < low_th:
        if has_murmur_review_cue(row):
            return ScreeningDecision(
                risk_level="review",
                title="建议复测",
                recommendation="最终风险分较低，但 HMS 三分类检测到较明显的心音异常提示特征。建议在安静环境下重新采集并复测。",
            )
        return ScreeningDecision(
            risk_level="low_risk",
            title="当前未见明显异常",
            recommendation="建议日常观察；若后续出现不适或复测偏高，请再次检测或就医。",
        )
    if prob < high_th:
        return ScreeningDecision(
            risk_level="review",
            title="建议复测",
            recommendation="建议在更安静环境下重复检测 2~3 次；若风险仍偏高，建议进一步检查。",
        )
    return ScreeningDecision(
        risk_level="high_risk",
        title="建议尽快进一步检查",
        recommendation="当前结果偏高风险，建议尽快到正规医疗机构进一步检查。",
    )


def evaluate_predictions(y_true, preds):
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    tp = int(((y_true == 1) & (preds == 1)).sum())

    accuracy = float((preds == y_true).mean())
    abnormal_precision = float(tp / max(tp + fp, 1))
    abnormal_recall = float(tp / max(tp + fn, 1))
    abnormal_f1 = float(
        2.0 * abnormal_precision * abnormal_recall / max(abnormal_precision + abnormal_recall, 1e-8)
    )
    specificity = float(tn / max(tn + fp, 1))

    return {
        "accuracy": accuracy,
        "abnormal_precision": abnormal_precision,
        "abnormal_recall": abnormal_recall,
        "abnormal_f1": abnormal_f1,
        "specificity": specificity,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int),
    }


def load_merged(csv_path, feat_csv_path, with_label=True):
    outcome_df = pd.read_csv(csv_path)
    feat_df = pd.read_csv(feat_csv_path)

    outcome_df.columns = [c.strip() for c in outcome_df.columns]
    feat_df.columns = [c.strip() for c in feat_df.columns]

    missing_position_cols = [col for col in POSITION_FEATURE_CANDIDATES if col not in feat_df.columns]
    if missing_position_cols and POSITION_FALLBACK_CSV.exists():
        pos_df = pd.read_csv(POSITION_FALLBACK_CSV)
        pos_df.columns = [c.strip() for c in pos_df.columns]
        add_cols = ["Patient ID"] + [col for col in missing_position_cols if col in pos_df.columns]
        feat_df = pd.merge(feat_df, pos_df[add_cols], on="Patient ID", how="left")

    outcome_df = outcome_df.drop_duplicates(subset=["Patient ID"]).copy()
    merged = pd.merge(outcome_df, feat_df, on="Patient ID", how="inner")

    for column_name in ["Age", "Sex", "Height", "Weight", "Pregnancy status", "Outcome", "Murmur"]:
        if column_name not in merged.columns:
            left_name = f"{column_name}_x"
            right_name = f"{column_name}_y"
            if left_name in merged.columns:
                merged[column_name] = merged[left_name]
            elif right_name in merged.columns:
                merged[column_name] = merged[right_name]

    age_map = {
        "Neonate": 0,
        "Infant": 1,
        "Child": 2,
        "Adolescent": 3,
        # CirCor training data has no explicit adult age bucket. Its missing-age
        # group is mostly adult/pregnancy records, so local young-adult recordings
        # are safer as the learned unknown/adult-like bucket than as child/adolescent.
        "Adult": -1,
        "Young Adult": -1,
    }
    merged["Age"] = merged["Age"].map(age_map).fillna(-1)

    sex_map = {"Female": 0, "Male": 1}
    merged["Sex"] = merged["Sex"].map(sex_map).fillna(-1)

    for column_name in ["Height", "Weight", "Pregnancy status"]:
        if column_name in merged.columns:
            merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce").fillna(-1)

    if with_label:
        merged = merged[merged["Outcome"].isin(["Normal", "Abnormal"])].copy()
        merged["label"] = merged["Outcome"].map({"Normal": 0, "Abnormal": 1})

    if {"prob_absent", "prob_present", "prob_unknown"}.issubset(merged.columns):
        merged["engineered_present_minus_absent"] = merged["prob_present"] - merged["prob_absent"]
        merged["engineered_present_plus_unknown"] = merged["prob_present"] + merged["prob_unknown"]
        merged["engineered_prob_entropy"] = -(
            merged[["prob_absent", "prob_present", "prob_unknown"]].clip(1e-6, 1.0)
            * np.log(merged[["prob_absent", "prob_present", "prob_unknown"]].clip(1e-6, 1.0))
        ).sum(axis=1)

    if {"max_prob_absent", "max_prob_present"}.issubset(merged.columns):
        merged["engineered_max_present_minus_absent"] = merged["max_prob_present"] - merged["max_prob_absent"]
        merged["engineered_max_present_ratio"] = merged["max_prob_present"] / (merged["max_prob_absent"] + 1e-6)

    if {"max_prob_present", "prob_present"}.issubset(merged.columns):
        merged["engineered_segment_present_gain"] = merged["max_prob_present"] - merged["prob_present"]

    if {"max_prob_absent", "prob_absent"}.issubset(merged.columns):
        merged["engineered_segment_absent_gain"] = merged["max_prob_absent"] - merged["prob_absent"]

    position_prob_cols = [col for col in POSITION_FEATURE_CANDIDATES if col in merged.columns]
    if position_prob_cols:
        merged["engineered_position_prob_range"] = merged[position_prob_cols].max(axis=1) - merged[position_prob_cols].min(axis=1)
        merged["engineered_position_prob_std"] = merged[position_prob_cols].std(axis=1)

    position_max_cols = sorted([c for c in merged.columns if c.startswith("position_present_max_")])
    if position_max_cols:
        merged["engineered_position_max_range"] = merged[position_max_cols].max(axis=1) - merged[position_max_cols].min(axis=1)
        merged["engineered_position_max_std"] = merged[position_max_cols].std(axis=1)
        merged["engineered_position_max_peak"] = merged[position_max_cols].max(axis=1)

    return merged


def prepare_feature_columns(merged, feature_set="base_timing_grade_position"):
    base_feature_cols = [
        "prob_absent",
        "prob_present",
        "prob_unknown",
        "max_prob_absent",
        "max_prob_present",
        "max_prob_unknown",
        "num_segments",
        "Age",
        "Sex",
    ]
    persistent_base_feature_cols = [
        "prob_absent",
        "prob_present",
        "prob_unknown",
        "num_segments",
        "Age",
        "Sex",
    ]
    calibrated_base_feature_cols = [
        c
        for c in [
            "calib_prob_absent",
            "calib_prob_present",
            "calib_prob_unknown",
            "calib_max_prob_absent",
            "calib_max_prob_present",
            "calib_max_prob_unknown",
        ]
        if c in merged.columns
    ]
    calibrated_persistent_base_feature_cols = [
        c
        for c in [
            "calib_prob_absent",
            "calib_prob_present",
            "calib_prob_unknown",
        ]
        if c in merged.columns
    ]
    timing_feature_cols = sorted([c for c in merged.columns if c.startswith("timing_prob_")])
    grade_feature_cols = sorted([c for c in merged.columns if c.startswith("grade_prob_")])
    shape_feature_cols = sorted([c for c in merged.columns if c.startswith("shape_prob_")])
    position_feature_cols = [c for c in POSITION_FEATURE_CANDIDATES if c in merged.columns]
    calibrated_position_feature_cols = [
        c
        for c in [
            "calib_murmur_prob_overall",
            "calib_murmur_prob_AV",
            "calib_murmur_prob_MV",
            "calib_murmur_prob_PV",
            "calib_murmur_prob_TV",
        ]
        if c in merged.columns
    ]
    clinical_feature_cols = [c for c in ["Height", "Weight", "Pregnancy status"] if c in merged.columns]
    rich_feature_cols = sorted(
        [
            c
            for c in merged.columns
            if any(c.startswith(prefix) for prefix in ENGINEERED_PREFIXES)
        ]
    )
    persistent_feature_cols = sorted(
        [
            c
            for c in merged.columns
            if c.startswith(
                (
                    "std_prob_",
                    "segment_present_frac_ge_",
                    "segment_margin_mean",
                    "segment_margin_std",
                    "segment_entropy_mean",
                    "segment_entropy_std",
                    "position_present_mean_",
                    "position_present_std_",
                    "position_present_top2_mean_",
                    "engineered_present_",
                    "engineered_prob_entropy",
                    "engineered_position_prob_",
                )
            )
        ]
    )
    peak_persistent_feature_cols = sorted(
        [
            c
            for c in merged.columns
            if c.startswith(
                (
                    "std_prob_present",
                    "segment_present_top3_mean",
                    "segment_present_frac_ge_",
                    "position_present_max_",
                    "position_present_mean_",
                    "position_present_std_",
                    "position_present_top2_mean_",
                    "engineered_max_present_",
                    "engineered_segment_present_gain",
                )
            )
        ]
    )
    embedding_cols = [c for c in merged.columns if c.startswith("embedding_")]

    feature_sets = {
        "base_timing_grade": base_feature_cols + timing_feature_cols + grade_feature_cols,
        "base_timing_grade_position": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
        ),
        "base_timing_grade_shape_position_clinical_raw": (
            base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + clinical_feature_cols
        ),
        "base_timing_grade_position_rich": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + rich_feature_cols
        ),
        "base_timing_grade_shape_position": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
        ),
        "base_timing_grade_shape_position_rich": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + rich_feature_cols
        ),
        "base_timing_grade_shape_position_clinical": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + clinical_feature_cols
        ),
        "base_timing_grade_shape_position_clinical_murmur_score": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + clinical_feature_cols
            + ["engineered_murmur_outcome_risk_score"]
        ),
        "base_timing_grade_shape_position_rich_clinical": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + rich_feature_cols
            + clinical_feature_cols
        ),
        "base_timing_grade_shape_position_persistent": (
            persistent_base_feature_cols
            + calibrated_persistent_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + persistent_feature_cols
        ),
        "base_timing_grade_shape_position_persistent_no_embedding": (
            persistent_base_feature_cols
            + calibrated_persistent_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + persistent_feature_cols
        ),
        "base_timing_grade_shape_position_persistent_clinical": (
            persistent_base_feature_cols
            + calibrated_persistent_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + persistent_feature_cols
            + clinical_feature_cols
        ),
        "base_timing_grade_shape_position_peak_persistent": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + peak_persistent_feature_cols
        ),
        "base_timing_grade_shape_position_peak_persistent_clinical": (
            base_feature_cols
            + calibrated_base_feature_cols
            + timing_feature_cols
            + grade_feature_cols
            + shape_feature_cols
            + position_feature_cols
            + calibrated_position_feature_cols
            + peak_persistent_feature_cols
            + clinical_feature_cols
        ),
    }

    if feature_set not in feature_sets:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    used_base_cols = feature_sets[feature_set]
    used_embedding_cols = [] if feature_set.endswith("_no_embedding") else embedding_cols
    all_feature_cols = used_base_cols + used_embedding_cols

    for column_name in all_feature_cols:
        if column_name not in merged.columns:
            merged[column_name] = 0.0

    merged[all_feature_cols] = merged[all_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(-1)

    return {
        "feature_set": feature_set,
        "used_base_cols": used_base_cols,
        "embedding_cols": used_embedding_cols,
        "timing_cols": timing_feature_cols,
        "grade_cols": grade_feature_cols,
        "shape_cols": shape_feature_cols,
        "position_cols": position_feature_cols,
        "clinical_cols": clinical_feature_cols,
        "rich_cols": rich_feature_cols,
    }


def get_outcome_ensemble_specs(preset_name):
    if preset_name not in OUTCOME_ENSEMBLE_PRESETS:
        raise ValueError(f"Unsupported outcome ensemble preset: {preset_name}")

    return [dict(spec) for spec in OUTCOME_ENSEMBLE_PRESETS[preset_name]]


def fit_pipeline(
    train_df,
    feature_set="base_timing_grade_position",
    pca_dim=8,
    c_value=0.5,
    class_weight=None,
    max_iter=4000,
    solver="liblinear",
    use_xgb=False,
    xgb_params=None,
    lr_weight=1.0,
    xgb_weight=1.0,
    random_state=0,
):
    column_info = prepare_feature_columns(train_df, feature_set=feature_set)

    x_base = train_df[column_info["used_base_cols"]].reset_index(drop=True)
    x_emb = train_df[column_info["embedding_cols"]].reset_index(drop=True)
    y_train = train_df["label"].reset_index(drop=True)

    if column_info["embedding_cols"]:
        emb_scaler = StandardScaler()
        x_emb_scaled = emb_scaler.fit_transform(x_emb)
        if pca_dim and pca_dim > 0:
            effective_pca_dim = max(1, min(pca_dim, x_emb_scaled.shape[0], x_emb_scaled.shape[1]))
            pca = PCA(n_components=effective_pca_dim)
            x_emb_features = pca.fit_transform(x_emb_scaled)
        else:
            effective_pca_dim = 0
            pca = None
            x_emb_features = x_emb_scaled
    else:
        effective_pca_dim = 0
        emb_scaler = None
        pca = None
        x_emb_features = np.empty((len(train_df), 0), dtype=np.float32)

    x_train_final = np.concatenate([x_base.values, x_emb_features], axis=1)

    final_scaler = StandardScaler()
    x_train_final = final_scaler.fit_transform(x_train_final)

    logreg_model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        C=c_value,
        solver=solver,
        random_state=random_state,
    )
    logreg_model.fit(x_train_final, y_train.values)

    x_raw_train = np.concatenate([x_base.values, x_emb_features], axis=1)
    xgb_model = None
    if use_xgb:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for use_xgb=True. Install it with `pip install xgboost`.")
        if xgb_params is None:
            xgb_params = {}
        negative_count = float((y_train.values == 0).sum())
        positive_count = float((y_train.values == 1).sum())
        scale_pos_weight = negative_count / max(positive_count, 1.0)
        xgb_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
            **xgb_params,
        )
        xgb_model.fit(x_raw_train, y_train.values)

    return {
        **column_info,
        "emb_scaler": emb_scaler,
        "pca": pca,
        "final_scaler": final_scaler,
        "model": logreg_model,
        "logreg_model": logreg_model,
        "xgb_model": xgb_model,
        "pca_dim": effective_pca_dim,
        "c_value": c_value,
        "class_weight": class_weight,
        "solver": solver,
        "use_xgb": use_xgb,
        "xgb_params": xgb_params,
        "lr_weight": lr_weight,
        "xgb_weight": xgb_weight,
        "random_state": random_state,
    }


def fit_extra_trees_pipeline(
    train_df,
    feature_set="base_timing_grade_position",
    pca_dim=0,
    extra_trees_params=None,
    random_state=0,
):
    column_info = prepare_feature_columns(train_df, feature_set=feature_set)

    x_base = train_df[column_info["used_base_cols"]].reset_index(drop=True)
    x_emb = train_df[column_info["embedding_cols"]].reset_index(drop=True)
    y_train = train_df["label"].reset_index(drop=True)

    if column_info["embedding_cols"]:
        emb_scaler = StandardScaler()
        x_emb_scaled = emb_scaler.fit_transform(x_emb)
        if pca_dim and pca_dim > 0:
            effective_pca_dim = max(1, min(pca_dim, x_emb_scaled.shape[0], x_emb_scaled.shape[1]))
            pca = PCA(n_components=effective_pca_dim)
            x_emb_features = pca.fit_transform(x_emb_scaled)
        else:
            effective_pca_dim = 0
            pca = None
            x_emb_features = x_emb_scaled
    else:
        effective_pca_dim = 0
        emb_scaler = None
        pca = None
        x_emb_features = np.empty((len(train_df), 0), dtype=np.float32)

    x_train_raw = np.concatenate([x_base.values, x_emb_features], axis=1)

    if extra_trees_params is None:
        extra_trees_params = {}

    model = ExtraTreesClassifier(
        random_state=random_state,
        n_jobs=1,
        **extra_trees_params,
    )
    model.fit(x_train_raw, y_train.values)

    return {
        **column_info,
        "emb_scaler": emb_scaler,
        "pca": pca,
        "final_scaler": None,
        "model": model,
        "tree_model": model,
        "model_backend": "extra_trees",
        "pca_dim": effective_pca_dim,
        "extra_trees_params": extra_trees_params,
        "random_state": random_state,
    }


def fit_pipeline_ensemble(
    train_df,
    feature_set="base_timing_grade_position",
    ensemble_specs=None,
    class_weight=None,
    max_iter=4000,
    solver="liblinear",
    random_state=0,
):
    if not ensemble_specs:
        raise ValueError("ensemble_specs must be a non-empty list.")

    fitted_models = []
    for index, spec in enumerate(ensemble_specs):
        backend = spec.get("backend", "hybrid")
        if backend == "extra_trees":
            fitted_pipeline = fit_extra_trees_pipeline(
                train_df,
                feature_set=feature_set,
                pca_dim=spec.get("pca_dim", 0),
                extra_trees_params=spec.get("extra_trees_params"),
                random_state=random_state,
            )
        else:
            fitted_pipeline = fit_pipeline(
                train_df,
                feature_set=feature_set,
                pca_dim=spec.get("pca_dim", 8),
                c_value=spec.get("c_value", 0.5),
                class_weight=spec.get("class_weight", class_weight),
                max_iter=spec.get("max_iter", max_iter),
                solver=spec.get("solver", solver),
                use_xgb=spec.get("use_xgb", False),
                xgb_params=spec.get("xgb_params"),
                lr_weight=spec.get("lr_weight", 1.0),
                xgb_weight=spec.get("xgb_weight", 1.0),
                random_state=random_state,
            )
        fitted_models.append(
            {
                "name": spec.get("name", f"model_{index}"),
                "weight": float(spec.get("weight", 1.0)),
                "pipeline": fitted_pipeline,
                "spec": spec,
            }
        )

    column_info = prepare_feature_columns(train_df.copy(), feature_set=feature_set)
    return {
        **column_info,
        "ensemble_models": fitted_models,
        "model_preset": "custom_ensemble",
        "pca_dim": "ensemble",
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
    }


def transform_features(df, pipeline):
    df = df.copy()
    used_columns = pipeline["used_base_cols"] + pipeline["embedding_cols"]
    for column_name in used_columns:
        if column_name not in df.columns:
            df[column_name] = 0.0

    df[used_columns] = df[used_columns].apply(pd.to_numeric, errors="coerce").fillna(-1)

    x_base = df[pipeline["used_base_cols"]].reset_index(drop=True)
    x_emb = df[pipeline["embedding_cols"]].reset_index(drop=True)

    if pipeline["embedding_cols"]:
        x_emb_scaled = pipeline["emb_scaler"].transform(x_emb)
        if pipeline["pca"] is not None:
            x_emb_features = pipeline["pca"].transform(x_emb_scaled)
        else:
            x_emb_features = x_emb_scaled
    else:
        x_emb_features = np.empty((len(df), 0), dtype=np.float32)

    x_raw = np.concatenate([x_base.values, x_emb_features], axis=1)
    if pipeline["final_scaler"] is None:
        x_scaled = x_raw
    else:
        x_scaled = pipeline["final_scaler"].transform(x_raw)
    return x_scaled


def transform_feature_blocks(df, pipeline):
    df = df.copy()
    used_columns = pipeline["used_base_cols"] + pipeline["embedding_cols"]
    for column_name in used_columns:
        if column_name not in df.columns:
            df[column_name] = 0.0

    df[used_columns] = df[used_columns].apply(pd.to_numeric, errors="coerce").fillna(-1)

    x_base = df[pipeline["used_base_cols"]].reset_index(drop=True)
    x_emb = df[pipeline["embedding_cols"]].reset_index(drop=True)

    if pipeline["embedding_cols"]:
        x_emb_scaled = pipeline["emb_scaler"].transform(x_emb)
        if pipeline["pca"] is not None:
            x_emb_features = pipeline["pca"].transform(x_emb_scaled)
        else:
            x_emb_features = x_emb_scaled
    else:
        x_emb_features = np.empty((len(df), 0), dtype=np.float32)

    x_raw = np.concatenate([x_base.values, x_emb_features], axis=1)
    if pipeline["final_scaler"] is None:
        x_scaled = x_raw
    else:
        x_scaled = pipeline["final_scaler"].transform(x_raw)
    return x_raw, x_scaled


def predict_positive_proba(df, pipeline):
    if "ensemble_models" in pipeline:
        total_probs = np.zeros(len(df), dtype=np.float32)
        total_weight = 0.0
        for ensemble_item in pipeline["ensemble_models"]:
            weight = float(ensemble_item.get("weight", 1.0))
            total_probs += weight * predict_positive_proba(df, ensemble_item["pipeline"])
            total_weight += weight
        return total_probs / max(total_weight, 1e-8)

    x_raw, x_scaled = transform_feature_blocks(df, pipeline)
    if pipeline.get("model_backend") == "extra_trees":
        return pipeline["tree_model"].predict_proba(x_raw)[:, 1]

    logreg_probs = pipeline["logreg_model"].predict_proba(x_scaled)[:, 1]

    if pipeline.get("xgb_model") is None:
        return logreg_probs

    xgb_probs = pipeline["xgb_model"].predict_proba(x_raw)[:, 1]
    lr_weight = float(pipeline.get("lr_weight", 1.0))
    xgb_weight = float(pipeline.get("xgb_weight", 1.0))
    return (lr_weight * logreg_probs + xgb_weight * xgb_probs) / max(lr_weight + xgb_weight, 1e-8)


def build_output_df(df, probs, preds):
    out_df = pd.DataFrame(
        {
            "Patient ID": df["Patient ID"].values,
            "abnormal_prob": probs,
            "binary_pred": preds,
        }
    )

    titles = []
    risk_levels = []
    recommendations = []
    for (_, row), prob in zip(df.iterrows(), probs):
        decision = decision_from_prob(float(prob), row=row)
        titles.append(decision.title)
        risk_levels.append(decision.risk_level)
        recommendations.append(decision.recommendation)

    out_df["risk_level"] = risk_levels
    out_df["display_title"] = titles
    out_df["recommendation"] = recommendations

    if "Outcome" in df.columns:
        out_df["true_outcome"] = df["Outcome"].values
    if "label" in df.columns:
        out_df["true_label"] = df["label"].values

    return out_df.sort_values(by="abnormal_prob", ascending=False).reset_index(drop=True)


def print_triage_summary(out_df):
    print("\n=== Three-way triage summary ===")
    counts = out_df["risk_level"].value_counts().reindex(["high_risk", "review", "low_risk"]).fillna(0).astype(int)
    for level_name, count in counts.items():
        print(f"{level_name}: {count}")

    if "true_label" not in out_df.columns:
        return

    high_df = out_df[out_df["risk_level"] == "high_risk"]
    review_df = out_df[out_df["risk_level"] == "review"]
    low_df = out_df[out_df["risk_level"] == "low_risk"]

    def frac_true_abnormal(df_part):
        if len(df_part) == 0:
            return 0.0
        return float((df_part["true_label"] == 1).mean())

    def frac_true_normal(df_part):
        if len(df_part) == 0:
            return 0.0
        return float((df_part["true_label"] == 0).mean())

    print(f"high_risk precision (true abnormal ratio): {frac_true_abnormal(high_df):.4f}")
    print(f"low_risk safety (true normal ratio): {frac_true_normal(low_df):.4f}")
    print(f"review ratio: {len(review_df) / max(len(out_df), 1):.4f}")
    print(f"abnormal in low_risk: {int((low_df['true_label'] == 1).sum())}")
    print(f"normal in high_risk: {int((high_df['true_label'] == 0).sum())}")
