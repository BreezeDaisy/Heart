from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_hms import POSITIONS, POSITION_TO_INDEX


AGE_MAP = {
    "Neonate": 0.0,
    "Infant": 1.0,
    "Child": 2.0,
    "Adolescent": 3.0,
    "Young Adult": 4.0,
    "Adult": 4.0,
}
SEX_MAP = {"Female": 0.0, "Male": 1.0}


class CirCorDirectOutcomeDataset(Dataset):
    """Patient-level dataset for direct outcome prediction.

    One item is one patient. The model receives all available segments from the
    patient's four auscultation positions plus clinical features. Murmur and
    murmur descriptors remain auxiliary labels, not required inference inputs.
    """

    def __init__(
        self,
        csv_path,
        wav_dir,
        sr=2000,
        segment_duration=3.0,
        segment_hop=2.0,
        n_mels=64,
        max_segments_per_patient=32,
        require_outcome=True,
    ):
        self.csv_path = Path(csv_path)
        self.wav_dir = Path(wav_dir)
        self.sr = sr
        self.segment_duration = segment_duration
        self.segment_hop = segment_hop
        self.segment_length = int(sr * segment_duration)
        self.segment_hop_length = int(sr * segment_hop)
        self.n_mels = n_mels
        self.max_segments_per_patient = max_segments_per_patient
        self.require_outcome = require_outcome

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df["Patient ID"] = self.df["Patient ID"].astype(str)
        self.df = self.df[self.df["Murmur"].isin(["Absent", "Present", "Unknown"])].copy()
        if self.require_outcome:
            self.df = self.df[self.df["Outcome"].isin(["Normal", "Abnormal"])].copy()

        self.murmur_map = {"Absent": 0, "Present": 1, "Unknown": 2}
        self.outcome_map = {"Normal": 0, "Abnormal": 1}
        self.df["murmur_label"] = self.df["Murmur"].map(self.murmur_map)
        if "Outcome" in self.df.columns:
            self.df["outcome_label"] = self.df["Outcome"].map(self.outcome_map)

        self.timing_col = "Systolic murmur timing"
        self.grade_col = "Systolic murmur grading"
        self.shape_col = "Systolic murmur shape"
        self.timing_map = self._build_aux_map(self.timing_col)
        self.grade_map = self._build_aux_map(self.grade_col)
        self.shape_map = self._build_aux_map(self.shape_col)
        self.num_timing_classes = len(self.timing_map)
        self.num_grade_classes = len(self.grade_map)
        self.num_shape_classes = len(self.shape_map)

        self.patients = []
        for _, row in self.df.iterrows():
            patient_id = str(row["Patient ID"])
            wav_paths = {
                pos: self.wav_dir / f"{patient_id}_{pos}.wav"
                for pos in POSITIONS
                if (self.wav_dir / f"{patient_id}_{pos}.wav").exists()
            }
            if not wav_paths:
                continue

            timing_label, timing_mask = self._extract_aux_target(row, self.timing_col, self.timing_map)
            grade_label, grade_mask = self._extract_aux_target(row, self.grade_col, self.grade_map)
            shape_label, shape_mask = self._extract_aux_target(row, self.shape_col, self.shape_map)

            self.patients.append(
                {
                    "patient_id": patient_id,
                    "wav_paths": wav_paths,
                    "available_positions": sorted(wav_paths.keys()),
                    "missing_positions": [pos for pos in POSITIONS if pos not in wav_paths],
                    "position_count": len(wav_paths),
                    "has_all_positions": len(wav_paths) == len(POSITIONS),
                    "clinical": self._clinical_features(row),
                    "y_outcome": int(row["outcome_label"]) if "outcome_label" in row else -1,
                    "y_murmur": int(row["murmur_label"]),
                    "y_timing": timing_label,
                    "timing_mask": timing_mask,
                    "y_grade": grade_label,
                    "grade_mask": grade_mask,
                    "y_shape": shape_label,
                    "shape_mask": shape_mask,
                }
            )

        print(f"Loaded {len(self.patients)} patients with at least one wav")

    @property
    def clinical_dim(self):
        return 10

    def _build_aux_map(self, column_name):
        if column_name not in self.df.columns:
            return {}
        values = []
        for value in self.df[column_name].dropna().unique():
            label = str(value).strip()
            if label and label.lower() != "nan":
                values.append(label)
        values = sorted(set(values))
        return {value: idx for idx, value in enumerate(values)}

    def _extract_aux_target(self, row, column_name, label_map):
        if not label_map or column_name not in row or pd.isna(row[column_name]):
            return -1, 0.0
        label_str = str(row[column_name]).strip()
        if not label_str or label_str.lower() == "nan" or label_str not in label_map:
            return -1, 0.0
        return label_map[label_str], 1.0

    def _clinical_features(self, row):
        age_raw = row.get("Age", np.nan)
        age_missing = float(pd.isna(age_raw) or str(age_raw).strip().lower() == "nan")
        age_value = AGE_MAP.get(str(age_raw).strip(), 2.0) / 4.0

        sex_raw = row.get("Sex", np.nan)
        sex_missing = float(pd.isna(sex_raw) or str(sex_raw).strip().lower() == "nan")
        sex_value = SEX_MAP.get(str(sex_raw).strip(), 0.5)

        height_raw = pd.to_numeric(row.get("Height", np.nan), errors="coerce")
        height_missing = float(pd.isna(height_raw))
        height_value = 0.0 if pd.isna(height_raw) else float(np.clip(height_raw / 200.0, 0.0, 1.5))

        weight_raw = pd.to_numeric(row.get("Weight", np.nan), errors="coerce")
        weight_missing = float(pd.isna(weight_raw))
        weight_value = 0.0 if pd.isna(weight_raw) else float(np.clip(weight_raw / 120.0, 0.0, 1.5))

        pregnancy_raw = row.get("Pregnancy status", np.nan)
        pregnancy_missing = float(pd.isna(pregnancy_raw) or str(pregnancy_raw).strip().lower() == "nan")
        pregnancy_text = str(pregnancy_raw).strip().lower()
        pregnancy_value = 1.0 if pregnancy_text == "true" else 0.0 if pregnancy_text == "false" else 0.5

        return np.asarray(
            [
                age_value,
                sex_value,
                height_value,
                weight_value,
                pregnancy_value,
                age_missing,
                sex_missing,
                height_missing,
                weight_missing,
                pregnancy_missing,
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.patients)

    def _compute_segment_starts(self, wav_len):
        if wav_len <= self.segment_length:
            return [0]
        starts = list(range(0, wav_len - self.segment_length + 1, self.segment_hop_length))
        last_start = wav_len - self.segment_length
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _load_audio(self, wav_path):
        try:
            y, _ = librosa.load(wav_path, sr=self.sr)
        except Exception as exc:
            print(f"Warning: failed to load {wav_path}: {exc}")
            return None
        return y

    def _normalize_segment(self, segment):
        if len(segment) < self.segment_length:
            segment = np.pad(segment, (0, self.segment_length - len(segment)))
        segment = segment.astype(np.float32)
        segment = segment - float(segment.mean())

        peak = float(np.max(np.abs(segment)))
        if peak > 1e-6:
            segment = segment / peak

        rms = float(np.sqrt(np.mean(segment ** 2)))
        if rms > 1e-6:
            segment = segment / rms

        return np.clip(segment, -5.0, 5.0).astype(np.float32)

    def _wav_to_logmel(self, y, n_fft, hop_length, n_mels):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        ref_value = float(np.max(mel))
        if ref_value <= 1e-10:
            return np.zeros_like(mel, dtype=np.float32)
        return librosa.power_to_db(mel, ref=ref_value).astype(np.float32)

    def _select_segments(self, segment_records):
        if len(segment_records) <= self.max_segments_per_patient:
            return segment_records

        # Keep coverage balanced across positions instead of taking only the first
        # recordings, which would bias long recordings and early positions.
        selected = []
        per_position = max(1, self.max_segments_per_patient // len(POSITIONS))
        leftovers = []
        for pos in POSITIONS:
            current = [record for record in segment_records if record["position"] == pos]
            selected.extend(current[:per_position])
            leftovers.extend(current[per_position:])
        room = self.max_segments_per_patient - len(selected)
        if room > 0:
            selected.extend(leftovers[:room])
        return selected[: self.max_segments_per_patient]

    def __getitem__(self, idx):
        patient = self.patients[idx]
        segment_records = []
        for pos in POSITIONS:
            wav_path = patient["wav_paths"].get(pos)
            if wav_path is None:
                continue
            y = self._load_audio(wav_path)
            if y is None:
                continue
            for start in self._compute_segment_starts(len(y)):
                segment_records.append({"position": pos, "start": start, "audio": y})

        segment_records = self._select_segments(segment_records)
        x_scale1 = []
        x_scale2 = []
        x_scale3 = []
        position_indices = []
        for record in segment_records:
            y = record["audio"]
            start = record["start"]
            segment = self._normalize_segment(y[start : start + self.segment_length])
            x_scale1.append(torch.tensor(self._wav_to_logmel(segment, 256, 64, self.n_mels)).unsqueeze(0))
            x_scale2.append(torch.tensor(self._wav_to_logmel(segment, 192, 48, self.n_mels)).unsqueeze(0))
            x_scale3.append(torch.tensor(self._wav_to_logmel(segment, 128, 32, self.n_mels)).unsqueeze(0))
            position_indices.append(POSITION_TO_INDEX[record["position"]])

        if not x_scale1:
            raise RuntimeError(f"No usable segments for patient {patient['patient_id']}")

        return {
            "patient_id": patient["patient_id"],
            "available_positions": patient["available_positions"],
            "position_count": patient["position_count"],
            "has_all_positions": patient["has_all_positions"],
            "x_scale1": torch.stack(x_scale1, dim=0).float(),
            "x_scale2": torch.stack(x_scale2, dim=0).float(),
            "x_scale3": torch.stack(x_scale3, dim=0).float(),
            "position_index": torch.tensor(position_indices, dtype=torch.long),
            "segment_mask": torch.ones(len(position_indices), dtype=torch.float32),
            "clinical": torch.tensor(patient["clinical"], dtype=torch.float32),
            "y_outcome": torch.tensor(patient["y_outcome"], dtype=torch.long),
            "y_murmur": torch.tensor(patient["y_murmur"], dtype=torch.long),
            "y_timing": torch.tensor(patient["y_timing"], dtype=torch.long),
            "timing_mask": torch.tensor(patient["timing_mask"], dtype=torch.float32),
            "y_grade": torch.tensor(patient["y_grade"], dtype=torch.long),
            "grade_mask": torch.tensor(patient["grade_mask"], dtype=torch.float32),
            "y_shape": torch.tensor(patient["y_shape"], dtype=torch.long),
            "shape_mask": torch.tensor(patient["shape_mask"], dtype=torch.float32),
        }


def direct_outcome_collate(batch):
    max_segments = max(item["x_scale1"].size(0) for item in batch)

    def pad_segments(key):
        tensors = []
        for item in batch:
            value = item[key]
            pad_shape = (max_segments - value.size(0), *value.shape[1:])
            if pad_shape[0] > 0:
                value = torch.cat([value, value.new_zeros(pad_shape)], dim=0)
            tensors.append(value)
        return torch.stack(tensors, dim=0)

    position_tensors = []
    mask_tensors = []
    for item in batch:
        positions = item["position_index"]
        masks = item["segment_mask"]
        if positions.size(0) < max_segments:
            pad_len = max_segments - positions.size(0)
            positions = torch.cat([positions, positions.new_zeros(pad_len)], dim=0)
            masks = torch.cat([masks, masks.new_zeros(pad_len)], dim=0)
        position_tensors.append(positions)
        mask_tensors.append(masks)

    return {
        "patient_id": [item["patient_id"] for item in batch],
        "available_positions": [item["available_positions"] for item in batch],
        "position_count": torch.tensor([item["position_count"] for item in batch], dtype=torch.long),
        "has_all_positions": torch.tensor([item["has_all_positions"] for item in batch], dtype=torch.bool),
        "x_scale1": pad_segments("x_scale1"),
        "x_scale2": pad_segments("x_scale2"),
        "x_scale3": pad_segments("x_scale3"),
        "position_index": torch.stack(position_tensors, dim=0),
        "segment_mask": torch.stack(mask_tensors, dim=0),
        "clinical": torch.stack([item["clinical"] for item in batch], dim=0),
        "y_outcome": torch.stack([item["y_outcome"] for item in batch], dim=0),
        "y_murmur": torch.stack([item["y_murmur"] for item in batch], dim=0),
        "y_timing": torch.stack([item["y_timing"] for item in batch], dim=0),
        "timing_mask": torch.stack([item["timing_mask"] for item in batch], dim=0),
        "y_grade": torch.stack([item["y_grade"] for item in batch], dim=0),
        "grade_mask": torch.stack([item["grade_mask"] for item in batch], dim=0),
        "y_shape": torch.stack([item["y_shape"] for item in batch], dim=0),
        "shape_mask": torch.stack([item["shape_mask"] for item in batch], dim=0),
    }
