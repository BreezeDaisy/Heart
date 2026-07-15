from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

POSITIONS = ["AV", "MV", "PV", "TV"]
POSITION_TO_INDEX = {pos: idx for idx, pos in enumerate(POSITIONS)}


class CirCorHMSDataset(Dataset):
    def __init__(
        self,
        csv_path,
        wav_dir,
        sr=2000,
        segment_duration=3.0,
        segment_hop=2.0,
        n_mels=64,
    ):
        self.csv_path = Path(csv_path)
        self.wav_dir = Path(wav_dir)
        self.sr = sr
        self.segment_duration = segment_duration
        self.segment_hop = segment_hop
        self.segment_length = int(sr * segment_duration)
        self.segment_hop_length = int(sr * segment_hop)
        self.n_mels = n_mels

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df = self.df[self.df["Murmur"].isin(["Absent", "Present", "Unknown"])].copy()

        self.murmur_map = {
            "Absent": 0,
            "Present": 1,
            "Unknown": 2,
        }
        self.df["murmur_label"] = self.df["Murmur"].map(self.murmur_map)

        self.timing_col = "Systolic murmur timing"
        self.grade_col = "Systolic murmur grading"
        self.shape_col = "Systolic murmur shape"

        self.timing_map = self._build_aux_map(self.timing_col)
        self.grade_map = self._build_aux_map(self.grade_col)
        self.shape_map = self._build_aux_map(self.shape_col)

        self.num_timing_classes = len(self.timing_map)
        self.num_grade_classes = len(self.grade_map)
        self.num_shape_classes = len(self.shape_map)

        self.samples = []

        patient_label_counts = {0: 0, 1: 0, 2: 0}
        segment_label_counts = {0: 0, 1: 0, 2: 0}

        for _, row in self.df.iterrows():
            patient_id = str(row["Patient ID"])
            murmur_label = int(row["murmur_label"])
            patient_label_counts[murmur_label] += 1

            timing_label, timing_mask = self._extract_aux_target(row, self.timing_col, self.timing_map)
            grade_label, grade_mask = self._extract_aux_target(row, self.grade_col, self.grade_map)
            shape_label, shape_mask = self._extract_aux_target(row, self.shape_col, self.shape_map)

            for pos in POSITIONS:
                wav_path = self.wav_dir / f"{patient_id}_{pos}.wav"
                if not wav_path.exists():
                    continue

                try:
                    y, _ = librosa.load(wav_path, sr=self.sr)
                except Exception as exc:
                    print(f"Warning: failed to load {wav_path}: {exc}")
                    continue

                starts = self._compute_segment_starts(len(y))
                for seg_idx, start in enumerate(starts):
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "position": pos,
                            "position_index": POSITION_TO_INDEX[pos],
                            "wav_path": str(wav_path),
                            "segment_index": seg_idx,
                            "start": start,
                            "y_murmur": murmur_label,
                            "y_timing": timing_label,
                            "timing_mask": timing_mask,
                            "y_grade": grade_label,
                            "grade_mask": grade_mask,
                            "y_shape": shape_label,
                            "shape_mask": shape_mask,
                        }
                    )
                    segment_label_counts[murmur_label] += 1

        print(f"Loaded {len(self.df)} patients")
        print(
            f"Patient labels - "
            f"Absent: {patient_label_counts[0]}, "
            f"Present: {patient_label_counts[1]}, "
            f"Unknown: {patient_label_counts[2]}"
        )
        print(f"Loaded {len(self.samples)} segments")
        print(
            f"Segment labels - "
            f"Absent: {segment_label_counts[0]}, "
            f"Present: {segment_label_counts[1]}, "
            f"Unknown: {segment_label_counts[2]}"
        )
        print("Timing classes:", self.timing_map)
        print("Grade classes:", self.grade_map)
        print("Shape classes:", self.shape_map)

    def _build_aux_map(self, column_name):
        if column_name not in self.df.columns:
            return {}
        values = sorted([str(x).strip() for x in self.df[column_name].dropna().unique()])
        return {value: idx for idx, value in enumerate(values)}

    def _extract_aux_target(self, row, column_name, label_map):
        if not label_map or column_name not in row or pd.isna(row[column_name]):
            return -1, 0

        label_str = str(row[column_name]).strip()
        if label_str not in label_map:
            return -1, 0

        return label_map[label_str], 1

    def __len__(self):
        return len(self.samples)

    def _compute_segment_starts(self, wav_len):
        if wav_len <= self.segment_length:
            return [0]

        starts = list(range(0, wav_len - self.segment_length + 1, self.segment_hop_length))
        last_start = wav_len - self.segment_length
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _load_segment(self, wav_path, start):
        y, _ = librosa.load(wav_path, sr=self.sr)
        end = start + self.segment_length
        segment = y[start:end]

        if len(segment) < self.segment_length:
            pad_len = self.segment_length - len(segment)
            segment = np.pad(segment, (0, pad_len))

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

        logmel = librosa.power_to_db(mel, ref=ref_value)
        return logmel.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        y = self._load_segment(sample["wav_path"], sample["start"])

        logmel_scale1 = self._wav_to_logmel(y, n_fft=256, hop_length=64, n_mels=64)
        logmel_scale2 = self._wav_to_logmel(y, n_fft=192, hop_length=48, n_mels=64)
        logmel_scale3 = self._wav_to_logmel(y, n_fft=128, hop_length=32, n_mels=64)

        return {
            "patient_id": sample["patient_id"],
            "position": sample["position"],
            "position_index": torch.tensor(sample["position_index"], dtype=torch.long),
            "segment_index": sample["segment_index"],
            "x_scale1": torch.tensor(logmel_scale1, dtype=torch.float32).unsqueeze(0),
            "x_scale2": torch.tensor(logmel_scale2, dtype=torch.float32).unsqueeze(0),
            "x_scale3": torch.tensor(logmel_scale3, dtype=torch.float32).unsqueeze(0),
            "y_murmur": torch.tensor(sample["y_murmur"], dtype=torch.long),
            "y_timing": torch.tensor(sample["y_timing"], dtype=torch.long),
            "timing_mask": torch.tensor(sample["timing_mask"], dtype=torch.float32),
            "y_grade": torch.tensor(sample["y_grade"], dtype=torch.long),
            "grade_mask": torch.tensor(sample["grade_mask"], dtype=torch.float32),
            "y_shape": torch.tensor(sample["y_shape"], dtype=torch.long),
            "shape_mask": torch.tensor(sample["shape_mask"], dtype=torch.float32),
        }
