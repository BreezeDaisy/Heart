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


class CirCorAudioPrimaryDataset(Dataset):
    """Outcome dataset that removes explicit clinical/collection shortcuts.

    The model receives:
    - heart sound log-mel segments
    - auscultation position for each segment
    - handcrafted acoustic descriptors per segment
    - weak clinical values only: age, sex, height, weight

    It deliberately does not expose missing flags, pregnancy status,
    position_count, has_all_positions, or per-position presence flags to the
    model. Those fields are still returned as metadata for diagnostics.
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
        training=False,
        position_dropout_prob=0.0,
        min_positions_after_dropout=1,
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
        self.training = training
        self.position_dropout_prob = position_dropout_prob
        self.min_positions_after_dropout = min_positions_after_dropout
        self.murmur_map = {"Absent": 0, "Present": 1, "Unknown": 2}
        self.outcome_map = {"Normal": 0, "Abnormal": 1}

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [column.strip() for column in self.df.columns]
        self.df["Patient ID"] = self.df["Patient ID"].astype(str)
        self.df = self.df[self.df["Murmur"].isin(["Absent", "Present", "Unknown"])].copy()
        self.df = self.df[self.df["Outcome"].isin(["Normal", "Abnormal"])].copy()

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
            self.patients.append(
                {
                    "patient_id": patient_id,
                    "wav_paths": wav_paths,
                    "available_positions": sorted(wav_paths.keys()),
                    "position_count": len(wav_paths),
                    "has_all_positions": len(wav_paths) == len(POSITIONS),
                    "weak_clinical": self._weak_clinical(row),
                    "y_outcome": int(self.outcome_map[row["Outcome"]]),
                    "y_murmur": int(self.murmur_map[row["Murmur"]]),
                    "raw": row.to_dict(),
                }
            )
        print(f"Loaded {len(self.patients)} patients with at least one wav")

    @property
    def weak_clinical_dim(self):
        return 4

    @property
    def acoustic_dim(self):
        return 12

    def _weak_clinical(self, row):
        age_raw = row.get("Age", np.nan)
        age_value = AGE_MAP.get(str(age_raw).strip(), 2.0) / 4.0

        sex_raw = row.get("Sex", np.nan)
        sex_value = SEX_MAP.get(str(sex_raw).strip(), 0.5)

        height_raw = pd.to_numeric(row.get("Height", np.nan), errors="coerce")
        height_value = 0.5 if pd.isna(height_raw) else float(np.clip(height_raw / 200.0, 0.0, 1.5))

        weight_raw = pd.to_numeric(row.get("Weight", np.nan), errors="coerce")
        weight_value = 0.5 if pd.isna(weight_raw) else float(np.clip(weight_raw / 120.0, 0.0, 1.5))

        return np.asarray([age_value, sex_value, height_value, weight_value], dtype=np.float32)

    def __len__(self):
        return len(self.patients)

    def _load_audio(self, wav_path):
        try:
            y, _ = librosa.load(wav_path, sr=self.sr)
        except Exception as exc:
            print(f"Warning: failed to load {wav_path}: {exc}")
            return None
        return y

    def _compute_segment_starts(self, wav_len):
        if wav_len <= self.segment_length:
            return [0]
        starts = list(range(0, wav_len - self.segment_length + 1, self.segment_hop_length))
        last_start = wav_len - self.segment_length
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _peak_normalize(self, segment):
        if len(segment) < self.segment_length:
            segment = np.pad(segment, (0, self.segment_length - len(segment)))
        segment = segment.astype(np.float32)
        segment = segment - float(segment.mean())
        peak = float(np.max(np.abs(segment)))
        if peak > 1e-6:
            segment = segment / peak
        return np.clip(segment, -1.0, 1.0).astype(np.float32)

    def _normalize_for_mel(self, segment):
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

    def _band_energy_ratio(self, power, freqs, low, high):
        total = float(power.sum())
        if total <= 1e-10:
            return 0.0
        mask = (freqs >= low) & (freqs < high)
        return float(power[mask].sum() / total)

    def _acoustic_features(self, segment):
        rms = float(np.sqrt(np.mean(segment ** 2)))
        abs_mean = float(np.mean(np.abs(segment)))
        peak = float(np.max(np.abs(segment)))
        crest = float(peak / max(rms, 1e-6))
        zcr = float(librosa.feature.zero_crossing_rate(y=segment, frame_length=512, hop_length=128).mean())

        stft = np.abs(librosa.stft(segment, n_fft=512, hop_length=128))
        power = stft ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=512)
        centroid = float(librosa.feature.spectral_centroid(S=stft, sr=self.sr).mean() / (self.sr / 2.0))
        bandwidth = float(librosa.feature.spectral_bandwidth(S=stft, sr=self.sr).mean() / (self.sr / 2.0))
        rolloff = float(librosa.feature.spectral_rolloff(S=stft, sr=self.sr, roll_percent=0.85).mean() / (self.sr / 2.0))
        flatness = float(librosa.feature.spectral_flatness(S=stft).mean())

        mean_power = power.mean(axis=1)
        prob = mean_power / max(float(mean_power.sum()), 1e-10)
        entropy = float(-(prob * np.log(prob + 1e-10)).sum() / np.log(len(prob)))
        low_ratio = self._band_energy_ratio(mean_power, freqs, 20.0, 150.0)
        mid_ratio = self._band_energy_ratio(mean_power, freqs, 150.0, 400.0)
        high_ratio = self._band_energy_ratio(mean_power, freqs, 400.0, 1000.0)

        features = np.asarray(
            [
                rms,
                abs_mean,
                peak,
                crest,
                zcr,
                centroid,
                bandwidth,
                rolloff,
                flatness,
                entropy,
                low_ratio,
                mid_ratio - high_ratio,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_segments(self, segment_records):
        if len(segment_records) <= self.max_segments_per_patient:
            return segment_records
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

    def _apply_position_dropout(self, segment_records):
        if not self.training or self.position_dropout_prob <= 0.0:
            return segment_records
        if np.random.random() >= self.position_dropout_prob:
            return segment_records

        positions = sorted({record["position"] for record in segment_records})
        min_keep = max(1, int(self.min_positions_after_dropout))
        if len(positions) <= min_keep:
            return segment_records

        keep_count = np.random.randint(min_keep, len(positions))
        keep_positions = set(np.random.choice(positions, size=keep_count, replace=False).tolist())
        dropped = [record for record in segment_records if record["position"] in keep_positions]
        return dropped if dropped else segment_records

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

        segment_records = self._apply_position_dropout(segment_records)
        segment_records = self._select_segments(segment_records)
        x_scale1, x_scale2, x_scale3 = [], [], []
        acoustic, position_indices = [], []
        for record in segment_records:
            y = record["audio"]
            start = record["start"]
            peak_norm = self._peak_normalize(y[start : start + self.segment_length])
            mel_segment = self._normalize_for_mel(peak_norm)
            x_scale1.append(torch.tensor(self._wav_to_logmel(mel_segment, 256, 64, self.n_mels)).unsqueeze(0))
            x_scale2.append(torch.tensor(self._wav_to_logmel(mel_segment, 192, 48, self.n_mels)).unsqueeze(0))
            x_scale3.append(torch.tensor(self._wav_to_logmel(mel_segment, 128, 32, self.n_mels)).unsqueeze(0))
            acoustic.append(torch.tensor(self._acoustic_features(peak_norm), dtype=torch.float32))
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
            "acoustic": torch.stack(acoustic, dim=0).float(),
            "position_index": torch.tensor(position_indices, dtype=torch.long),
            "segment_mask": torch.ones(len(position_indices), dtype=torch.float32),
            "weak_clinical": torch.tensor(patient["weak_clinical"], dtype=torch.float32),
            "y_outcome": torch.tensor(patient["y_outcome"], dtype=torch.long),
            "y_murmur": torch.tensor(patient["y_murmur"], dtype=torch.long),
            "raw": patient["raw"],
        }


def audio_primary_collate(batch):
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

    positions, masks = [], []
    for item in batch:
        position = item["position_index"]
        mask = item["segment_mask"]
        if position.size(0) < max_segments:
            pad_len = max_segments - position.size(0)
            position = torch.cat([position, position.new_zeros(pad_len)], dim=0)
            mask = torch.cat([mask, mask.new_zeros(pad_len)], dim=0)
        positions.append(position)
        masks.append(mask)

    return {
        "patient_id": [item["patient_id"] for item in batch],
        "available_positions": [item["available_positions"] for item in batch],
        "position_count": torch.tensor([item["position_count"] for item in batch], dtype=torch.long),
        "has_all_positions": torch.tensor([item["has_all_positions"] for item in batch], dtype=torch.bool),
        "x_scale1": pad_segments("x_scale1"),
        "x_scale2": pad_segments("x_scale2"),
        "x_scale3": pad_segments("x_scale3"),
        "acoustic": pad_segments("acoustic"),
        "position_index": torch.stack(positions, dim=0),
        "segment_mask": torch.stack(masks, dim=0),
        "weak_clinical": torch.stack([item["weak_clinical"] for item in batch], dim=0),
        "y_outcome": torch.stack([item["y_outcome"] for item in batch], dim=0),
        "y_murmur": torch.stack([item["y_murmur"] for item in batch], dim=0),
        "raw": [item["raw"] for item in batch],
    }
