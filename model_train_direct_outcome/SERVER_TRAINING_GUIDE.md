# Linux GPU Training Guide

本文档用于在 Linux GPU 服务器上复现当前保留的 audio-primary 训练版本。

## 1. 环境要求

- Linux x86_64
- NVIDIA GPU
- NVIDIA Driver 支持 CUDA 12.x
- Python 3.10 或 3.11
- 建议磁盘空间不少于 8GB

## 2. 获取代码

```bash
git clone https://github.com/BreezeDaisy/Heart.git
cd Heart/model_train_direct_outcome
```

如果服务器已配置 GitHub SSH key，也可以使用：

```bash
git clone git@github.com:BreezeDaisy/Heart.git
cd Heart/model_train_direct_outcome
```

## 3. 配置环境

```bash
chmod +x setup_gpu_env.sh run_audio_primary.sh run_audio_primary_v3.sh run_audio_primary_v4.sh
./setup_gpu_env.sh
```

如果需要指定 Python：

```bash
PYTHON_BIN=python3.10 ./setup_gpu_env.sh
```

如果服务器更适合 CUDA 11.8 wheel：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 ./setup_gpu_env.sh
```

## 4. 推荐训练入口

当前推荐先复现 v4：

```bash
./run_audio_primary_v4.sh
```

v4 默认配置：

```text
segment_duration = 8.0
segment_hop = 3.0
max_segments_per_patient = 24
epochs = 70
patience = 14
batch_size = 6
selection_mode = auc
```

快速检查链路：

```bash
EPOCHS=2 PATIENCE=2 BATCH_SIZE=4 MAX_SEGMENTS=8 ./run_audio_primary_v4.sh
```

保留完整训练日志：

```bash
mkdir -p logs
./run_audio_primary_v4.sh 2>&1 | tee logs/audio_primary_v4_$(date +%Y%m%d_%H%M%S).log
```

## 5. 其它保留版本

复现 v1：

```bash
./run_audio_primary.sh
```

复现 v3：

```bash
./run_audio_primary_v3.sh
```

音频 baseline：

```bash
./run_audio_only_baseline.sh
```

临床 baseline：

```bash
./run_clinical_baseline.sh
```

## 6. 训练结果位置

每个版本会写入各自的 `results/<version>/` 目录，重点查看：

```text
diagnosis_summary.json
patient_predictions.csv
group_error_summary.csv
subgroup_auc_summary.csv
threshold_metrics.csv
epoch_history.csv
```

`epoch_history.csv` 是当前必须保留的复盘文件，用于比较不同 epoch 的：

- loss
- ROC-AUC
- PR-AUC
- selected threshold
- accuracy
- recall
- specificity
- FN
- FP

## 7. 当前不再使用的入口

以下版本已经从仓库移除：

- `run_audio_primary_v2.sh`
- `run_audio_primary_v5.sh`
- `run_gpu_training.sh`
- `run_gpu_diagnosis.sh`

不要再基于 v5 triage 结果判断模型好坏。该方案曾出现 all-Uncertain 退化，无法满足 coverage 要求。
