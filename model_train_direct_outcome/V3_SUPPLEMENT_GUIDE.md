# V3 Supplement Guide

本文档对应外部分析中“第八点：需要补充的信息”。

当前以 `audio_primary_v3` 作为目前最佳模型版本。

## 1. 患者级预测结果

已有文件：

```text
results/audio_primary_v3/patient_predictions.csv
```

该文件每位患者一行，已经包含：

- `patient_id`
- `prob_abnormal`
- `y_outcome`
- `outcome`
- `y_murmur`
- `murmur`
- `position_count`
- `has_all_positions`
- `available_positions`
- `segment_count`
- `Age`
- `Sex`
- `Height`
- `Weight`
- `Pregnancy status`
- `has_AV/MV/PV/TV`
- `segments_AV/MV/PV/TV`
- `attention_AV/MV/PV/TV`
- 不同阈值下的 `pred_*` 和 `error_*`

对应外部分析需要的第 1 项信息。

## 2. Segment 级输出

新增脚本：

```text
src/export_audio_primary_diagnostics.py
run_export_audio_primary_v3_diagnostics.sh
```

它会导出：

```text
results/audio_primary_v3_supplement/patient_level_diagnostics.csv
results/audio_primary_v3_supplement/segment_level_diagnostics.csv
results/audio_primary_v3_supplement/position_segment_summary.csv
results/audio_primary_v3_supplement/subgroup_auc_summary.csv
results/audio_primary_v3_supplement/murmur_absent_threshold_tradeoff.csv
results/audio_primary_v3_supplement/murmur_absent_constraint_summary.csv
results/audio_primary_v3_supplement/diagnostic_export_summary.json
```

其中 `segment_level_diagnostics.csv` 每个 segment 一行，包含：

- `patient_id`
- `segment_index`
- `position`
- `start_sec`
- `end_sec`
- `segment_prob_abnormal`
- `segment_score_source`
- `attention_weight`
- `is_evidence_segment`
- `patient_prob_abnormal`
- `outcome`
- `murmur`
- acoustic descriptors

对应外部分析需要的第 2 项信息。

注意：v3 checkpoint 训练时还没有 `segment_outcome_head`，因此 segment 风险不是来自未训练的 segment head。导出脚本会使用 `single_segment_patient_forward`：

```text
只保留当前单个 segment + 原患者临床信息
送入已训练的患者级 outcome head
得到该 segment 的异常风险代理分数
```

这比使用随机初始化的 segment head 更可靠，也更贴近 v3 模型真实学到的患者级判别能力。

## 3. 每位患者各位置录音数量和时长统计

导出脚本会生成：

```text
results/audio_primary_v3_supplement/position_segment_summary.csv
```

该文件每位患者每个位置一行，包含：

- `patient_id`
- `position`
- `segment_count`
- `first_start_sec`
- `last_end_sec`
- `mean_segment_prob`
- `max_segment_prob`
- `attention_sum`
- `evidence_segment_count`

其中 `last_end_sec` 可近似理解为该位置被模型实际采样覆盖到的时长上界。

对应外部分析需要的第 3 项信息。

## 4. 服务器运行方式

先确认 v3 权重存在：

```bash
ls checkpoints/audio_primary_v3.pth
```

如果不存在，把 `audio_primary_v3.pth` 放到：

```text
model_train_direct_outcome/checkpoints/audio_primary_v3.pth
```

然后运行：

```bash
chmod +x run_export_audio_primary_v3_diagnostics.sh
./run_export_audio_primary_v3_diagnostics.sh
```

如需 CPU：

```bash
DEVICE=cpu NUM_WORKERS=0 ./run_export_audio_primary_v3_diagnostics.sh
```

## 5. 交给外部 ChatGPT 的文件清单

建议提供以下文件：

```text
PROJECT_ANALYSIS_BRIEF.md
results/audio_primary_v3/diagnosis_summary.json
results/audio_primary_v3/patient_predictions.csv
results/audio_primary_v3/subgroup_auc_summary.csv
results/audio_primary_v3/group_error_summary.csv
results/audio_primary_v3_supplement/patient_level_diagnostics.csv
results/audio_primary_v3_supplement/segment_level_diagnostics.csv
results/audio_primary_v3_supplement/position_segment_summary.csv
results/audio_primary_v3_supplement/murmur_absent_constraint_summary.csv
```

其中后四个文件需要运行本 guide 中的导出脚本后生成。
