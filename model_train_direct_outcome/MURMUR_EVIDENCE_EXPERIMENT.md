# Murmur Evidence Experiment

本实验对应外部分析建议中的两个最小可验证方向：

1. Segment auxiliary supervision
2. Patient-level evidence aggregation

目标不是更换 backbone，而是验证：

- 让 segment encoder 学习更贴近声音本身的 `Murmur Present/Absent`，是否能改善 Outcome 排序。
- 使用模型预测出的 segment murmur risk 构造患者级 evidence features，是否能降低 FP，同时不显著增加 FN。

## 实验 1：v6 Murmur Auxiliary

入口：

```bash
./run_audio_primary_v6_murmur_aux.sh
```

输出：

```text
results/audio_primary_v6_murmur_aux
checkpoints/audio_primary_v6_murmur_aux.pth
```

相对 v3 的主要变化：

- 新增 `segment_murmur_head`
- 对 `Murmur=Absent/Present` 患者做 segment-level weak supervision
- `Murmur=Unknown` 不参与 segment murmur loss
- 不开启 patient-level evidence aggregation

主要验证问题：

> 仅增加声音相关辅助任务，是否能改善 Outcome 的患者级排序能力，尤其是 `Murmur=Absent` 子集？

## 实验 2：v7 Murmur Evidence Aggregation

入口：

```bash
./run_audio_primary_v7_murmur_evidence.sh
```

输出：

```text
results/audio_primary_v7_murmur_evidence
checkpoints/audio_primary_v7_murmur_evidence.pth
```

相对 v6 的主要变化：

- 开启 `--use-murmur-evidence`
- 从 segment murmur probability 生成患者级显式 evidence features
- evidence features 拼接到患者级 outcome head

当前 evidence features 包含：

- segment murmur probability mean
- max
- top-3 mean
- top-5 mean
- std
- soft high-risk ratio
- 每个位置 AV/MV/PV/TV 的 mean / max / high-risk ratio

主要验证问题：

> 显式患者级 evidence structure 是否比 attention/max pooling 更能降低 FP？

## 推荐执行顺序

先跑 v6，再跑 v7：

```bash
git pull origin main
cd /home/zdx/python_daima/01heart/model_train_direct_outcome
chmod +x run_audio_primary_v6_murmur_aux.sh run_audio_primary_v7_murmur_evidence.sh
mkdir -p logs

./run_audio_primary_v6_murmur_aux.sh 2>&1 | tee logs/audio_primary_v6_murmur_aux.log
./run_audio_primary_v7_murmur_evidence.sh 2>&1 | tee logs/audio_primary_v7_murmur_evidence.log
```

训练完成后提交结果：

```bash
git add results/audio_primary_v6_murmur_aux results/audio_primary_v7_murmur_evidence
git commit -m "Add murmur evidence experiment results"
git push origin main
```

如果一次只跑一个实验，也可以只提交对应结果目录。

## 重点比较指标

和 v3 对比：

```text
results/audio_primary_v3/diagnosis_summary.json
results/audio_primary_v3/subgroup_auc_summary.csv
results/audio_primary_v3/group_error_summary.csv
```

重点看：

- full ROC-AUC / PR-AUC
- selected threshold 下的 FN / FP
- `Murmur=Absent` 子集 ROC-AUC
- `Murmur=Absent` 子集的 FN/FP tradeoff
- complete-position 子集表现

成功信号：

- `Murmur=Absent` AUC 明显高于 v3 的约 0.64
- 在 FN 不明显上升的情况下 FP 下降
- 或者在 FP 不上升的情况下 FN 下降

失败信号：

- full AUC 接近或低于 v3
- `Murmur=Absent` AUC 仍约 0.64
- FN/FP tradeoff 与 v3 基本一致

如果 v6 有收益但 v7 没收益，说明 segment Murmur 辅助有价值，但当前 evidence features 设计不够好。

如果 v6、v7 都无收益，说明当前标签和音频特征的可分性瓶颈更可能是主因。
