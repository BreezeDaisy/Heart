# Heart Sound Direct Outcome Experiments

本仓库保留当前任务主线：用患者级心音输入直接预测 `Outcome` 二分类结果。

核心业务目标：

- 尽量降低 Abnormal 漏判，即降低 FN。
- 在 FN 可控的前提下降低 Normal 误判，即降低 FP。
- 当前证据显示，`Murmur=Absent` 子集中 Abnormal 与 Normal 高度重叠，是主要瓶颈。

## 当前保留内容

### 有效或有分析价值的训练入口

- `run_audio_primary.sh`
  - audio-primary v1。
  - 原始 audio-primary 主线，用于和后续版本对比。

- `run_audio_primary_v3.sh`
  - audio-primary v3。
  - 加入 position dropout 和 hard negative 约束后的有效对照版本。

- `run_audio_primary_v4.sh`
  - audio-primary v4。
  - 每段 8 秒、hop 3 秒。
  - 当前更适合作为下一轮复现实验和分析的主入口。

- `run_audio_only_baseline.sh`
  - 只使用音频的 baseline。
  - 用于判断临床信息是否形成捷径。

- `run_clinical_baseline.sh`
  - 只使用临床信息的 baseline。
  - 用于检查 age/sex/height/weight 等临床字段是否过强影响结果。

### 结果目录

保留的 `results/` 子目录：

- `audio_only_baseline_v1`
- `clinical_only_baseline_v1`
- `audio_primary_v1`
- `audio_primary_v3`
- `audio_primary_v4_8s_hop3`

这些目录用于复盘不同输入和不同模型配置下的性能差异。

## 已移除内容

已删除明确无效或容易误导后续工作的版本：

- audio-primary v2
  - 过度正则化后效果下降。

- audio-primary v5 triage / fnfp
  - 之前出现 all-Uncertain 退化，`coverage=0` 仍可能被错误保存为 best。
  - 已删除 v5 启动脚本和结果目录。
  - 已从 `src/train_audio_primary.py` 移除 triage checkpoint 选择逻辑。

- 早期 direct-outcome residual / position-aware 训练入口
  - 已不是当前主线。
  - 保留其结论在分析文档中讨论，不再保留可启动入口。

- 旧 HMS 两阶段训练/推理入口
  - 与当前 direct outcome 主线不一致。
  - 仅保留 `dataset_hms.py` 和 `model_hms.py`，因为 audio-primary 仍复用位置定义和 `ScaleEncoder`。

## 推荐训练命令

服务器环境配置：

```bash
chmod +x setup_gpu_env.sh run_audio_primary_v4.sh
./setup_gpu_env.sh
```

启动当前推荐版本：

```bash
./run_audio_primary_v4.sh
```

快速 smoke test：

```bash
EPOCHS=2 PATIENCE=2 BATCH_SIZE=4 MAX_SEGMENTS=8 ./run_audio_primary_v4.sh
```

保留完整日志：

```bash
./run_audio_primary_v4.sh 2>&1 | tee logs/audio_primary_v4_$(date +%Y%m%d_%H%M%S).log
```

## 输出说明

训练完成后主要看：

- `results/<version>/diagnosis_summary.json`
- `results/<version>/patient_predictions.csv`
- `results/<version>/group_error_summary.csv`
- `results/<version>/subgroup_auc_summary.csv`
- `results/<version>/threshold_metrics.csv`
- `results/<version>/epoch_history.csv`

其中 `epoch_history.csv` 用于复盘每个 epoch 的 loss、AUC、PR-AUC、threshold、FN、FP 等指标，避免只看最后一个 epoch。

## 当前判断

根据既有实验，单纯阈值调整无法把 FP 从 60 多降到十几，同时保持 FN 很低。后续更合理的方向是先做患者级可分性诊断：

- 片段分数聚合特征。
- 四位置风险分布。
- 高风险片段数量和位置数量。
- `Murmur=Absent` 子集单独分析。

如果这些患者级特征仍不能区分 Absent-Abnormal 与 Absent-Normal，则说明当前数据和弱监督标签本身不足以支撑 FN、FP 都低于 10 的目标。
