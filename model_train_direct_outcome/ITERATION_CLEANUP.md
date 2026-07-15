# Iteration Cleanup Record

本次清理目标：让仓库只保留当前可复盘、可继续迭代的主线文件，删除已验证无效或容易误导后续训练的旧入口。

## 保留的有效迭代

| 类别 | 文件/目录 | 保留原因 |
| --- | --- | --- |
| audio-primary v1 | `run_audio_primary.sh`, `results/audio_primary_v1` | 原始 audio-primary 主线，作为后续版本对照。 |
| audio-primary v3 | `run_audio_primary_v3.sh`, `results/audio_primary_v3` | 引入 position dropout 和 hard negative 后的有效对照。 |
| audio-primary v4 | `run_audio_primary_v4.sh`, `results/audio_primary_v4_8s_hop3` | 8 秒片段、3 秒 hop，当前最适合作为下一轮复现入口。 |
| audio-only baseline | `run_audio_only_baseline.sh`, `results/audio_only_baseline_v1` | 判断纯音频是否具有区分能力。 |
| clinical-only baseline | `run_clinical_baseline.sh`, `results/clinical_only_baseline_v1` | 检查临床字段捷径风险。 |

## 删除的无效或废弃迭代

| 类别 | 删除内容 | 删除原因 |
| --- | --- | --- |
| audio-primary v2 | `run_audio_primary_v2.sh`, `results/audio_primary_v2` | 过度正则化后效果下降，不作为后续主线。 |
| audio-primary v5 fn/fp | `results/audio_primary_v5_8s_hop3_fnfp` | v5 方向已废弃，避免误当成有效结果。 |
| audio-primary v5 triage | `run_audio_primary_v5.sh`, `results/audio_primary_v5_triage_8s_hop3` | 出现 all-Uncertain 退化，`coverage=0` 仍可能被错误选为 best。 |
| early direct-outcome | `run_gpu_training.sh`, `run_gpu_diagnosis.sh`, `src/train_direct_outcome.py`, `src/model_direct_outcome.py`, `src/diagnose_direct_outcome.py` 等 | 早期 residual/position-aware 方案不是当前主线，继续保留会干扰后续实现。 |
| old HMS pipeline | HMS 训练、特征生成、outcome 二阶段推理入口 | 与当前 direct outcome audio-primary 主线不一致。仅保留 `dataset_hms.py` 和 `model_hms.py`，因为当前代码仍复用位置定义和 `ScaleEncoder`。 |

## 训练脚本改动

`src/train_audio_primary.py` 中删除了失败 triage 方案相关逻辑：

- 删除 `triage_fn_fp` checkpoint selection mode。
- 删除 triage 阈值搜索。
- 删除无条件写 triage 输出。
- 保留二分类 checkpoint selection mode：
  - `auc`
  - `recall_specificity`
  - `fn_fp`

这样后续训练不会再出现 `coverage=0` 却被保存为 best 的问题。

## 当前推荐入口

```bash
./run_audio_primary_v4.sh
```

后续如果要继续做患者级可分性诊断，应基于 v4 的输出表扩展，而不是恢复 v5 triage。
