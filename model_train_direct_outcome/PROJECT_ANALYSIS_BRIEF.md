# Heart Sound Outcome Project Analysis Brief

这份文档用于交给外部 ChatGPT 或研究助手分析，目标是探究当前心音二分类任务更合理的建模方案，再把可落地方案交回代码仓实现。

## 1. 项目目标

任务：通过患者心音数据判断心脏是否异常。

业务期望：

- 输出二分类：`Normal` / `Abnormal`
- 优先降低 `FN`：异常患者不能漏判
- 同时希望降低 `FP`：正常患者不要大量误判为异常
- 当前期望目标：
  - `FN < 10`
  - `FP < 10`
  - 如果引入三分类/分诊，则 `coverage >= 80%`

其中：

- `FN`：真实 Abnormal，被模型判为 Normal
- `FP`：真实 Normal，被模型判为 Abnormal
- `coverage`：如果允许 `Uncertain`，则直接给出 Normal/Abnormal 的样本比例

## 2. 当前数据集概况

数据集：CirCor DigiScope 儿科心音数据集。

患者级标签：

- `Outcome`
  - `Normal`
  - `Abnormal`
- `Murmur`
  - `Absent`
  - `Present`
  - `Unknown`

已知分布：

- 总患者数约 942
- `Outcome=Normal`：约 486
- `Outcome=Abnormal`：约 456
- `Murmur=Absent`：约 695
- `Murmur=Present`：约 179
- `Murmur=Unknown`：约 68

四个标准采集位置：

- `AV`
- `MV`
- `PV`
- `TV`

录音位置并不总是完整。之前统计中，四位置齐全患者约 588，其余患者只有 1-3 个位置或少量 5-6 个位置。

## 3. 当前仓库状态

仓库主线已经清理完成。

保留的有效入口：

- `run_audio_primary.sh`
  - audio-primary v1，对照版本
- `run_audio_primary_v3.sh`
  - audio-primary v3，有效对照版本
- `run_audio_primary_v4.sh`
  - audio-primary v4，当前推荐复现实验入口
  - 使用 8 秒片段，3 秒 hop
- `run_audio_only_baseline.sh`
  - 纯音频 baseline
- `run_clinical_baseline.sh`
  - 纯临床 baseline

保留的关键源码：

- `src/dataset_audio_primary.py`
- `src/train_audio_primary.py`
- `src/train_audio_only_baseline.py`
- `src/train_clinical_baseline.py`
- `src/baseline_utils.py`
- `src/dataset_direct_outcome.py`
- `src/dataset_hms.py`
- `src/model_hms.py`

已删除：

- 无效 v2
- 失败 v5 triage / fnfp 方案
- 早期 direct-outcome residual / position-aware 入口
- 旧 HMS 两阶段训练/推理入口

清理记录见：

- `ITERATION_CLEANUP.md`

## 4. 当前主线模型结构

当前 audio-primary 模型以患者为单位输入，核心逻辑如下：

1. 每个患者可包含多个位置的 wav。
2. 每个 wav 被切成多个片段。
3. 每个片段转成多尺度 log-mel 特征。
4. 每个片段额外带有：
   - 采集位置 embedding
   - handcrafted acoustic descriptors
5. 模型使用 `ScaleEncoder` 编码多尺度音频。
6. 片段级特征经过 attention pooling 和 max pooling 聚合为患者级表示。
7. 患者级表示融合弱临床信息：
   - age
   - sex
   - height
   - weight
8. 输出患者级 `Outcome` 二分类概率。

模型有意识地不直接输入：

- `Murmur`
- `Pregnancy status`
- `position_count`
- `has_all_positions`
- 缺失标记

原因：这些字段容易形成数据集捷径，不一定代表真实可泛化的心音异常特征。

## 5. 历史迭代结论

### v1

audio-primary 初版。

大致表现：

- ROC-AUC 约 0.71
- PR-AUC 约 0.73
- 在高 recall 阈值下，FN 可压低，但 FP 很高
- `Murmur=Absent` 子集 AUC 约 0.64

### v2

引入更强正则化和 hard negative 后，整体排序能力下降。

结论：无效，已删除。

### v3

相对有效的对照版本。

曾观察到较好的业务点：

```text
Epoch 23/70 | loss=0.6244 | auc=0.7080 | pr_auc=0.7029 | threshold=0.25 | acc=0.6138 | recall=0.9231 | specificity=0.3265 | FN=7 | FP=66
```

问题：FN 较低，但 FP 仍约 60 多。

### v4

片段改为 8 秒，hop 3 秒。

曾观察到较好的业务点：

```text
Epoch 26/70 | loss=0.6450 | auc=0.6952 | pr_auc=0.6973 | threshold=0.16 | acc=0.6085 | recall=0.8901 | specificity=0.3469 | FN=10 | FP=64
```

相对 v3，AUC 略低，但在特定阈值点 FP 有小幅降低。

问题仍然是 FP 很高。

### v5 triage

尝试 Normal / Abnormal / Uncertain 三分类分诊逻辑。

失败现象：

```text
coverage=0.0000
uncertain=189
```

也就是模型把全部样本放进 `Uncertain`，却仍可能被错误选择为 best。

结论：该方案目标函数和 checkpoint 选择逻辑存在严重漏洞，已删除。

## 6. 当前最核心瓶颈

最大问题不是简单阈值没调好，而是：

`Murmur=Absent + Outcome=Abnormal`

和

`Murmur=Absent + Outcome=Normal`

在当前音频特征空间里高度重叠。

之前对 `Murmur=Absent` 子集做过患者级概率扫描：

- 子集验证样本约 145
- Abnormal 约 54
- Normal 约 91
- v1/v3/v4 在该子集 ROC-AUC 均约 0.63-0.64

典型现象：

- 如果要求 `FN <= 7`，FP 仍约 58-69
- 如果要求 `FP <= 10`，FN 会升到 38-41 左右

这说明当前模型输出的患者级概率排序能力不足。Normal 与 Abnormal 分布重叠严重，单靠阈值无法同时得到低 FN 和低 FP。

## 7. 需要重点分析的问题

请重点分析以下问题，并给出可落地方案。

### 问题 1：目标是否可达

在当前 CirCor 数据集和当前标签条件下，是否现实地可能达到：

- `FN < 10`
- `FP < 10`
- `coverage >= 80%`

如果不可达，请说明理论原因和数据证据应该如何验证。

### 问题 2：Murmur Absent 子集如何处理

当前最大难点是：

- `Murmur Absent + Abnormal`
- `Murmur Absent + Normal`

请分析：

- 二者是否可能仅靠心音区分？
- 是否应该单独训练/单独建模该子集？
- 是否应该把 `Murmur` 作为分层诊断变量，而不是简单辅助任务？
- 如果不能使用 Murmur 作为输入，如何避免模型退化？

### 问题 3：患者级聚合是否有希望

当前输入是片段级心音，再聚合到患者级。

请分析是否应该设计显式患者级特征，例如：

- 每个位置的异常风险
- top-k 片段风险
- 高风险片段数量
- 高风险片段覆盖的位置数量
- attention entropy
- 片段风险方差
- 各位置风险差异
- 信号质量指标

并判断这些特征是否可能帮助区分 Absent-Abnormal 与 Absent-Normal。

### 问题 4：模型结构是否需要改变

当前结构是 CNN-like `ScaleEncoder` + position embedding + attention/max pooling。

请分析是否应考虑：

- MIL 多实例学习
- gated attention MIL
- per-position encoder + patient-level fusion
- cycle-level segmentation 后再建模
- contrastive learning
- supervised pretraining on Murmur/Timing/Grade，再 fine-tune Outcome
- abnormal evidence detector + patient aggregator
- calibration-aware model

需要指出每种方案的优缺点、实现复杂度和最推荐的路线。

### 问题 5：损失函数如何设计

当前只靠 BCE/CE、class weight、hard negative、soft FN penalty 的调参已经进展有限。

请分析是否应该使用：

- focal loss
- asymmetric loss
- AUC surrogate loss
- pairwise ranking loss
- hard negative mining
- subgroup-aware loss
- cost-sensitive loss
- constrained optimization for FN/FP/coverage

特别注意：损失函数决定参数更新方向，但最终业务指标是患者级 FN/FP，因此要说明训练目标和验证选择指标如何对齐。

### 问题 6：三分类/分诊是否可行

如果二分类不可靠，是否应该输出：

- Normal
- Abnormal
- Uncertain / Recollect / Manual review

但要求：

- coverage >= 80%
- 不能出现 all-Uncertain 退化

请给出正确的目标函数、验证指标和 checkpoint 选择方式。

### 问题 7：下一步最小可验证实验

请不要直接给大而空的方案。请给出 1-3 个最小可验证实验，每个实验说明：

- 修改哪些代码模块
- 输入输出是什么
- 预期验证什么假设
- 成功/失败判据是什么
- 预计对 FN/FP/coverage 有什么影响

## 8. 当前倾向的下一步

目前比较合理的下一步不是继续盲目调阈值，而是先做患者级可分性诊断：

1. 从现有模型导出患者级聚合特征。
2. 特别针对 `Murmur=Absent` 子集训练一个透明的患者级二层模型。
3. 比较它是否能显著超过当前神经网络的患者级概率。
4. 如果仍无法分开，则说明当前数据和标签难以支撑目标，需要改变任务定义或引入人工复查/重采集机制。

希望外部分析重点回答：

**在当前数据条件下，什么方案最有可能真正降低 FP，同时不显著增加 FN？**
