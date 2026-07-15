# Linux GPU 服务器训练指导

本文档用于把本地打包好的 `direct_outcome_gpu_bundle.tar.gz` 上传到 Linux GPU 服务器后，完成解压、环境配置和启动训练。

## 1. 服务器要求

建议环境：

- Linux x86_64
- NVIDIA GPU
- NVIDIA Driver 支持 CUDA 12.x
- Python 3.10 或 3.11 优先，Python 3.12 通常也可用
- 磁盘空间建议至少 5GB

本项目默认安装 PyTorch CUDA 12.1 wheel：

```bash
https://download.pytorch.org/whl/cu121
```

如果服务器 CUDA/驱动更适合其他 PyTorch wheel，可以通过 `TORCH_INDEX_URL` 覆盖。

## 2. 上传压缩包

本地生成的压缩包：

```text
direct_outcome_gpu_bundle.tar.gz
```

上传示例：

```bash
scp direct_outcome_gpu_bundle.tar.gz user@server:/data/heart_sound/
```

登录服务器：

```bash
ssh user@server
cd /data/heart_sound
```

## 3. 解压

```bash
tar -xzf direct_outcome_gpu_bundle.tar.gz
cd model_train_direct_outcome
```

解压后应能看到：

```text
src/
data/circor/training_data.csv
data/circor/training_data/*.wav
setup_gpu_env.sh
run_gpu_training.sh
requirements-linux.txt
```

检查 wav 数量：

```bash
find data/circor/training_data -name "*.wav" | wc -l
```

预期约：

```text
3163
```

## 4. 一键配置环境

```bash
chmod +x setup_gpu_env.sh run_gpu_training.sh
./setup_gpu_env.sh
```

脚本会执行：

1. 创建 `.venv`
2. 安装 GPU 版 PyTorch
3. 安装 `librosa / pandas / scikit-learn / numpy` 等依赖
4. 检查 `torch.cuda.is_available()`

如果需要指定 Python：

```bash
PYTHON_BIN=python3.10 ./setup_gpu_env.sh
```

如果需要改 PyTorch CUDA wheel，例如 CUDA 11.8：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 ./setup_gpu_env.sh
```

## 5. 启动 GPU 训练

默认训练：

```bash
./run_gpu_training.sh
```

默认配置：

```text
device = cuda
epochs = 35
patience = 8
batch_size = 8
num_workers = 4
max_segments_per_patient = 32
target_recall = 0.95
outcome_abnormal_weight = 3.0
fn_penalty_weight = 0.50
checkpoint = checkpoints/direct_outcome_gpu_v1.pth
```

自定义示例：

```bash
EPOCHS=50 \
BATCH_SIZE=8 \
NUM_WORKERS=8 \
MAX_SEGMENTS=32 \
TARGET_RECALL=0.95 \
ABNORMAL_WEIGHT=2.5 \
FN_PENALTY=0.35 \
CHECKPOINT_PATH=checkpoints/direct_outcome_gpu_recall95_v2.pth \
./run_gpu_training.sh
```

如果 GPU 显存不足，优先降低：

```bash
BATCH_SIZE=4 MAX_SEGMENTS=16 ./run_gpu_training.sh
```

## 6. 训练输出说明

训练日志每轮会输出两组验证结果：

1. 全量验证集
2. 四位置齐全验证子集

示例：

```text
Epoch 3/35 | loss=... | threshold=... | acc=... | recall=... | specificity=... | FN=... | FP=...
  Complete-position val with full-val threshold | acc=... | recall=... | specificity=... | FN=... | FP=...
```

核心优先级：

```text
1. Abnormal recall
2. FN 数量
3. specificity
4. accuracy
```

不要只看 accuracy。当前业务目标是“宁可误报，不能漏判”，因此先看 recall 和 FN。

训练完成后会保存：

```text
checkpoints/*.pth
checkpoints/*.json
```

其中 `.json` 会记录最佳验证指标和阈值。

## 7. 先跑小训练验证链路

如果只是确认服务器链路：

```bash
EPOCHS=2 \
PATIENCE=2 \
BATCH_SIZE=4 \
MAX_SEGMENTS=8 \
CHECKPOINT_PATH=checkpoints/server_smoke_train.pth \
./run_gpu_training.sh
```

## 8. 常见问题

### CUDA 不可用

如果脚本显示：

```text
cuda_available: False
```

先检查：

```bash
nvidia-smi
```

再检查 PyTorch wheel 是否匹配服务器驱动。可以重建环境并指定其他 wheel：

```bash
rm -rf .venv
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 ./setup_gpu_env.sh
```

### 显存不足

降低 batch 和片段数：

```bash
BATCH_SIZE=4 MAX_SEGMENTS=16 ./run_gpu_training.sh
```

### 误报太多

在 recall 仍然满足要求的前提下，降低异常权重或 FN 惩罚：

```bash
ABNORMAL_WEIGHT=2.0 FN_PENALTY=0.25 ./run_gpu_training.sh
```

### 漏判太多

提高异常权重或 FN 惩罚：

```bash
ABNORMAL_WEIGHT=3.5 FN_PENALTY=0.75 ./run_gpu_training.sh
```

