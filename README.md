# R²Mem 消融实验：仅代码仓库

本仓库只包含三组消融的训练、评测、输入准备代码和10份配置。公共参考只训练一次，因此是7个消融变体加1个完整参考。每个随机种子共4次 memory 预训练和8次 action 训练。

不包含数据、缓存索引、teacher 权重、模型 checkpoint、历史运行、日志、图片、报告、虚拟环境或认证文件。这些运行产物都由 `.gitignore` 排除。代码会在接收服务器的本地磁盘生成数据派生物和训练结果。

## 上传这个目录

先在 GitHub、GitLab 或其他 Git 服务创建一个空仓库，取得它的 SSH/HTTPS 地址。在本目录执行：

```bash
# 如果尚未初始化 Git，先执行 git init -b main。
git add .
git diff --cached --stat
git commit -m "Add reproducible R2Mem ablation code"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

将示例仓库地址换成实际地址。上传对象应是这个精简仓库，不要在原来的大型实验目录执行 `git add .`。如果 Git 没有作者身份，先设置你自己的 `user.name` 和 `user.email`。

## 其他服务器：环境和外部输入

```bash
git clone git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

需要另行准备以下依赖，Git clone 本仓库不会自动下载它们：

- RoboMemArena 源码：`https://github.com/OpenHelix-Team/RoboMemArena.git`，commit `d09f63b671f362985e1e0d038d2014c2d2519488`。
- OpenPI 源码：`https://github.com/Physical-Intelligence/openpi.git`，commit `215abfb217dbac7d5f1273282331b9b1866c0479`，建议位于 RoboMemArena 的 `training/openpi/`。
- Python 3.11 的 OpenPI 训练环境和对应 Transformers PyTorch 补丁。按上述固定版本 OpenPI 的 README 安装；该版本提供 `GIT_LFS_SKIP_SMUDGE=1 uv sync` 和 Transformers 补丁步骤。
- RoboMemArena 模拟器环境，解释器位于 `RoboMemArena/.venv/bin/python`；需要 MuJoCo、robosuite、LIBERO、EGL 和 PyYAML。LIBERO 配置位于 `RoboMemArena/local_setup/libero_config/config.yaml`。该环境按固定版本 RoboMemArena 的文档安装，必须能运行它的26个任务。
- 官方 pi05_base 的 PyTorch checkpoint 目录，包含 `model.safetensors`；所有 action 训练都从这个底座开始。
- 数据集 `yinchenghust/robomemarena_lerobot`，revision `b6595d7af67749d3644056bab0e096194332c1ba`。可以在目标服务器独立下载，也可以通过 Git 之外的存储传输。

固定版本也记录在 `configs/upstream_sources.json`。训练环境曾验证的主要版本为 Torch 2.7.1、NumPy 1.26.4、JAX 0.5.3、Flax 0.10.2、Transformers 4.53.2；仿真环境曾验证 MuJoCo 3.2.7 和 robosuite 1.4.1。本仓库不复制上述环境。

先激活目标服务器的 OpenPI 训练环境，再安装本代码：

```bash
source /absolute/path/to/RoboMemArena/training/openpi/.venv/bin/activate
python -m pip install -e '.[test,prepare]'
```

## 配置目标服务器路径

只修改 `server.local.yaml`，不要为修改路径而调整实验超参数。该文件不会进入 Git。

```bash
python scripts/configure_server.py \
  --benchmark-root /absolute/path/to/RoboMemArena \
  --assets-root /absolute/path/to/ablation_assets
source scripts/activate.sh
```

默认资产布局如下；脚本只写路径配置，不创建或下载这些资产：

```text
ablation_assets/
  robomemarena_lerobot/                    # 原始 LeRobot 数据
  pi05_base_pytorch/model.safetensors      # 官方底座
  metadata/robomemarena_lerobot_manifest.json
  metadata/robomemarena_norm/norm_stats.json
  metadata/robomemarena_norm/provenance.json
```

如果现有布局不同，使用 `--dataset-root`、`--manifest`、`--norm-stats`、`--base-checkpoint` 和 `--openpi-root` 指定实际位置。只允许本地路径覆盖，学习率、batch、seed、损失权重和训练步数不能通过 `server.local.yaml` 改写。`QRM_SERVER_CONFIG` 可以指定另一份本地路径文件。

## 在本机准备派生输入

所有服务器使用相同数据 revision、split_seed=20261012 和 teacher_seed=20261016。如果已有本服务器生成且校验通过的输入，可以复用它们。

没有数据时，可在目标服务器显式下载；此命令不属于上传内容：

```bash
python scripts/download_robomemarena_lerobot.py \
  --output /absolute/path/to/ablation_assets/robomemarena_lerobot
```

在目标服务器生成 manifest 和训练集归一化，然后编码完整视觉缓存并生成固定历史特征：

```bash
python scripts/prepare_training_data.py \
  --config ablation_memory_capacity/configs/k8.yaml

# 消耗 GPU 并读取完整数据；下面的物理 GPU 0 仅为示例。
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_history_cache.py \
  --config ablation_memory_capacity/configs/k8.yaml \
  --out data/history --encode-missing --device cuda:0

# CPU 转换，生成固定 teacher 的训练集归一化产物。
python scripts/prepare_history_features.py \
  --config ablation_memory_capacity/configs/k8.yaml \
  --index data/history/index.json --out data/history_fixed

python scripts/check_server.py
python -m pytest
```

缓存索引与 manifest、文件内容和源代码指纹绑定。不要用字符串替换强行修改旧缓存索引；路径变化后在目标服务器重新准备。本仓库的 `pi05_bridge.py`、固定 teacher、历史损失、写入器和初始化实现保留原方法代码；去除了无关的旧 PPO 入口。

## 运行消融

`run.py` 默认只预览，加 `--execute` 才训练；GPU 编号必须选择空闲物理卡。默认 seed=11，memory 60k、action 30k、global action batch=64。

| 目录 | variant | memory 来源 |
| --- | --- | --- |
| `ablation_memory_capacity` | `k4`、`k16` | 各自训练60k |
| `ablation_memory_capacity` | `k8` | 公共参考 |
| `ablation_residual_memory` | `residual32` | 独立训练60k |
| `ablation_residual_memory` | `residual128` | 公共参考 |
| `ablation_residual_memory` | `no_residual` | 跳过，直接 action |
| `ablation_optimization` | `reference` | 公共参考 |
| `ablation_optimization` | `no_joint` | 复用公共参考 memory |
| `ablation_optimization` | `no_pretraining`、`no_pretraining_no_joint` | 跳过，直接 action |

例如在同一服务器完成公共参考与关闭联合重建的对照：

```bash
python ablation_optimization/run.py --list

python ablation_optimization/run.py \
  --variant reference --stage memory --gpus 0 --execute

# 上一步完成60k且通过留出集验收后，再分别执行两个独立 action。
python ablation_optimization/run.py \
  --variant reference --stage action --gpus 0,1 --execute
python ablation_optimization/run.py \
  --variant no_joint --stage action --gpus 0,1 --execute
```

其他需要预训练的变体分别执行其 `--stage memory` 和 `--stage action`。跳过预训练的变体直接执行 `--stage action`。默认 action 固定读取对应 `memory/step_060000.pt`，并校验内部步数、采样下限和 seed；不使用 `latest.pt`。两阶段使用同一个 `--seed`。中断后使用相同命令加 `--resume --execute`；运行期间保持源码和配置不变。

memory 训练和留出集评估统一最短17个观测，参数采用命名层独立初始化；这是三组共同的 `table2_v1` 协议。每个新的完整参考都要按该协议重新训练。

## 多服务器分工与结果回传

建议将 `reference` 和 `no_joint` 分配到同一台服务器，共用一个 memory checkpoint。其他服务器分别承担 `k4`、`k16`、`residual32` 或直接 action 的变体。一个两阶段实验最好在同一台服务器完成，避免跨机路径和 index SHA 不一致。

各服务器统一 Git commit、输入版本、训练 seed、评测任务、评测 seed 和 checkpoint 选择规则。新服务器上的完整环境与正式 GPU 训练需要本机验证。

训练结果写入 `runs/ablations/<experiment>/seed_11/`，全部被 Git 忽略。通过其他存储回传所需的 JSON 指标或 checkpoint；不要把运行目录提交到代码仓库。

完整 EMA 的评测例子：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_joint.py \
  --checkpoint runs/ablations/reference/seed_11/action/checkpoints/step_030000/ema.pt \
  --device cuda:0 --split test --seed-start 50 --episodes-per-task 50 \
  --out runs/ablations/reference/seed_11/eval_test
```

表中的模拟占位分数不属于实验结果。训练中的 validation 用于权重选择，各变体再用相同 test 方案比较真实 TSR/CSR。
