# Residual memory

变体：`no_residual`、`residual32`、`residual128`，固定 K=8。`residual128` 复用公共参考。

`residual32` 需要独立 memory 60k → action 30k；`no_residual` 保留粗摘要、移除残差存储和注入、关闭残差目标，直接 action 30k。未启用的残差模块保留冻结的 checkpoint 键，但不参与前向和优化，实际持久残差张量为空。

环境、路径配置及数据准备见[根目录说明](../README.md)。使用 `python ablation_residual_memory/run.py --list` 查看变体。默认预览，`--execute` 才训练。
