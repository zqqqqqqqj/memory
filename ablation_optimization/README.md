# Optimization

变体：`reference`、`no_joint`、`no_pretraining`、`no_pretraining_no_joint`。

`reference` 先训练 memory 60k，再训练 action 30k；`no_joint` 复用同一个 memory，只关闭 action 阶段的历史重建。建议把两者放在同一服务器。

`no_pretraining` 和 `no_pretraining_no_joint` 直接训练 action 30k：记忆结构保留，writer/reader 随机初始化，通过动作损失学习；前者还加入历史重建目标。所有 action 骨干均从官方底座开始。

环境、路径配置及启动命令见[根目录说明](../README.md)。默认预览，`--execute` 才训练。
