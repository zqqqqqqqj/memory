# Memory capacity

变体：`k4`、`k8`、`k16`，残差维度128。`k8` 是公共参考，等同于另外两组的 `residual128` / `reference`，只训练一次。

每个独立容量需要 memory 60k → action 30k。先按[根目录说明](../README.md)安装环境、配置本服务器路径并准备固定缓存，再运行 `python ablation_memory_capacity/run.py --list`。

默认预览，`--execute` 才训练；结果只写本地 `runs/`，不会上传到 Git。
