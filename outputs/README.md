# outputs/ —— 运行产物

> 宿主机这个目录 = 容器里的 `/outputs`。**所有运行产物写这里**（视频 / 图片 /
> 指标 / 临时输出），别写回 `code/`。结构约定如下，**不要平铺**。

```
outputs/
├── runs/       实验 run（指标类）：一次测试/实验一个目录
│   └── <YYYY-MM-DD>-<名字>[-s<seed>]/
│       ├── summary.md / metrics.json / config_snapshot.yaml   ← run_bench 三件套
│       └── （该次附带的视频/图也放这个目录里）
├── videos/     独立录像：<YYYY-MM-DD>-<描述>.mp4
├── figures/    离线出图：<YYYY-MM-DD>-<描述>.png
└── scratch/    临时/诊断输出，可随时 rm -rf
```

- **命名带日期前缀**（`YYYY-MM-DD-`），按时间排、好查找。
- 指标三件套由 `run_bench.py --out /outputs/runs/<名字>` 自动写出；
  `config_snapshot.yaml` 是**最终生效配置**，复现就拿它当 `--config`。
- 整目录可随时备份/清理，**不含任何源码**。
- 结构约定**以本文件为准**（工作区与交接包同一份）；挂载关系与使用规则见顶层 `README.md`。
