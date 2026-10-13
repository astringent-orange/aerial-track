# 使用规则（交接包 · coding agent 动手前必读）

> **⚠️ 不要随意修改本文件。** 只放**每次动手前必须遵守的硬规则**和**目录入口**，
> 要求 ≤50 行。原理、命令速查、踩坑、流水账一律写进 README.md / docs/，
> 本文件只留一行指针。信息写哪 → `README.md`「使用规则与工作流」。

## 1. GPU：显式选卡

跑仿真**显式给 `--device cuda:N`**；用 `run_bench.py` 时
**不要自己设 `CUDA_VISIBLE_DEVICES`**（脚本自己锁单卡，设了会崩或多占卡）。
共享机器上**任何程序最多同时用 2 张 GPU**（单任务多卡、多任务单卡都算）；
挑卡**按显存判断，不按利用率**（idle 容器会白占几 GB）。
→ 选卡命令、单卡锁原理：`README.md` §6；`docs/SETUP_NOTES.md`「GPU 怎么用」

## 2. 容器与输出红线

- 重建/进入容器一律 `bash scripts/start_container.sh`，**别手敲 `docker run`**
  （pip 依赖已烧进镜像，重建零成本，但挂载都配在脚本里）。
- `docker ps -a` 是**整台机器所有人**的容器，只动自己创建的。
- **跑之前先 `cd /workspace`**；运行产物一律写 `/outputs/`，结构约定
  （`runs/` `videos/` `figures/` `scratch/`，带日期命名，**不要平铺**）→ `outputs/README.md`。

## 3. 改环境 / 渲染 / 驱动物体前必读

`docs/SETUP_NOTES.md` 的「用代码搭建环境」「跟踪闭环踩过的坑」——
**三条会静默出错、不报异常的坑**（warp 看不见解析地面 / 深度值域
`[-1]∪[0,1]` / warp 网格不跟物体动），不读必踩。任何新物体必须先有 URDF，
加完在 `aerial_gym/env_manager/__init__.py` 注册名字才能用 `--env`。

## 4. 结论落盘，对话即弃

**只存在于对话里的信息视为丢失。** 实验数字 → 进度条目（模板见
`README.md`「使用规则与工作流」）；踩的坑 → `docs/SETUP_NOTES.md`
（现象/根因/修法/验证 四行）；决策与理由 → 文档正文。

## 5. 目录入口

| 要找什么 | 去哪 |
|---|---|
| 装机步骤 / 目录地图 / 使用规则与工作流 | `README.md` |
| 接口规格（写 tracker 只看它） | `code/tracking_bench/INTERFACE.md` |
| 怎么跑测试 / 改场景 / 排查速查表 | `code/tracking_bench/README.md` |
| 指标口径、失败判据 | `code/tracking_bench/INTERFACE.md` §6 |
| 环境原理、GPU 原理、踩坑全记录 | `docs/SETUP_NOTES.md` |
| 设计取舍与验收数字 | `docs/BENCH_PLAN.md` |
| 输出产物结构 | `outputs/README.md` |
