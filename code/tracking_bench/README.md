# tracking_bench —— 无人机单目标跟踪测试环境

**给接手的人**：一个 chaser 无人机（挂着相机）在 20×20×5 m 的障碍场地里
跟踪一架自由飞行的目标无人机。你只需要写一个 `Tracker` 类，就能在这里跑
标准化测试、拿到统一口径的指标。**5 分钟跑通**见下面第 3 步。

- **接口文档**（写 tracker 前必读）→ [INTERFACE.md](INTERFACE.md)
- 测试怎么判成功/失败、指标口径 → `INTERFACE.md` §6
- 背后设计与实测数字 → `docs/BENCH_PLAN.md` / `docs/SETUP_NOTES.md`（工作区与交接包里都在 `docs/`）

---

## 1. 环境准备（一次性）

整个环境跑在 Docker 容器 `aerialgym_dev` 里（IsaacGym + PyTorch + 本包），
**宿主机的 python 什么都没有**，一切命令都在容器内执行：

```bash
# 在工作区根目录（本包的上一级）
./start_container.sh                 # 进容器；或 docker exec -it aerialgym_dev bash
cd /workspace/tracking_bench         # 容器内，代码目录的映射见下
```

| 宿主 | 容器 |
|---|---|
| `code/tracking_bench/`（本包） | `/workspace/tracking_bench/` |
| `code/aerial_gym_simulator/`（aerial_gym 源码） | `/workspace/aerial_gym_simulator/` |

**依赖**：无需 pip install —— 容器里已有 Python 3.8 + torch + numpy + PyYAML。
本包也不用 `pip install -e .`：入口脚本自己把包根加进 `sys.path`。

### GPU 规则（这台服务器是多人共享的）

- **任何程序最多同时用 2 张 GPU**。本包默认（warp 渲染，深度+分割）**只占 1 张**；
  要 RGB（`use_warp: false`）才占 2 张。
- 先看哪张卡空（**按显存判断，不按利用率**）：
  `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
- 显式给 `--device cuda:N`；**不要自己设 `CUDA_VISIBLE_DEVICES`**
  （脚本会自己锁到单卡并重 exec，设了反而会占 2 张卡/或直接崩）。

### ⚠️ 对 aerial_gym 的本地修改（换机器必看）

本环境依赖 **aerial_gym 的一处本地补丁**，丢了会**静默出错**（不报异常）：

> `aerial_gym/env_manager/warp_env_manager.py` 的 `post_physics_step()` 里
> 必须调 `self.update_meshes()` —— 否则 warp 渲染的网格不跟物体动，
> 移动物体的深度图冻在出生点。

回归测试 `tests/regression/test_warp_patch.py` 会对着安装好的 aerial_gym 检查这条。
另有两处有意的配置：`tracking_env.py` 的 `reset_on_collision = False`
（碰撞不自动 reset，由测试环境判失败）、`num_envs = 1`（多环境未开放）。

---

## 2. 目录结构

```
tracking_bench/
├── README.md              ← 本文件
├── INTERFACE.md           ← 接口文档（写 tracker 只看它）
├── configs/               场地布局 / 场景 / 测试配置（YAML）
├── scripts/
│   └── run_bench.py       ← **对外唯一入口**
├── examples/
│   └── minimal_example.py 最小 tracker 示例（与 INTERFACE §5 一字不差）
├── assets/                URDF 生成器（墙 / 圆柱 / 目标），一般不用碰
├── tests/
│   ├── run_all.py         离线回归全套（不用 GPU）
│   ├── smoke_import.py
│   ├── regression/        补丁 / 槽位 / 深度值域语义
│   └── acceptance/        check_p1..p4（要 GPU，完整验收）
└── tracking_bench/        ← Python 包
    ├── sim/               **全项目唯一碰 IsaacGym 的地方**
    ├── target/            目标轨迹（路点 + 五次多项式）
    ├── sensors/           相机模型 + 真值
    ├── trackers/          Tracker 基类 + baseline
    ├── eval/              指标 + 报告
    ├── tracking_lib.py    几何与跟随控制律（纯 Python）
    ├── bench_config.py    配置
    ├── bench_env.py       Gym 风格封装（训练用）
    └── runner.py          episode 编排
```

---

## 3. 跑一次（5 分钟）

```bash
cd /workspace/tracking_bench

# ① 离线回归先过一遍（不用 GPU，几十秒）
python tests/run_all.py

# ② 跑一个短 episode（60 步 ≈ 6 s 仿真，1–2 分钟）
python scripts/run_bench.py --config configs/bench.yaml \
    --steps 60 --device cuda:N --out /outputs/runs/demo     # ← N 换成空闲卡号

# ③ 看结果
cat /outputs/runs/demo/summary.md
```

跑完整 600 步（60 s 仿真，几分钟）：把 `--steps 60` 去掉即可。

> **输出约定**：指标写 `/outputs/runs/<日期>-<名字>/`（视频/图/临时产物的目录
> 结构见 `outputs/README.md`，工作区与交接包里都有）。机器上没挂
> `/outputs` 时，`--out` 换成任意可写目录即可，代码不挑路径。

**看结果**：`/outputs/runs/demo/` 里三样东西

| 文件 | 给谁 |
|---|---|
| `summary.md` | 人读：结果 / 跟踪精度 / 可见性 / 复现信息 |
| `metrics.json` | 机器读：同一批数字 + `seed` + 完整路点序列 |
| `config_snapshot.yaml` | **最终生效**的完整配置（复现就拿它当 `--config`） |

关键指标：`distance_err`（离期望跟随距离的偏差）、`los_err_deg`（**视线误差**，
相机指得准不准）、`in_fov_ratio`（目标在视场内的步数占比）、`collision_count`。
口径细节见 `INTERFACE.md` §6.2。判失败的条件（丢目标 3 s / 碰撞）见 §6.1 ——
**失败的 episode 照样写指标**，能看出来差在哪一步。

---

## 4. 写自己的 tracker

**只要一个类**，不用碰 IsaacGym、不用碰仿真内部。模板抄
[examples/minimal_example.py](examples/minimal_example.py)：

```python
from tracking_bench.trackers.tracker_api import Tracker, TrackerAction

class MyTracker(Tracker):
    name = "mine"
    def reset(self, seed=None): ...
    def act(self, obs):
        ...
        return TrackerAction.position(setpoint_xyz, yaw)
```

```bash
python scripts/run_bench.py --tracker-file my_tracker.py \
    --device cuda:N --out /outputs/runs/mine
```

- 每步拿到什么（深度图 / 位姿 / 目标真值 …）→ `INTERFACE.md` §2
- 输出什么（位置设定点 + 偏航，约束）→ §3
- **红线**：带 🟠 标记的字段是仿真真值，**拿它做决策 = 作弊**，只能用于
  reward / 评估。真算法请用 `depth`（必给）/ `rgb`（可选）。

---

## 5. 换场景 / 调参数

| 想改什么 | 改哪 |
|---|---|
| 场地障碍布局 | `configs/arena_20x20.yaml`（墙 + 圆柱，**唯一来源**） |
| 目标速度 / 路点 | `configs/bench.yaml` 的 `target_policy:` |
| 丢失判据 / 超时 / 碰撞是否算失败 | `configs/bench.yaml` 的 `episode:` |
| 种子 / 跑几个 | `--seed` / `--seeds 1,2,3` |
| 特定场景（遮挡 / 丢目标） | `configs/scene_occluded.yaml`、`configs/scene_lost.yaml` |

改了场地布局的墙数/圆柱数 → 重跑 `assets/` 下的生成器重建 URDF，
再跑 `tests/run_all.py`（槽位顺序回归会替你把关）。

---

## 6. 出问题先查这里

| 现象 | 原因 → 修法 |
|---|---|
| 深度图全是 `+1.0` | 超出 10 m 或没打中（**分不开**）；或 warp 相机位姿是 NaN —— 建完环境必须先 reset |
| 深度有负数 | **不是 bug**：`-1.0` = 比 0.2 m 还近。值域是 `[-1] ∪ [0,1]` |
| 目标深度/位置冻在出生点 | aerial_gym 的 `update_meshes()` 补丁丢了，见 §1 |
| 输出落错地方 | `--out` 相对**当前目录**，先 `cd` 再跑 |
| 一张卡变两张 | 自己设了 `CUDA_VISIBLE_DEVICES`，或者开了 RGB（`use_warp: false`） |
| `import` YAML 报 `UnicodeDecodeError` | isaacgym 把默认编码变 ASCII 了，`open()` 要显式 `encoding="utf-8"` |

更完整的踩坑记录 → `docs/SETUP_NOTES.md`（工作区与交接包里都在 `docs/`）。

---

## 7. 验收与复现

- **离线回归**（每次改动后跑）：`python tests/run_all.py`
  —— import 冒烟 + warp 补丁 + 槽位顺序 + 深度值域语义 + 场地布局单一真相源。
- **完整验收**（上 GPU，约 20 分钟）：
  `python tests/acceptance/check_p4.py --device cuda:N --out-dir p4_check`
  （check_p1–p3 是底层组件的验收，按需跑）。
- **复现一次实验**：拿 `/outputs/runs/<name>/config_snapshot.yaml` 当 `--config`，
  加 `--seed <metrics.json 里的 seed>`，逐点一致（`INTERFACE.md` §7 有清单）。
