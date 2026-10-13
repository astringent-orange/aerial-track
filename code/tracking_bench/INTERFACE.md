# 跟踪器测试环境 · 接口文档

> **面向：要在这个环境里跑自己跟踪器的人。** 只看这一份就能写一个 tracker、
> 跑一次测试、看懂报告。背景与踩坑记录 → `docs/SETUP_NOTES.md`；
> 建设计划与进度 → `docs/BENCH_PLAN.md`（工作区与交接包里都在 `docs/`）；
> 共享服务器上的硬规则 → 顶层 `AGENTS.md`（工作区与交接包里都在顶层）。
>
> 最近更新：2026-10-12（对应 docs/BENCH_PLAN.md 阶段 2 / P4）

---

## 0. 五分钟上手

```bash
# 容器内，进包目录（本目录）
cd /workspace/tracking_bench

# 1. 用自带的 baseline 跑一次
python scripts/run_bench.py --config configs/bench.yaml --device cuda:3 --out /outputs/runs/demo

# 2. 看结果
cat /outputs/runs/demo/summary.md
python -c "import json;print(json.load(open('/outputs/runs/demo/metrics.json'))['distance_err'])"
```

产物：`metrics.json`（机器读）、`summary.md`（人读）、`config_snapshot.yaml`（复现用）。
配置路径会自己解析（在哪个目录跑都行）；`--out` 的相对路径是相对**当前目录**的。

**GPU**：默认走 warp 渲染（深度 + 分割），**只占 1 张卡**。显式给
`--device cuda:N`，**不要自己设 `CUDA_VISIBLE_DEVICES`**（脚本会自己锁到单卡）。
要 RGB 就 `use_warp: false`，那样**占 2 张卡**。

---

## 1. 坐标系与单位

| 系 | 定义 | 备注 |
|---|---|---|
| 世界系 | x/y 水平，z 上 | IsaacGym 世界坐标 |
| 机体系 | x 前 / y 右 / z 下 | IsaacGym 机体约定 |
| 相机系 | **OpenCV**：x 右 / y 下 / z 前（光轴） | **v 向下增大** |
| 像素系 | u 右 / v 下，原点左上 | 480 × 270 |

- 单位：**m / s / rad**（角度另标注 deg 的除外）
- **四元数一律 `(x, y, z, w)`** —— IsaacGym 的顺序，别按 `(w, x, y, z)` 写
- **深度图值域 `[-1] ∪ [0,1]`**：
  - `[0,1]` = `cam_z / max_range`（`max_range = 10 m`）
  - `+1.0` = 超出 10 m **或**没打中（**分不开**）
  - `-1.0` = 比 `min_range`（0.2 m）还近。**负数不是"没打中"**
- 相机 480 × 270，HFOV 87°（半角 43.5°），VFOV 半角 28.1°
- **内参**（针孔，方像素）：`fx = fy = f = 240 / tan(43.5°) = 252.908`，
  `cx = 240`，`cy = 135`。投影

  ```
  cx, cy, cz = R_cam.T @ (p_world - cam_pos)     # R_cam 的列是相机三轴在世界系的表示
  u = 240 + f * cx / cz
  v = 135 + f * cy / cz                          # v 向下
  ```

  ⚠️ 这是**针孔**，不是等距柱状。`u = 240 + angle/43.5*240` 那条是**错的**，
  和渲染器中间差 16 px。详见 `truth.project()` / `truth.project_equirect()`。

**字段标记**：

- 🔵 **真实系统里也有**（传感器 / 里程计）
- 🟠 **仿真真值**。**用它做决策 = 作弊；用它做监督 / 评估 = 正常。**

---

## 2. `TrackerInput` —— 每步拿到的观测

定义在 `tracker_api.py`，字段如下。

### 2.1 本机状态 🔵

| 字段 | 类型 / 形状 | 系 | 说明 |
|---|---|---|---|
| `t` | float | — | 仿真时刻 [s] |
| `step` | int | — | 步号（1-based） |
| `dt` | float | — | 控制周期，**0.1 s** |
| `chaser_pos` | (3,) | 世界 | 无人机位置 |
| `chaser_vel` | (3,) | 世界 | 无人机速度 |
| `chaser_attitude` | (3,) | — | `(roll, pitch, yaw)` rad |
| `chaser_angvel` | (3,) | 机体系 | 可选 |

### 2.2 传感器

| 字段 | 类型 / 形状 | 标记 | 说明 |
|---|---|---|---|
| `depth` | (270, 480) float32 | 🔵 | **必给。** 值域 `[-1]∪[0,1]`，见 §1 |
| `rgb` | (270, 480, 3) uint8 | 🔵 | 可选。**要占 2 张 GPU**，`use_warp: true` 时是 `None` |
| `seg` | (270, 480) int32 | 🟠 | 可选。逐实例 id，**目标 = 100**，地板 = 13，`-2` = 没打中 |
| `K` | (4,) `fx, fy, cx, cy` | 🔵 | 相机内参 |
| `T_body_cam` | (4, 4) | 🔵 | 机体系 → 相机系的齐次变换：`p_cam = T_body_cam @ [p_body, 1]`。**旋转块是 body→cam**；相机三轴在**机体系**里的方向是 `T_body_cam[:3,:3].T` 的列。现在是常量安装（相机在机体原点） |

### 2.3 目标真值（状态域）🟠

| 字段 | 类型 | 系 | 说明 |
|---|---|---|---|
| `target_pos` | (3,) | 世界 | |
| `target_vel` | (3,) | 世界 | **轨迹的解析导数**（目标是运动学驱动的，物理速度为 0） |
| `target_yaw` | float | 世界 | = 速度方向 |
| `target_valid` | bool | — | 目标是否存在 |

### 2.4 目标观测（相机域）🟠

| 字段 | 类型 | 说明 |
|---|---|---|
| `in_fov` | bool | **目标有一部分在视场里**：`中心投影在图内 or visible`。**不管挡没挡** |
| `visible` | bool | **渲染**：分割图里有目标像素 |
| `occluded` | bool | `in_fov and not visible` = 中心在图内但被挡住 |
| `bbox` | `(u0,v0,u1,v1)` \| None | 掩码外接矩形。None = 不可见 |
| `center_pixel` | `(u,v)` \| None | **掩码质心**。感知系统实际产出的是这个 |
| `projected_pixel` | `(u,v)` | **几何投影点**。供对照 |
| `target_pixel_area` | int | 掩码像素数 |
| `target_depth_mean` | float \| None | 掩码内深度均值 |
| `distance` | float | 两机欧氏距离 |

> **两个中心点都要给。** 评估默认用 `center_pixel`（感知系统实际产出的）；
> `projected_pixel` 用来定位是跟踪器的问题还是成像的问题。
>
> **⚠️ 目标是扩展物体，`in_fov` 不能只看中心投影。** 直径 0.6 m 的圆柱，
> 中心刚出图时边缘还占着半屏 —— 只判中心会把"目标明明在画面里"算成丢失。
> 所以 `in_fov = 中心在图内 or visible`。三条不变量由此成立：
> `visible ⊆ in_fov`、`occluded ⇒ 中心在图内`、**丢失 = 中心出图且一个目标像素都没有**。
>
> **⚠️ `target_depth_mean` 是混合均值，不能当距离用。** 目标贴着相机时，
> 近端像素写 `-1.0`、远端写正值，均值落在中间（实测 −0.543，直接换算距离会得到
> −5.43 m）。要距离请用 `distance`。
>
> **⚠️ `in_fov` 和 `visible` 必须分开。** 加了墙之后"目标丢了"分不清是
> **转头转丢了**（`in_fov=False`）还是**被墙挡了**（`in_fov=True, visible=False`），
> 这两种的处理策略完全不同。

### 2.5 环境上下文 🟠

| 字段 | 类型 | 说明 |
|---|---|---|
| `arena_bounds` | `(lo(3), hi(3))` | 场地边界，20×20×5 |
| `obstacles` | `list[Obstacle]` | **墙 + 圆柱**的 `(pos, size, yaw, kind)`，供避障用 |
| `wall_layout_id` | str | 当前布局的标识（= arena YAML 的文件名） |

> 圆柱和墙一样是**固定布局**，数据在 `configs/arena_20x20.yaml` 的 `walls:` /
> `cylinders:` 两节（唯一来源）。⚠️ 圆柱**必须**同时进你的避障和自由空间检查 ——
> 踩过：路点采样只查墙不查柱，路点直接采到柱面上（离柱面只剩 0.16 m）。

---

## 3. `TrackerAction` —— 跟踪器输出

**动作模式在配置里定死**（`tracker.action_mode`），不用每步给标签。

| 模式 | 字段 | 说明 |
|---|---|---|
| `position`（**默认**） | `position_setpoint (3,)`、`yaw_setpoint (float)`、`velocity_ff (3,)\|None` | Lee 位置控制器吃这个 |
| `velocity` | `velocity_setpoint (3,)`、`yaw_rate (float)` | 接口预留，**本期跑不了** |
| `acceleration` | `accel (3,)`、`yaw_rate (float)` | 低层。**本期不做**，接口预留 |

**约束**（跟踪器自己要满足，超了会被截断）：

| 项 | 值 |
|---|---|
| 速度上限 | 3.0 m/s（位置设定点的变化率） |
| 位置范围 | \|x\|, \|y\| ≤ 9.5，z ∈ [0.5, 4.5] |

> **⚠️ `velocity_ff` 现在传不进控制器。** `LeePositionController.update()` 把
> `setpoint_velocity` 写死成 `zeros_like(...)`（`aerial_gym/.../position_control.py:30`）。
> 字段收下但丢弃。要真吃到前馈得改 aerial_gym 那一行 —— 本期不动它。
>
> **⚠️ `velocity` / `acceleration` 模式会直接抛 `NotImplementedError`**，
> 因为控制器的动作就是 `(num_envs, 4) = [位置(3), 偏航]`。

---

## 4. `TargetPolicy` —— 目标控制接口

默认实现 `WaypointPolicy`（`waypoint_policy.py`）：路点 + 分段五次多项式，
`p、v` 连续、段边界 `a = 0`，目标**连续飞不停**。

```python
class TargetPolicy:
    def reset(seed: int) -> None                 # 重新生成路点序列
    def state_at(t: float) -> (pos, vel, yaw)    # 查询，不推进
    def set_waypoints(waypoints: list) -> None   # 手动指定（脚本化场景）
    def set_state(pos, vel) -> None              # 直接钉住（调试）
    def clear_state() -> None                    # 恢复轨迹
    def pause() / resume() -> None               # 冻结 / 解冻
    def set_speed_scale(k: float) -> None        # 整体调速，k>1 加速
    @property
    def waypoints -> list                        # 当前路点序列（供前瞻）
    @property
    def duration -> float                        # 已生成轨迹的总时长
```

- 路点用 **seed 预生成**，同 seed 完全一致，换 seed 不同。
- 路点 + 弦的净距 `wall_clearance = 0.5 m`；曲线本身只查 `curve_clearance = 0.05 m`
  （弯道鼓肚会超弦，这是故意分两档的）。
- `set_waypoints()` 会做自由空间检查：路点不在自由空间 / 弦穿墙 / **曲线穿墙**都会拒绝。
  ⚠️ 折线拐急弯时五次曲线会**鼓肚穿墙**（弦是通的也会拒）——把拐角放缓或改用圆弧。
- ⚠️ 自由空间 = **墙 + 圆柱** 都躲。2026-10-12 之前只查墙不查柱，路点能采到柱面上。

---

## 5. 怎么写一个自己的 tracker

**只要一个类。** 不用碰 IsaacGym，不用碰仿真内部。

```python
import math
from tracking_bench.trackers.tracker_api import Tracker, TrackerAction, XY_LIMIT, Z_MIN, Z_MAX


class MinimalTracker(Tracker):
    name = "minimal"

    def reset(self, seed=None):
        return None

    def act(self, obs):
        tp = obs.target_pos
        p = [
            max(-XY_LIMIT, min(XY_LIMIT, float(tp[0]))),
            max(-XY_LIMIT, min(XY_LIMIT, float(tp[1]) - 2.5)),
            max(Z_MIN, min(Z_MAX, float(tp[2]) + 0.8)),
        ]
        yaw = math.atan2(tp[1] - obs.chaser_pos[1], tp[0] - obs.chaser_pos[0])
        return TrackerAction.position(p, yaw)
```

保存成 `my_tracker.py`，跑：

```bash
python scripts/run_bench.py --tracker-file my_tracker.py --device cuda:3 --out /outputs/runs/mine
```

`run_bench.py` 会去文件里找 `Tracker` 的子类（有多个就取第一个）。

> 上面这段就是 `minimal_example.py` 的内容，**一字不差**，`check_p4` 会跑它。
> ⚠️ 它用 `obs.target_pos` 是为了短。那是**真值**，拿它做决策 = 作弊。
> 真算法请用 `obs.depth` / `obs.rgb`。

### 更完整的例子

`baseline_tracker.py` 里的 `BaselineTracker` 是个真能用的实现：沿目标速度反方向
退 `standoff` 米、机头指向目标（预测视线），带速度限幅。控制律在
`tracking_lib.chaser_follow_behind`，两个前馈超前量是实测定稿的
（`POS_LEAD_TIME = 0.87 s`、`YAW_LEAD_TIME = 0.30 s`）。

它还带一层**设定点级的几何避障**（`tracking_lib.avoid_setpoint`）：期望点进了障碍
或直线被挡就绕追踪机水平转个角。**必须有** —— 期望点是 `target − standoff·v̂`，
目标自己在自由空间里飞不代表这个偏移点也在；不避障的话 600 步里有几步会直接
穿进圆柱/墙，物理体撞上就判失败。偏航不受避障影响，始终指向目标。

### 一个纯深度的 tracker 骨架

```python
class DepthOnlyTracker(Tracker):
    name = "depth_only"

    def act(self, obs):
        d = obs.depth                       # (270, 480) float32，[-1]∪[0,1]
        # ……你自己的检测 / 跟踪逻辑，只用 obs.depth / obs.K / obs.chaser_* ……
        return TrackerAction.position(self.next_setpoint, self.next_yaw)
```

**红线**：`target_pos` / `target_vel` / `bbox` / `center_pixel` / `seg` 这些 🟠 字段
**用它做决策 = 作弊**。它们是给 reward / 评估用的。

---

## 6. Episode 结果与统计

### 6.1 终止条件（任一满足就停，统计立即冻结）

| 条件 | 结果 | 说明 |
|---|---|---|
| 目标**连续**丢失 ≥ `lost_timeout`（默认 3.0 s = 30 步） | **失败** | `fail_reason = "lost_timeout"` |
| 发生碰撞 | **失败** | `fail_reason = "collision"`，可由 `collision_is_fail: false` 关掉 |
| 跑满 `max_steps` | 成功 | 前两条都没触发 |

**"丢失"的定义**（`lost_criterion`）：

- `in_fov`（**默认**）：目标中心投影落在图像外。这是**跟踪器能控制的那一项**
  （把相机指向目标）。
- `visible`：分割图里没有目标像素。
- **遮挡（`in_fov=True, visible=False`）在 `in_fov` 判据下不算丢失** ——
  墙挡住了不是跟踪器能控制的。等真上感知再切到 `visible`。
- 计数器**连续计**：某步没丢就清零。到 `lost_timeout` 立即终止，
  **没有"之后恢复"这回事**。

### 6.2 指标口径

每 episode 写 `metrics.json` + `summary.md`。**失败的 episode 照常把已收集的指标
写出来**，只是 `result = "fail"`，这样能看出来差在哪一步。

| 组 | 字段 | 口径 |
|---|---|---|
| 结果 | `result` / `fail_reason` / `fail_step` / `steps_run` / `time_run` | `fail_step == steps_run`（在记录之后才判终止） |
| 跟踪精度 | `distance_err.{mean,std,max,bias}` | `\|distance − standoff\|` 的统计；`bias` 是带符号均值 `distance − standoff` |
| | `los_err_deg.{mean,std,max,n_samples}` | **视线误差**（deg，恒 ≥ 0）：相机光轴 与 相机→目标连线 的夹角，由 `projected_pixel` 按针孔模型反推 `atan2(hypot(u−cx, v−cy), f)`。**相机指得准不准看这个** |
| | `yaw_err_deg.{mean,std,max,bias}` | `\|wrapped(target_yaw − chaser_yaw)\|` 的统计，deg；`bias` 带符号均值。⚠️ 这是**航向对齐**误差，**不是**视线误差 —— 目标横穿画面时相机好好盯着，它一样能到 90° |
| | `pixel_err.{u_mean,u_std,v_mean,v_std,n_samples}` | `center_pixel − 图像中心`，**带符号**（要看系统性偏移）。只统计**目标在像平面前**的帧；目标跑到相机背后时 `projected_pixel` 是 ±1e9 的哨兵，会被剔掉，`n_samples` 是实际用掉的帧数 |
| 可见性 | `in_fov_ratio` / `visible_ratio` | 为真的步数占比 |
| | `occluded_steps` | `occluded` 为真的**步数** |
| 安全 | `collision_count` | 发生过碰撞的**控制周期数** |
| 复现 | `config_snapshot` / `config_snapshot_path` / `seed` / `wall_layout_id` / `waypoints` | 见 §7 |

> **`los_err_deg` 和 `pixel_err` 不是同一件事。** 前者用 `projected_pixel`
> （几何投影），量相机指得准不准；后者用 `center_pixel`（掩码质心），
> 是感知系统实际产出的东西，会把检测抖动算进来。定位问题时两个都看。

**不输出**（按需求收窄）：重捕获时间、丢失事件统计、多跟踪器对比类指标。

### 6.3 碰撞怎么测的

每个控制周期里，`‖robot_contact_force‖ > 0.05 N` 的物理子步会被 aerial_gym
累进 `global_tensor_dict["crashes"]`。周期内只要有一次就算"撞了"。

⚠️ `tracking_env.py` 里 **`reset_on_collision = False` 是故意的**：撞了之后
aerial_gym 不自动 reset，由测试环境自己判失败。要是改回 `True`，一次碰撞就会把
布局和目标全重采样，episode 数据直接作废。

---

## 7. 怎么复现一次实验

跑完之后 `/outputs/runs/<name>/` 里有三样东西，**缺一不可**：

| 文件 | 作用 |
|---|---|
| `config_snapshot.yaml` | **最终生效**的完整配置（含所有默认值），不是你写的那一份 |
| `metrics.json` | 指标 + `seed` + 完整 `waypoints` 序列 |
| `summary.md` | 人读摘要 |

复现就是：

```bash
cd /workspace/tracking_bench
python scripts/run_bench.py --config /outputs/runs/<name>/config_snapshot.yaml \
                            --seed <metrics.json 里的 seed> \
                            --device cuda:3 --out /outputs/runs/<name>_repro
```

同 seed 的 `waypoints` 会**逐点一致**（`check_p2` 的 3a 验过：逐点偏差 0）。

### 复现清单

跑一次实验要记齐这几项：

| # | 项 | 哪里拿 |
|---|---|---|
| 1 | `seed` | `metrics.json` 的 `seed` |
| 2 | 配置 | `config_snapshot.yaml`（整份） |
| 3 | 场地布局 | `configs/arena_20x20.yaml`（`wall_layout_id` 指向它） |
| 4 | 路点序列 | `metrics.json` 的 `waypoints`（也可以从 seed 重新生成） |
| 5 | 资产版本 | 场地 YAML 里每个 box / 圆柱 / 目标的 `size`；URDF 由 `make_wall_assets.py` / `make_cylinder_assets.py` / `make_target_asset.py` 生成 |
| 6 | 跟踪器代码 | 你自己的 `.py`（`--tracker-file` 那个） |
| 7 | 前馈常数 | `tracking_lib.py` 的 `POS_LEAD_TIME` / `YAW_LEAD_TIME`（换控制周期或 Lee 增益要重测） |

> **⚠️ 换控制周期 / 换 Lee 增益 / 换轨迹速度，两个前馈超前量都要重测。**
> 它们是针对 `dt = 0.1 s`、`v_cruise = 1.2 m/s`、Lee 增益 `(2,2)/(2,2)` 标定的。

---

## 8. 环境 / 场景配置

YAML 驱动（`bench_config.py`），所有字段见 `configs/bench.yaml` 的注释。
最小配置可以只有几行，没写的字段用默认值。

```yaml
arena_yaml: configs/arena_20x20.yaml

target_policy:
  type: waypoint
  v_cruise: 1.2          # m/s
  v_max: 1.2             # 段时长按它反推
  a_max: 1.5
  waypoints: null        # 给了就走 set_waypoints()（脚本化场景）

tracker:
  type: baseline         # baseline | do_nothing | 你注册的类名
  action_mode: position
  standoff: 2.0

episode:
  max_steps: 600         # 600 步 x 0.1 s = 60 s
  lost_timeout: 3.0      # s
  lost_criterion: in_fov # in_fov | visible
  collision_is_fail: true
  settle_steps: 40
  start_pose: null       # [x, y, z, yaw]；null = 按 baseline 几何反推

seed: 20261012
num_envs: 1              # ⚠️ P4 只支持 1（见下）
use_warp: true           # true = depth/seg，1 卡；false = RGB，2 卡
```

几何约束（路点的 xy/z 范围、墙净距）在 **`arena_yaml` 的 `waypoints:` 一节**，
不在 `target_policy` 里 —— 两处都写会变成两个真相源。

### 已知限制

| 限制 | 说明 |
|---|---|
| `num_envs` 只能是 1 | 多环境是锁步跑的，要等 `tracking_bench.sim` 门面（P5）。现在会抛 `NotImplementedError` |
| `velocity_ff` 传不进控制器 | 见 §3 |
| `velocity` / `acceleration` 动作模式 | 接口留着，本期跑不了 |
| RGB 要 2 张卡 | `use_warp: false`；默认关 |

---

## 9. 用 Gym 风格接口做训练

`bench_env.py` 的 `TrackingEnv`（刻意不叫 `tracking_env`，那个名字被 aerial_gym
的环境配置占了）：

```python
from tracking_bench.bench_env import TrackingEnv

env = TrackingEnv(cfg, device="cuda:3")
obs = env.reset(seed=20261012)
done = False
while not done:
    action = my_policy(obs)                 # -> TrackerAction
    obs, done, info = env.step(action)
    r = info["reward"]                      # 或 env.reward_fn(obs, action)
env.close()
stats = env.stats                           # §6 那张表
```

- `done` 由 §6.1 的终止条件驱动。
- `info` 携带当步诊断：`distance_err` / `in_fov` / `visible` / `occluded` /
  `reward` / `fail_reason`。
- **默认 `reward_fn` 只是让封装能用**：`−|distance − standoff| + 1·visible − 0.5·(not in_fov)`。
  **训练时用你自己的覆盖它**（`TrackingEnv(cfg, reward_fn=my_reward)`）。

⚠️ 记录口径：`TrackingEnv` 每个 `step()` 记一条（描述该步**结束**时的状态），
`reset()` 返回的初始观测不记录。所以 `stats["steps_run"] == step()` 的调用次数。

---

## 10. 常见问题

**深度图全是 `+1.0`？** 超出 10 m 或者没打中。也可能是 warp 相机位姿是 NaN ——
建完环境必须先 `env_manager.reset()`，这个错了**不报异常**。

**目标不动了 / 冻在出生点？** warp 渲染的网格不跟物体动。`warp_env_manager.py`
的 `post_physics_step()` 里必须调 `update_meshes()` —— 丢了这条会静默出错。

**`import` YAML 报 `UnicodeDecodeError`？** `import isaacgym` 之后默认编码变 ASCII，
`open()` 必须显式 `encoding="utf-8"`。

**命令行被 isaacgym 吃了？** 它的 gymutil 会解析你的 argv，`--sim` 能匹配它的
`--sim_device`。`runner.build_env_quiet()` 已经把 argv 藏起来了，别绕过它。

**判据换了个 reset 就假失败？** `env_manager.reset()` 会随机机器人初始位姿。
验收脚本里**必须**钉死 `episode.start_pose`。
