# 跟踪器测试环境建设计划（阶段 2）

> **面向：执行这份计划的人 / agent。这是逐步执行的运行手册，不是背景资料。**
> 背景、原理、踩坑记录 → `SETUP_NOTES.md`
> 阶段 0/1 的设计与实测数字 → `TRACKING_PLAN.md`
> 动手前的硬规则 → `AGENTS.md`
>
> 最近更新：2026-10-13

---

## 0. 执行约定（先读这段，再动手）

1. **按顺序执行 P1 → P5。** 每期验收没过，不进下一期。
2. **⚠️ 遇到问题自行迭代修复，不要停下来等确认。**
   - 报错、数值不对、验收不过 → 自己诊断、自己修、自己复测；
   - 修完把「现象 / 根因 / 修法 / 怎么验证的」写进 `SETUP_NOTES.md`；
   - 在本文档对应步骤下加**一行指针**指向那条记录，不要把长文抄回来；
   - **只有需求本身有歧义**（比如某个接口字段的含义有两种合理解释）才停下来问。
3. **改 aerial_gym 的渲染 / 资产 / 槽位之前**，先读 `SETUP_NOTES.md` 的
   「跟踪闭环踩过的坑」。那里的坑**全部是静默出错**，不报异常。
4. **GPU 最多同时 2 张**，跑仿真显式选卡。见 `AGENTS.md` §1。
5. **可复现优先。** 布局、路点、seed 全部落盘。"跑起来看着对"但不可复现 = 没做完。
6. **不要 `docker rm` 这个容器，不要动别人的容器。** 见 `AGENTS.md` §2。
   跑之前 `cd /workspace`。

---

## 1. 目标与范围

把现在 10×10 单圆闭环，升级成**可交接的跟踪器测试环境**。

| # | 需求 | 落点 |
|---|---|---|
| 1 | 场地扩到 20×20，加薄墙（可产生遮挡） | P1 |
| 2 | 目标随机路点 + 多项式平滑轨迹，连续飞行 | P2 |
| 3 | 目标真值输出：in_fov / 中心点 / 边界框 / 状态 | P3 |
| 4 | 标准输入输出接口 + 目标控制接口 + 自动统计 | P4 |
| 5 | 代码整理成包，交接给他人复现其他跟踪方法 | P5 |

**不做**：感知算法本身、目标外观建模、多跟踪器对比排行榜、避障规划、低层力矩控制。

**目标使用方式**：只对**一个**跟踪器做训练 / 测试。环境提供真值和传感器，
跟踪器输出控制指令；跑完自动出统计。

---

## 2. 接口规格

> 这一节就是 P4 要产出的 `INTERFACE.md` 的正文。P4 做的时候把它抽出去独立成文件，
> 并补上"最小可运行示例"和"复现清单"两节。

### 2.1 坐标系与单位

| 系 | 定义 | 备注 |
|---|---|---|
| 世界系 | x/y 水平，z 上 | IsaacGym 世界坐标 |
| 机体系 | x 前 / y 右 / z 下 | IsaacGym 机体约定 |
| 相机系 | **OpenCV**：x 右 / y 下 / z 前（光轴） | **v 向下增大** |
| 像素系 | u 右 / v 下，原点左上 | 480 × 270 |

- 单位：**m / s / rad**（角度输出另标注 deg 的除外）
- **四元数一律 `(x, y, z, w)`** —— IsaacGym 的顺序，别按 `(w,x,y,z)` 写
- 深度图值域 **`[-1] ∪ [0,1]`**：`+1.0` = 超出 `max_range`(10m) **或**没打中（分不开）；
  `-1.0` = 比 `min_range`(0.2m) 还近；`[0,1]` = `cam_z / max_range`。
  负数**不是**"没打中"。详见 `SETUP_NOTES.md`。
- 相机 480×270，HFOV 87°（半角 43.5°），VFOV 半角 28.1°。

字段标记：

- 🔵 **真实系统里也有**（传感器 / 里程计）
- 🟠 **仿真真值**（真实系统里没有）。**用它做决策 = 作弊；用它做监督 / 评估 = 正常。**

### 2.2 `TrackerInput` —— 跟踪器每步拿到的

#### 本机状态

| 字段 | 类型 / 形状 | 系 | 标记 | 说明 |
|---|---|---|---|---|
| `t` | float | — | 🔵 | 仿真时刻 [s] |
| `step` | int | — | 🔵 | 步号 |
| `dt` | float | — | 🔵 | 控制周期，0.1 s |
| `chaser_pos` | (3,) | 世界 | 🔵 | 无人机位置 |
| `chaser_vel` | (3,) | 世界 | 🔵 | 无人机速度 |
| `chaser_attitude` | (3,) | — | 🔵 | `(roll, pitch, yaw)` |
| `chaser_angvel` | (3,) | 机体系 | 🔵 | 可选 |

#### 传感器

| 字段 | 类型 / 形状 | 标记 | 说明 |
|---|---|---|---|
| `depth` | (270, 480) float32 | 🔵 | **必给。** 值域 `[-1]∪[0,1]`，见 2.1 |
| `rgb` | (270, 480, 3) uint8 | 🔵 | 可选。**要占 2 张 GPU**，默认关 |
| `seg` | (270, 480) int32 | 🟠 | 可选。逐实例 id，目标 = 100 |
| `K` | `fx, fy, cx, cy` | 🔵 | 相机内参 |
| `T_body_cam` | (4, 4) | 🔵 | 机体系 → 相机系。现在是常量安装 |

#### 目标真值（状态域）

| 字段 | 类型 | 系 | 标记 | 说明 |
|---|---|---|---|---|
| `target_pos` | (3,) | 世界 | 🟠 | |
| `target_vel` | (3,) | 世界 | 🟠 | |
| `target_yaw` | float | 世界 | 🟠 | |
| `target_valid` | bool | — | 🟠 | 目标是否存在 |

#### 目标观测（相机域）

| 字段 | 类型 | 标记 | 说明 |
|---|---|---|---|
| `in_fov` | bool | 🟠 | **几何**：目标中心投影落在图内。**不管挡没挡** |
| `visible` | bool | 🟠 | **渲染**：分割图里有目标像素。**含遮挡** |
| `occluded` | bool | 🟠 | `in_fov and not visible`。被墙挡住时为真 |
| `bbox` | `(u0, v0, u1, v1)` \| None | 🟠 | 掩码外接矩形。None = 不可见 |
| `center_pixel` | `(u, v)` \| None | 🟠 | **掩码质心**。感知系统实际产出的是这个 |
| `projected_pixel` | `(u, v)` | 🟠 | **几何投影点**。供对照 |
| `target_pixel_area` | int | 🟠 | 掩码像素数 |
| `target_depth_mean` | float \| None | 🟠 | 掩码内深度均值 |
| `distance` | float | 🟠 | 两机欧氏距离 |

> **两个中心点都要给。** 评估默认用 `center_pixel`（掩码质心，感知系统实际产出的）；
> `projected_pixel` 用来定位是跟踪器的问题还是成像的问题。
>
> **⚠️ 投影必须用针孔**：`u = 240 + f·cx/cz`，`f = 240/tan(43.5°) = 252.908`
> （和 `warp_cam.py` 同一套内参，`K = (fx, fy, cx, cy)`，fx = fy = f）。
> 本文档早先写的等距柱状 `u = 240 + angle/43.5·240` 是**错的**，和渲染器中间差 16 px。
> 那个"+16px u / +2px v 系统性偏置"也就不存在 —— 实测针孔残差 **< 1 px**。
> 详见 `SETUP_NOTES.md`「投影公式是针孔，不是等距柱状」。

> **`target_depth_mean` 是混合均值，不能当距离用。** 目标贴着相机时，
> 近端像素写 `-1.0`、远端写正值，均值落在中间（实测 -0.543）。
> `truth.py` 同时给 `target_depth_neg1_ratio`（掩码里 `-1.0` 的占比）。

> **`in_fov` 和 `visible` 必须分开。** 加了墙之后"目标丢了"分不清是
> **转头转丢了**（`in_fov=False`）还是**被墙挡了**（`in_fov=True, visible=False`），
> 这两种的处理策略完全不同。

#### 环境上下文

| 字段 | 类型 | 标记 | 说明 |
|---|---|---|---|
| `arena_bounds` | `(lo(3), hi(3))` | 🟠 | 场地边界 |
| `obstacles` | `list[Obstacle]` | 🟠 | 可选。墙/柱的 `(pos, size, yaw)`，供避障用 |
| `wall_layout_id` | str | 🟠 | 当前布局的标识，便于对号 |

### 2.3 `TrackerAction` —— 跟踪器输出

动作模式在**配置里定死**，不用每步给标签。

| 模式 | 字段 | 说明 |
|---|---|---|
| `position`（**默认**） | `position_setpoint (3,)`、`yaw_setpoint (float)`、`velocity_ff (3,) \| None` | Lee 位置控制器吃这个。`velocity_ff` 是可选前馈 |
| `velocity` | `velocity_setpoint (3,)`、`yaw_rate (float)` | 常见替代 |
| `acceleration` | `accel (3,)`、`yaw_rate (float)` | 低层。**本期不做**，接口预留 |

**约束**（跟踪器自己要满足，超了会被截断并记一次警告）：

| 项 | 值 |
|---|---|
| 速度上限 | 3.0 m/s |
| 位置范围 | \|x\|, \|y\| ≤ 9.5，z ∈ [0.5, 4.5] |

### 2.4 `TargetPolicy` —— 目标控制接口

```python
class TargetPolicy:
    def reset(seed: int) -> None                 # 重新生成路点序列
    def state_at(t: float) -> (pos, vel, yaw)    # 查询，不推进
    def set_waypoints(waypoints: list) -> None   # 手动指定（脚本化场景）
    def set_state(pos, vel) -> None              # 直接设定（调试）
    def pause() / resume() -> None               # 冻结 / 解冻
    def set_speed_scale(k: float) -> None        # 整体调速，k>1 加速
    @property
    def waypoints -> list                        # 当前路点序列（供前瞻）
```

默认实现 `WaypointPolicy`（见 P2）。

### 2.5 环境 / 场景配置

YAML 驱动，全部落盘到输出目录。

```yaml
arena:      # configs/arena_20x20.yaml
  bounds: [[-10,-10,0], [10,10,5]]
  walls: [{size: [2.0,0.3,3.0], pos: [4,3], yaw: 0.3}, ...]

camera:
  image: [270, 480]
  hfov_deg: 87.0
  max_range: 10.0
  min_range: 0.2

target_policy:
  type: waypoint
  v_cruise: 1.2        # m/s
  a_max: 1.5           # m/s^2
  z_range: [1.5, 4.0]
  xy_margin: 1.0       # 离墙/离边界的安全边距

tracker:
  type: baseline       # 或别人注册的类名
  standoff: 2.0

episode:
  max_steps: 600
  lost_timeout: 3.0    # 连续丢失 3s 判失败
  collision_is_fail: true

seed: 20261012
num_envs: 4
```

### 2.6 Episode 结果与统计

**终止条件**（任一满足就停，统计立即冻结）：

| 条件 | 结果 | 说明 |
|---|---|---|
| 目标**连续**丢失 ≥ 3.0 s（30 步） | **失败** | 见下方"丢失"的定义 |
| 发生碰撞 | **失败** | 可由 `collision_is_fail` 关掉 |
| 跑满 `max_steps` | 成功 | 前两条都没触发 |

**"丢失"的定义**：`in_fov == False`，即目标中心投影落在图像外。

- 这是**跟踪器能控制的那一项**（把相机指向目标）。
- **遮挡（`in_fov=True, visible=False`）不算丢失**，只记为诊断量——墙挡住了不是跟踪器能控制的。
- 判据做成配置项 `lost_criterion: in_fov | visible`，默认 `in_fov`。
  **等真上感知再切到 `visible`。**
- 计数器连续计：某步 `in_fov=True` 就清零。到 3.0 s 立即终止，**没有"之后恢复"这回事**。

**每 episode 输出**（写 `metrics.json` + `summary.md`）：

| 组 | 字段 |
|---|---|
| 结果 | `result`（success/fail）、`fail_reason`、`fail_step`、`steps_run`、`time_run` |
| 跟踪精度 | `distance_err.{mean,std,max}`（实际距离 vs `standoff`） |
| | `yaw_err_deg.{mean,std,max}` |
| | `pixel_err.{u_mean,u_std,v_mean,v_std}`（`center_pixel` vs 图像中心） |
| 可见性 | `visible_ratio`、`occluded_steps`、`in_fov_ratio` |
| 安全 | `collision_count` |
| 复现 | `config_snapshot`、`seed`、`wall_layout_id`、`waypoints`（完整路点序列） |

**不输出**（按需求收窄）：重捕获时间、丢失事件统计、多跟踪器对比类指标。

> 失败的 episode **照常把已收集的指标写出来**，只是 `result = "fail"`。
> 这样能看出来是差在哪一步。

### 2.7 Gym 风格封装（训练用）

```python
class TrackingEnv:
    def reset(seed=None) -> TrackerInput
    def step(TrackerAction) -> (TrackerInput, done: bool, info: dict)
    def close()
```

- `done` 由 2.6 的终止条件驱动
- `info` 携带当步诊断量（距离误差、偏航误差、像素位置等）
- 可选钩子 `reward_fn(obs, action) -> float`，默认给一个简单实现
  （负的距离误差 + 可见奖励），**训练时用你自己的 reward 覆盖它**

---

## 3. 分期步骤

### P1　场地 20×20 + 薄墙（0.5–1 天）

**做法**

1. `make_wall_assets.py` —— 生成任意尺寸的 box URDF。
   **⚠️ 原点放在底面**（和 `make_cylinder_assets.py` 一致），不要沿用现成
   `1_x_1_wall.urdf` 那种"原点在几何中心"的写法 —— 两套约定混用会摆错高度。
2. 生成两种尺寸：`2.0 × 0.3 × 3.0`（需求给的）和 `1.2 × 0.25 × 2.0`。
   实例数量和摆位由 YAML 决定，生成器只管造 URDF。
3. 新增 `wall_asset_config.py`，`fix_base_link=True`、`semantic_id=-1`。
4. `tracking_env`：边界改 `[-10,-10,0] → [10,10,5]`，`include_asset_type` 加墙。
   **槽位顺序要重算**（`keep_in_env` 用 `appendleft`，后写的插更前），
   目标必须仍在槽 0。启动打印核对。
5. `configs/arena_20x20.yaml` 固定布局。墙放在**目标航路的侧面**，
   能产生间歇遮挡，但不要把整个场地切成两半。

**验收**

- [ ] 资产数量对，启动打印的槽位表和预期一致
- [ ] 墙**立在地面上**（不陷地、不悬空）—— 打一帧深度图肉眼看
- [ ] **warp 深度图里能看到墙**（不是隐形的）
- [ ] 跑 600 步墙**一动不动**（不被 reset 重随机、不被 `env_actions` 推走）
- [ ] 目标不会穿墙（先用固定路点验证）

**产物**：`make_wall_assets.py`、`wall_asset_config.py`、
`tracking_env.py`（改）、`configs/arena_20x20.yaml`

---

### P2　目标运动模型：路点 + 五次多项式（1 天）

**设计**

```
L0 路点层   按 seed 预生成 P0..Pn，自由空间采样
L1 轨迹层   分段五次多项式 Pi -> Pi+1，速度连续
L2 采样层   每步求值 -> (pos, vel, yaw)
```

**路点**：x,y ∈ [−9, 9]、z ∈ [1.5, 4.0]，**必须过自由空间检查**
（墙体 footprint 外扩 0.5m，同时线段 `Pi→Pi+1` 不能穿过墙）。
整条路点序列**用 seed 预生成**——可复现，段间有前瞻可用。

**多项式**：分段五次，边界条件 `p、v` 连续、`a = 0`：

```
p(0)=Pi,   v(0)=vi,   a(0)=0
p(1)=Pi+1, v(1)=vi+1, a(1)=0
```

切向速度取 Catmull-Rom 式：`vi = v_cruise · normalize(Pi+1 − Pi−1)`，
首尾用前向/后向差分。**速度连续 ⇒ 目标不停**。
零加速度边界保证段间没有加速度突变，速度剖面比纯最小-jerk 更接近匀速巡航。

归一化时间 `τ = t/Ti` 下的系数（`p` 按分量算）：

```
a0 = Pi
a1 = vi · Ti
a2 = 0
a3 = 10(Pi+1−Pi) − (6·vi + 4·vi+1)·Ti
a4 = −15(Pi+1−Pi) + (8·vi + 7·vi+1)·Ti
a5 = 6(Pi+1−Pi) − (3·vi + 3·vi+1)·Ti
```

**段时长**按约束反推，不写死：
`Ti = max(T_min, |Δp|/v_max, sqrt(2·|Δp|/a_max))`，
`v_max = 1.2 m/s`、`a_max = 1.5 m/s²`、`T_min = 1.0 s`。

**yaw** = 速度方向（和现在一致）。

**验收**

- [ ] 数值差分验证 `p、v` 连续，`a` 在段边界为 0
- [ ] 目标全程不出墙、不出场地
- [ ] 同 seed 轨迹完全一致，换 seed 轨迹不同
- [ ] 峰值速度 ≤ `v_max`、峰值加速度 ≤ `a_max`
- [ ] 跑 600 步目标全程在动，无瞬移

**⚠️ 这一步末尾必须重测跟随**：

- `POS_LEAD_TIME = 0.83` 理论上仍成立（`τ = Kd/Kp` 是常数，与轨迹形状无关），
  但要复测确认。
- **`YAW_LEAD_TIME` 的减号要重新推。** 那个结论建立在"追踪机在径向外侧、
  视线矢量指向原点"的几何前提上；跟随方位换成"沿目标速度反方向退"之后
  前提没了。重推完把结论写进 `SETUP_NOTES.md`。

**产物**：`tracking_bench/target/waypoint_policy.py`（P5 前可先放 `code/`）

---

### P3　目标真值输出（0.5 天）

**做法**

1. `truth.py`：从 `segmentation_pixels` + `warp_sensor` 真实位姿
   （**别用 euler 猜**，读 `sensor_position` / `sensor_orientation`）算出
   `in_fov / visible / occluded / bbox / center_pixel / projected_pixel /
   target_pixel_area / target_depth_mean / distance`。
2. 投影公式（**针孔**，和 `warp_cam.py` 同一套内参，别改成等距柱状）：

   ```
   cx, cy, cz = R_cam.T @ (target_pos - cam_pos)
   u = 240 + f * cx / cz
   v = 135 + f * cy / cz          # v 向下
   f = 240 / tan(radians(43.5)) = 252.908      # K = (fx, fy, cx, cy)，fx = fy = f
   ```

   ⚠️ 早先这里写的是等距柱状 `u = 240 + angle/43.5·240`，**是错的**，
   和渲染器中间差 16 px。改的理由和实测 → `SETUP_NOTES.md`
   「投影公式是**针孔**，不是等距柱状」。

3. 接进主循环，每步写日志。

**验收**

- [ ] 静止场景手工核对 `bbox` / `center_pixel` / `projected_pixel`，
      确认掩码质心和**针孔**投影对齐（残差 < 3 px，实测 < 1 px）
      —— ~~确认 +16px u / +2px v 偏置仍在~~ 那条是误记，见 §2.2
- [ ] 把目标放到墙后：`in_fov=True, visible=False, occluded=True`
- [ ] 把目标转出画面：`in_fov=False`
- [ ] 目标在近处（< 0.2m）时 `target_depth_mean` 反映 `-1.0` 语义，不报错

**产物**：`truth.py`

---

### P4　接口层 + 自动统计 + 文档（1–1.5 天）

**做法**

1. `Tracker` 基类 + `TrackerInput` / `TrackerAction` dataclass（字段见 §2）。
2. baseline 跟踪器：迁移 `tracking_lib.chaser_follow`，
   **跟随方位改"沿目标速度反方向退 `standoff`"**（目标速度永远不为零，
   这个方向始终有定义）。
3. `run_episode()` / `run_suite()`，实现 §2.6 的终止条件与统计。
4. `eval/`：指标计算 + `metrics.json` + `summary.md`。
5. `TrackingEnv` Gym 风格封装。
6. 把 §2 抽出去独立成 `INTERFACE.md`，补两节：
   - **"怎么写一个自己的 tracker"** —— 一份 20 行以内的最小可运行示例
   - **"怎么复现一次实验"** —— seed / 配置 / 资产版本清单

**验收**

- [ ] baseline 跑 600 步，出 `metrics.json` + `summary.md`，数字合理
- [ ] 把 baseline 换成"什么都不做"的 tracker：**3 s 后判失败并停止**，
      `fail_reason = "lost_timeout"`，`fail_step ≈ 30`
- [ ] 把 `lost_timeout` 调成 30 s，同上场景应该跑满 600 步
- [ ] 遮挡场景：目标在墙后 > 3 s，**不算丢失**（`in_fov` 仍为真）
- [ ] `INTERFACE.md` 里的最小示例**能直接复制粘贴跑通**
- [ ] 换个 seed 跑两次，`metrics.json` 里的 `waypoints` 不同但都能跑通

**产物**：`tracking_bench/` 包、`INTERFACE.md`

---

### P5　代码整理 + 交接（0.5–1 天）

**做法**

1. 抽成自包含包，**核心是把 aerial_gym 藏在 `sim/` 门面后面**：

   ```
   tracking_bench/
   ├── README.md              5 分钟跑通
   ├── INTERFACE.md           接口文档（P4 产出）
   ├── configs/               arena / scenario / camera，YAML
   ├── tracking_bench/
   │   ├── sim/               aerial_gym 封装 ← 全项目唯一 import isaacgym 的地方
   │   ├── assets/            URDF 生成器（墙/柱/目标）
   │   ├── target/            WaypointPolicy + 五次多项式
   │   ├── sensors/           帧读出 + 真值
   │   ├── trackers/          Tracker 基类 + baseline
   │   ├── eval/              指标 + 报告
   │   └── runner.py
   ├── scripts/run_test.py    对外唯一入口
   └── tests/
   ```

2. README：环境准备 → 装依赖 → 跑一次 → 看结果，**5 分钟内走完**。
3. **回归测试断言 vendored 的补丁还在**：
   `warp_env_manager.py` 的 `post_physics_step()` 里必须调 `update_meshes()`。
   这是修"warp 网格不跟物体动"的关键补丁，丢了会静默出错。
   README 里显式说明这是对 aerial_gym 的本地修改。
4. 清理 `code/` 下的临时/诊断脚本，保留能独立复现的。

**验收**

- [ ] 在一个**全新目录**按 README 从零跑通一次
- [ ] 回归测试通过（补丁存在 + 槽位顺序正确 + 深度值域语义）
- [ ] 不看本文档，单凭 README + `INTERFACE.md` 能写一个新的 tracker 并跑起来

**产物**：`tracking_bench/` 包、`README.md`、`tests/`

---

## 4. 已知坑指针

| 坑 | 一句话 | 详见 |
|---|---|---|
| warp 网格不跟物体动 | 移动物体渲染冻结在出生点，**不报错** | `SETUP_NOTES.md`「跟踪闭环踩过的坑」 |
| 槽位顺序 | `keep_in_env` 用 `appendleft`，后写的插更前 | 同上 |
| 深度值域 | `[-1]∪[0,1]`，`+1` = 超程**或**没打中 | 同上 |
| 相机外参 | `randomize_placement=False` 时相机在**机体原点** | 同上 |
| `env_actions` | 不开 `write_to_sim_at_every_timestep` 就完全无效 | 同上 |
| **墙 URDF 原点** | 现成的在**几何中心**，圆柱在**底面** —— 本文 P1 要统一 | §3 P1 |
| **前馈符号** | 偏航那路的减号**已作废**，新几何用预测视线 τ=0.30；位置 0.87 | 同上「跟随方位换了以后…」 |
| **验收判据别依赖随机位姿** | 否则换次 `reset()` 就假失败，量的不是渲染 | 同上 |
| **投影是针孔不是等距柱状** | 用错公式会把 16 px 的参照误差当成"成像偏置" | 同上 |
| **混合深度均值不能当距离** | `target_depth_mean` 可能是 `-1.0` 和正值的混合 | 同上 |
| 对外脚本 cwd | `--out` 相对路径相对**当前目录**，先 `cd /workspace` | `AGENTS.md` §2 |
| `env_actions` 宽度 | = `env_asset_state_tensor.shape[1]`，**别再 `-1`** | `SETUP_NOTES.md`「阶段 2 踩过的坑」 |
| 建完环境先 `reset()` | 否则 warp 相机位姿 NaN、渲染全 miss，**不报错** | 同上 |
| `import isaacgym` 后默认编码变 ASCII | 读 YAML/文本**必须显式 `encoding="utf-8"`** | 同上 |
| 轨迹段定稿后不能再改 | 否则播放中 `state_at(t)` 前后不一致，目标瞬移 | 同上 |
| 别把追踪机指到 `(0,0,0)` | 撞地触发 reset，**全部资产被重采样** | 同上 |
| 路点 0.5 m / 曲线 0.05 m | 两种净距别混用，混了会误报穿墙 | 同上 |
| **自由空间只查墙不查圆柱** | 路点能采到柱面上，目标穿过去、追踪机撞死 | `SETUP_NOTES.md`「阶段 3 踩过的坑」 |
| **一局撞卡，后面全废** | 机身陷在几何体里，settle 的位置设定点顶不出来 —— 要**传送**不要飞 | 同上 |
| **跟随点 ≠ 目标位置** | 目标在自由空间飞，`target − standoff·v̂` 可能在障碍里 | 同上 |
| **`yaw_err_deg` 不是视线误差** | 它量航向对齐；相机指得准不准看 `los_err_deg` | 同上 / `INTERFACE.md` §6.2 |

---

## 5. 总验收

P1–P5 全部通过后，对照下面这张表逐条打勾（**2026-10-12 全部勾完**，各项验收
数字见 §6 对应阶段）：

| # | 验收项 | 对应 |
|---|---|---|
| 1 | ✅ 20×20 场地，固定布局的薄墙，换 YAML 能换场景 | P1 |
| 2 | ✅ 目标按随机路点 + 五次多项式平滑连续飞行，同 seed 可复现 | P2 |
| 3 | ✅ 真值输出含 `in_fov` / `visible` / `bbox` / 两个中心点 / `target_state` | P3 |
| 4 | ✅ 深度图、本机状态、相机内外参都是标准输入的一部分 | P4 §2.2 |
| 5 | ✅ 目标有完整控制接口（reset / set_waypoints / pause / 调速） | P4 §2.4 |
| 6 | ✅ 连续丢失 3 s 自动终止并判失败，遮挡不算丢失 | P4 §2.6 |
| 7 | ✅ 跑完自动出 `metrics.json` + `summary.md`，含配置与 seed | P4 §2.6 |
| 8 | ✅ `INTERFACE.md` 里的最小示例能直接跑通 | P4 |
| 9 | ✅ 别人只写一个 `Tracker` 类就能跑，不用碰 IsaacGym | P5 |
| 10 | ✅ 新目录按 README 从零跑通，回归测试过 | P5 |

---

## 6. 进度记录

> 执行时在这里追加。格式：`日期　P阶段　做了什么　验收过/没过　指针`

**2026-10-12　P1　场地 20×20 + 薄墙　✅ 全部验收通过**

- 做了：`configs/arena_20x20.yaml`（布局唯一来源，8 面墙构成半径 5.5~6.4 的
  **断环**，留走廊）、`make_wall_assets.py`（URDF 原点统一放**底面**）、
  `wall_asset_config.py`（一面墙一个类，`min==max`，reset 就摆对）、
  `tracking_env.py`（边界 20×20×5，`compute_asset_slot_order()` 复现
  `asset_loader` 的排序）、`tracking_lib.py`（槽位按墙数推导，`pin_static_layout`）、
  `check_p1.py`（P1 验收脚本，12 项自动检查）。
- 验收：12/12 通过 —— 槽位表正确（target=0 / 地板=1 / 墙 2..9 YAML 逆序 / 圆柱 10..15，
  共 16）；8/8 面墙 z=0 贴地；warp 分割图里墙/地板/目标都看得到（目标 id=100）；
  600 步静态槽位最大位移 **0.00e+00 m**；目标圆轨迹离墙最近 **3.35 m**。
  产物 `code/p1_check/{depth,seg,topdown}.png`。
- 修了两个坑（现象/根因/修法/验证见 `SETUP_NOTES.md`「阶段 2 踩过的坑」）：
  ① `env_actions` 宽度多减了一次 1；② 建完环境没 `env_manager.reset()`，
  warp 相机位姿 NaN、渲染整帧 miss 且不报错。
- 下一步：P2 目标运动模型（路点 + 五次多项式）。

**2026-10-12　P2　路点 + 五次多项式目标轨迹　✅ 全部验收通过（17/17）**

- 做了：`waypoint_policy.py`（L0 路点层 seed 预生成 + 自由空间检查 →
  L1 分段五次 `p/v` 连续、`a=0` 边界、Catmull-Rom 切向 → L2 逐步采样），
  `check_p2.py`（17 项自动检查，前 4 组纯数值不碰 IsaacGym，第 5 组接仿真跑 600 步）。
- 验收数字：1a 段边界最大跳变 **3.04e-14 m**；1d 解析速度 vs 数值差分 **1.88e-09 m/s**
  （专门抓 `a1 = vi·Ti` 少乘/多乘 `Ti`）；1e 续段到 200 s 后前 12 s 位移 **0.00e+00**；
  2a 50/50 段曲线不出墙不出界；2b 离墙表面最近 **0.78 m**；
  3a/3b 同 seed 逐点一致、换 seed 不同；4a 峰值速度 **1.200/1.2 m/s**；
  4b 峰值加速度 **1.277/1.5 m/s²**；4c 段时长 ∈ [2.47, 6.55] s；
  5e 600 步 **0 次 reset**；5a 目标位置偏差 **6.19e-07 m**；
  5b 步长 ∈ [0.074, 0.120] m（0.74–1.20 m/s）；5c 最大步长 0.120 m vs 阈值 0.360 m。
- **修了一个会静默出错的架构 bug**（现象/根因/修法/验证见 `SETUP_NOTES.md`
  「⚠️⚠️ 轨迹一旦定稿就**不能再改**」）：旧版每加一个路点就整体 `_rebuild()`，
  Catmull-Rom 会回头改写前面几段的形状和 `_cum_t`，播放中 `state_at(t)` 前后不一致，
  仿真里目标跳 0.50 m（4.95 m/s）还会跳进墙。改成**段 append-only**
  （段表滞后一个路点，末端切向定了才建段），并加回归检查 **1e** 把它钉死。
- 另修三个坑：文本编码（`import isaacgym` 后变 ASCII）、追踪机被指到 `(0,0,0)`
  撞地触发 reset 把资产全重采样、5d 拿 0.5 m 路点净距去卡 0.05 m 的曲线而误报穿墙。
- **P2 末尾的强制重测已完成**（`check_follow.py` 三轮 base / refine / confirm）：
  跟随方位换成"沿目标速度反方向退"之后两个前馈超前量全部重推 ——
  **`POS_LEAD_TIME` 0.83 → 0.87**（`沿 v̂` = −1.997，离理想 3 mm）；
  **偏航旧减号作废**（8.53°，比零超前的 4.80° 还差），定稿改用**预测视线**
  `atan2((目标+v·τ)−(追踪机+v_c·τ))`，**τ = 0.30 s → 0.22°**（好 22 倍）。
  重推过程含一处**自我更正**（"沿 v 挪瞄准点转不动视线"是错的，它透过跟踪误差
  `e` 有二阶效果 `δθ = −|v|·τ·(v̂×e)/|LOS|²`）→ `SETUP_NOTES.md`
  「跟随方位换了以后，两个前馈超前量全部重推」。
- 另：P1 的 3a 判据修成与随机初始位姿无关（窗口法是假失败，整帧里墙一直在），
  修完重跑 **12/12 全绿** → `SETUP_NOTES.md`「验收脚本的判据要和机器人初始位姿无关」。
- 下一步：P3 目标真值输出 `truth.py`。

**2026-10-12　P3　目标真值输出 `truth.py`　✅ 全部验收通过（12/12）**

- 做了：`truth.py`（`compute_truth()` 一次给出 §2.2「目标观测」那一整块
  `in_fov / visible / occluded / bbox / center_pixel / projected_pixel /
  target_pixel_area / target_depth_mean / distance`，外加 `K` 内参和几个调试旁支量）、
  `check_p3.py`（12 项自动检查，4 个场景：静止核对 / 墙后遮挡 / 出画 / 贴着相机）。
- 验收数字：1d 针孔投影 vs 掩码质心残差均值 **(−0.25, +0.20) px**、最差 **0.75 px**、
  std (0.35, 0.36)；2a/2b/2c 墙后 `in_fov=True / visible=False / occluded=True`
  （目标在墙外 8 m，掩码 0 px）；3a/3b 出画 `in_fov=False` 且 `occluded=False`；
  4a/4b/4c 贴相机（表面距 0.186 m < MIN_RANGE 0.2 m）时掩码 106273 px、
  其中 **58778 个是 `-1.0`**、`target_depth_mean = −0.543`，且 `in_fov/visible` 不崩。
- **⚠️ 修正了计划里两处错的记述**（现象/根因/修法/验证见 `SETUP_NOTES.md`）：
  ① 投影公式是**针孔**不是等距柱状 —— 那个"+16px u / +2px v 掩码质心偏置"
  **根本不存在**，是 `u = 240 + angle/43.5·240` 和 warp 针孔渲染器之间的差。
  换成 `u = 240 + f·cx/cz`（`f = 240/tan(43.5°) = 252.908`）后残差 < 1 px。
  §2.2 / §3 P3 的公式和验收都已改过来；旧公式留成 `truth.project_equirect()` 留档。
  ② `target_depth_mean` 是 `-1.0` 和正值的**混合均值**，不能当距离用
  （实测 −0.543 → 换算距离 −5.43 m）；补了 `target_depth_neg1_ratio`。
  另踩一个轴向坑：OpenCV 光轴是 **cam-z** 不是 cam-x。
- 下一步：P4 接口层 + 自动统计 + `INTERFACE.md`。

**2026-10-12　P4　接口层 + 自动统计 + 文档　✅ 全部验收通过（22/22，全量 600 步）**

- 做了：`tracker_api.py`（`TrackerInput` / `TrackerAction` / `Obstacle` /
  `Tracker` 基类，§2.2/§2.3 全字段，🔵/🟠 标记）、`truth.py` 接进 `runner.build_obs`、
  `waypoint_policy.py` 实现 §2.4 的 `TargetPolicy`（`set_waypoints` / `pause` /
  `set_speed_scale` / `waypoints` / `duration`）、`bench_config.py`（YAML 驱动）、
  `bench_eval.py`（§2.6 那张表 → `metrics.json` + `summary.md` + `config_snapshot.yaml`）、
  `baseline_tracker.py`（`BaselineTracker` + `DoNothingTracker` + `ConstOffsetTracker`）、
  `runner.run_episode` / `run_suite`、`bench_env.TrackingEnv`（§2.7 Gym 封装）、
  `run_bench.py`（`--tracker-file` 支持外部 tracker）、`minimal_example.py`、
  **`INTERFACE.md`**（对外交接文档，5 分钟上手 + 接口 + 指标口径 + 复现清单）、
  `check_p4.py`（6 组 22 项自动验收）。
- 验收数字（全量 600 步，seed 20261012）：baseline `result=success steps=600`、
  `distance_err mean=0.223 max=0.973`、`in_fov_ratio=0.97`、`collision_count=0`、
  `los_err_deg mean=12.16° max=88.3°`（594 帧）、`yaw_err_deg mean=13.55°`；
  `lost_timeout` 3 s → 第 30 步判失败、30 s / 超长 → 跑满 600 步（3c 三种配置都对）；
  遮挡场景 `occluded_steps=62`、`in_fov_ratio=1.0`、`visible_ratio=0.225`、
  **不判丢失**（4b），判据换成 `visible` 同场景第 30 步判失败（4c）；
  `MinimalTracker` 跑 80 步 `in_fov_ratio=0.875`（5a/5b）；两个 seed 路点不同（6b）。
- 本阶段踩的坑（现象/根因/修法/验证见 `SETUP_NOTES.md`「阶段 3 踩过的坑」，共 10 条）：
  ① `in_fov` 只看中心投影，扩展物体会漏判（改成 `中心在图内 or visible`）；
  ② `T_body_cam` 建环境后没同步就去"实测"，拿到的是初值（改成从安装参数构造常量）；
  ③ `pixel_err` 把 `±1e9` 哨兵平均进去（剔掉 + 报 `n_samples`）；
  ④ `DoNothingTracker` 只锁位置没锁偏航（改成从 settle 结果锁）；
  ⑤ `CHASER_Z_OFFSET=0.8` 是给旧绕圈几何标的，新几何下把目标顶出画面（→0.30）；
  ⑥ 验收脚本里"跑满多少步"误用了主场景的 `max_steps`（改用该组自己的）；
  ⑦ **自由空间检查只查墙不查圆柱**，路点采到柱面上，追踪机第 161 步撞死
     （圆柱布局搬进 arena YAML 的 `cylinders:`，墙+柱一起查）；
  ⑧ **一局撞卡后 `SimSession` 中毒**，后面每个 episode 第 2 步撞死、`dist_err≈12 m`
     （`SimSession.teleport_robot()` 传送机身，`settle` 开头先传）；
  ⑨ **跟随点 ≠ 目标位置**，目标在自由空间飞不代表 `target − standoff·v̂` 也在
     （baseline 加设定点级避障 `tl.avoid_setpoint`，偏航不受避障影响）；
  ⑩ **`yaw_err_deg` 不是视线误差**，它量航向对齐 —— 补 `los_err_deg`（由
     `projected_pixel` 按针孔反推），两个指标在 `INTERFACE.md` §6.2 写清区别。
- **需求措辞的一处消歧**（记在这里免得以后又绕）：§2.6 那两条「3 s 丢失判失败」
  和「跑满 600 步」不矛盾 —— 靠**场景**区分：`scene_lost` 是个绕圈轨迹、
  最长连续出画 **17.4 s**，所以 3 s 超时下第 30 步判失败、30 s 超时下能跑满；
  主场景 `bench.yaml` 的 baseline 本来就一直看着目标，600 步跑满。
- 下一步：P5 代码整理 + 交接。

**2026-10-12　P5　代码整理 + 交接　✅ 全部验收通过（离线回归 4/4、全量 check_p4 复验 22/22、全新目录跑通）**

- 做了：自包含包 `code/tracking_bench/`（树如下），核心是 **IsaacGym 收进
  `sim/` 门面**（`sim/session.py` 是全项目唯一 import isaacgym 的地方）；
  `paths.py` 统一收口所有路径解析（配置 / aerial_gym 资产目录）；
  `scripts/run_bench.py` 对外唯一入口（沿用 P4 已写进 `INTERFACE.md` 的名字，
  不叫计划里草拟的 `run_test.py`，免得文档和脚本对不上）；
  `README.md`（5 分钟跑通）；`tests/`（`smoke_import` + 三条回归 + `run_all.py`
  + `acceptance/check_p1..p4`）；`tests/fixtures/handoff_tracker.py`（交接验证）；
  `INTERFACE.md` 更新到包布局（import / 命令），`code/INTERFACE.md` 改成指针。
- 和计划里那张树的两处出入（不影响验收）：`assets/`（URDF 生成器）放在包外层
  和 `configs/` 平级，不在 Python 包里；入口叫 `run_bench.py`（上一条）。
- 清理：`code/` 只留独立复现的工具 —— `record_sim.py` / `run_tracking.py` /
  `check_follow.py` / `diag_cam.py` / `diag_collide.py` / `aerial_gym_smoke_test.py`
  （import 已改指 `tracking_bench/`），flat 副本（bench 全家桶 + check_p1..p4 +
  make_*_assets + `configs/`）与视频/帧/check 产物全删。`AGENTS.md` 环境速查里
  `make_cylinder_assets` 那行命令随之更新。
- 验收数字：
  - 离线回归 `python tests/run_all.py` **5/5**（import 冒烟 / warp 补丁 /
    槽位 0..15 不重不漏 / 深度值域 `[-1]∪[0,1]` 语义 / 场地布局单一真相源）；
  - **全新目录** `/tmp/fresh_p5` 按 README 走：离线回归全过 →
    `run_bench.py --steps 60` baseline `success 60 步`（`dist_err mean=0.322`
    `in_fov=0.85` `collisions=0`）→ `--tracker-file` 跑 `handoff_tracker.py`
    （**只依据 README + INTERFACE 写的**）`success 60 步`（`dist_err mean=0.551`
    `in_fov=0.833` `collisions=0`），三样产物齐；
  - 障碍表圆柱尺寸修复后**重跑全量 `check_p4` 22/22 绿**：baseline
    `success 600 步`、`dist_err mean=0.227 max=0.977 bias=0.208`、
    `los_err_deg mean=11.71° max=88.34°`（594 帧）、`yaw_err_deg mean=15.11°`、
    `in_fov=0.985 visible=0.985 collisions=0`；`lost_timeout@30`（2b）；
    3c 超时配置驱动；4a `occluded_steps=62`、4b `in_fov=1.0 visible=0.225`
    不判丢、4c 换 `visible` 判据 `lost_timeout@30`；5a/5b MinimalTracker 80 步
    `in_fov=0.875`；6a–6c 两个 seed 路点不同。
- 本阶段踩的坑（现象/根因/修法/验证见 `SETUP_NOTES.md`「阶段 4 踩过的坑」）：
  ⑪ **打包搬家后按 `__file__` 拼的配置路径全歪**（模块深度变了）——
     收成 `paths.py` 一处解析，配置里的 `arena_yaml` 也不再相对 cwd；
  ⑫ 拆 `sim/session.py` 漏了 `Obstacle` import，**运行期**才炸
     （import 冒烟测不出 —— 回归要在全新目录真跑）；
  ⑬ **障碍表把圆柱尺寸写成目标的尺寸** `(0.6,0.6,0.35)`，飞行高度避障
     **静默失效**（改成取 `tl.CYLINDER_SIZE`；之前 check_p4 能过纯属侥幸）；
  ⑭ **清理老 `configs/` 把 aerial_gym 的墙布局读取弄断**（读者不止一个，
     vendored 侧漏改）—— `CFG_PATH` 两级搜索 + 回归 `test_arena_single_source`
     钉"单一真相源"，真建环境复核。
- P1–P5 至此全部完成，总验收表 10/10 已勾（§5）。

**2026-10-12　P5 补充　精简交接包 `aerial-track/`　✅（模拟接收者全流程验证）**

- 背景（新要求）：尽量减少打包内容 —— ① 不传 Docker 镜像，只留 Dockerfile
  让他人自己构建；② aerial_gym 核心是下载的，不打包，只给获取方式 + 补丁；
  ③ 解压后顶层有 README，记录每一部分的作用、该去哪找下一阶段的文件。
- 做了：工作区新增 `aerial-track/`（手写部分：顶层 `README.md` 目录地图、
  `docker/`（Dockerfile + build_image.sh + requirements.txt + Vulkan/EGL json）、
  `scripts/`（setup_workspace.sh / start_container.sh））+ 根目录 `make_aerial_track.sh`
  （同步生成物 + 打 tar）。**产物 `aerial-track.tar.gz` = 161 KB。**
- 有意不打包（理由 → 拿法，交接包 README §5 有表）：镜像 ~27GB → Dockerfile 一条命令；
  IsaacGym tar → NVIDIA 许可自行申请；aerial_gym / urdfpy 源码 →
  `setup_workspace.sh` clone 并钉版本（aerial_gym@f0d0f05 即 tag 2.0.1、
  urdfpy@5466842，PyPI 版 urdfpy 有 cylinder mesh bug 故必须源码）；
  pip 包 ~600MB → 全部烧进镜像；场景 URDF → 生成器三条命令重建；
  阶段 0/1 开发脚本 → 与测试环境无关。
- 本地修改收成**一个补丁** `patches/aerial_gym_local.patch`：`git add -N` 把
  5 个新文件并进 `git diff`（3 改 + 5 增 = 8 文件，678 行，注释自解释），
  URDF 生成物不进补丁。
- 镜像两套机制并存：开发机版 pip 在 bind-mount 的 `container_home/.local`
  （`docker rm` 即丢）；**交接版全部烧进镜像**，容器只挂 `code/`，
  aerial_gym/urdfpy 改用 `PYTHONPATH` 指向 `/workspace`（替代 egg-link），
  重建容器零成本。
- 验收（**按接收者视角全流程**）：解压 tar → `setup_workspace.sh`
  （ghfast 镜像 clone 两个仓库 + `git apply` 成功、正好 8 文件、重复跑幂等）
  → 接收者的 `code/aerial_gym_simulator` 与工作区**逐字节 diff 一致**
  （除 `__pycache__`/egg-info）→ 离线回归 5/5 →
  `run_bench.py --steps 10 --device cuda:3` GPU 冒烟 `success`（16 槽）。
- 未验：Dockerfile 完整 `docker build`（27GB 镜像，共享盘上不重建）——
  与现用镜像同源逐行核对，仅多 pip 层与 PYTHONPATH 两段。
- 交接包内文档两级：顶层 `README.md`（地图 + 三步装机）→
  `code/tracking_bench/README.md` → `INTERFACE.md`；`docs/`（BENCH_PLAN /
  SETUP_NOTES）为可选读物。

**2026-10-13　工作区整理：docs/ 搬迁 + AGENTS 瘦身 + outputs 挂载 + 工作流固化　✅**

- 做了（按用户三点要求）：
  1. **文档按功能归位**：`BENCH_PLAN / SETUP_NOTES / TRACKING_PLAN` 从根目录
     迁到 `docs/`；全库引用 sweep 成 `docs/X.md`（code/ 与交接包文档同步）。
     根目录只留 `README.md`（地图/命令/工作流）+ `AGENTS.md`（红线+入口）。
  2. **AGENTS.md 瘦身到 ≤50 行**：命令速查表、三条坑的正文等可迁移内容全部
     迁出（→ README / docs/SETUP_NOTES.md），只留规则 + 目录入口表；
     新增硬规则「**结论落盘，对话即弃**」。README 新增「工作流」一节：
     开工读什么、收工写什么（进度条目 + 踩坑四行条目两个模板）。
  3. **outputs/ 独立挂载**（宿主 `outputs/` ↔ 容器 `/outputs`，按功能分离，
     与 `container_home/.local` 同一逻辑）：结构约定
     `runs/<YYYY-MM-DD>-<名字>/`（指标三件套）、`videos/`、`figures/`、
     `scratch/`，**禁止平铺**（约定见 `outputs/README.md`）。
     `start_container.sh` 加挂载 + 非交互调用保护；容器已重建。
     `run_bench.py --out` 默认/示例与文档统一到 `/outputs/runs/...`
     （代码不挑路径，没挂载的机器换任意可写目录）。
- **上下文 → 文档的盘点**（用户要求"agent 少依赖上下文"）：核对了私有
  memory 与会话事实，把两条文档里没有的排查经验写进
  `docs/SETUP_NOTES.md`「网络」节（`ls-remote` 偶尔通≠clone 能通；
  **git 重写在容器里，重建后要重配**——本次重建就踩到，已重配）；
  私有 memory 改为纯指针缓存（并修正了其中过期的路径/结论）。
- 验收数字：容器重建后挂载 3/3 在（/workspace、/outputs、~/.local），
  pip 环境完好（numpy 1.23.5），离线回归 **5/5**；
  GPU 冒烟 `run_bench --steps 10 --device cuda:2 --out /outputs/runs/2026-10-13-mount-check`
  `success`，三样产物落在宿主 `outputs/runs/2026-10-13-mount-check/`。
- 踩的坑：无新坑；仅按已记录的「git 重写重建容器会丢」处置。
- 下一步：按 README「工作流」收工即为模板首次落地；aerial-track 侧同步
  （start_container 挂 outputs、示例改 `/outputs/runs/`）并重打 tar。

**2026-10-13　交接包补全：AGENTS.md + outputs 结构 + 使用规则与工作流　✅**

- 背景（新要求）：交接包里补上 ① outputs 输出结构约定；② 工作区整理后的
  使用规则（硬规则 + 工作流模板）。
- 做了：
  1. `aerial-track/AGENTS.md` 新增（硬规则 + 目录入口，与工作区同风格）：
     显式选卡 / 容器与输出红线 / 改环境前必读三条静默坑 / 结论落盘。
  2. `outputs/README.md` 定为**单一事实源**（工作区与交接包同一份，
     `make_aerial_track.sh` 同步）；`aerial-track/outputs/` 带
     `runs/ videos/ figures/ scratch/` 子目录骨架（.gitkeep）。
  3. `aerial-track/README.md` 新增 **§3 使用规则与工作流**（开工/收工仪式 +
     进度/踩坑条目模板 + 信息写哪表）；目录地图补 `AGENTS.md`、`outputs/`。
  4. 插入 §3 后旧 §3–§6 顺移为 §4–§7，**全库章节号交叉引用 sweep**
     （INTERFACE.md、tracking_bench/README、SETUP_NOTES、Dockerfile 注释、
     start_container.sh 注释等 6 处）。
- 验收数字：`./make_aerial_track.sh` 重打 tar **168 KB / 83 项**，补丁仍 8 文件；
  离线回归 **5/5**。
- 踩的坑：无。
- 下一步：讨论把 aerial-track/ 内容发到 GitHub 的方案（见对话）。

**2026-10-13　发布到 GitHub：改名 `aerial-track` + 仓库上线　✅**

- 背景（新要求）：交接文件上传 GitHub，他人按教程逐步构建环境，后续更新好推。
- 做了：
  1. **改名**（用户嫌 `handoff` 不好听、`aerial-gym-tracking-bench` 太长）：
     `handoff/` → `aerial-track/`，`make_handoff.sh` → `make_aerial_track.sh`，
     `handoff.tar.gz` → `aerial-track.tar.gz`；全库引用 sweep（测试夹具
     `handoff_tracker.py` / tracker `name="handoff"` 是接口标识符，刻意保留）。
  2. **仓库**：github.com/astringent-orange/**aerial-track**（Public），
     仓库根 = 原交接包内容；`.gitignore` 挡掉上游源码、运行产物、pyc；
     首个提交 58 项 + tag **`v2026-10-13`**。
  3. **推送通道**：`github.com` 直连超时，但 `ssh.github.com:443` 可达且本机
     密钥已注册 —— 宿主 `~/.ssh/config` 配 `github.com → ssh.github.com:443`，
     push/pull 照常（详见 SETUP_NOTES「网络」节）。`api.github.com` 也可直连。
  4. README 补「怎么拿到它」（git clone / ghfast 镜像 / ZIP / tar 四种）；
     tar 打包时排除 `.git`（169 KB）。
- 验收数字：`git push -u origin main` 成功，`git ls-remote` 见 `main` +
  `v2026-10-13` 两条 ref；改名后打包 169 KB、补丁仍 8 文件、离线回归 5/5。
- 踩的坑：无（SSH 443 通道是新发现，已记 SETUP_NOTES「网络」）。
- 下一步：有代码更新时 `./make_aerial_track.sh` → `cd aerial-track && git commit && git push`；
  可选：GitHub Release 挂 `aerial-track.tar.gz`（网页手动，非必需）。

**2026-10-13　场景巡览视频：tracking_env 三路原地转一圈　✅**

- 做了：
  1. `code/record_sim.py` 补 tracking_env **布局钉固**（reset 后自动
     `tracking_lib.pin_static_layout()`，之前圆柱随机撒，见 SETUP_NOTES
     「用代码搭建环境」末条）。
  2. 录视频 `outputs/videos/2026-10-13-tracking-pan-3view.mp4`：
     `record_sim.py --env tracking_env --view all --use-warp False --pan
     --pan-pos 0 0 2 --steps 300 --device cuda:4`（机位场地中心 z=2，偏航
     -π→+π 匀速扫满一圈；三路 = RGB|深度|分割 并排，同一帧逐像素对齐）。
- 验收数字：300 帧 **1440×270**（3×480×270），10.0 s；抽 4 帧（0/75/150/225）
  核对：墙环带缺口、目标钉在 (2,0,2)、圆柱外圈、四向各有内容。
  占 2 卡（cuda:4 仿真 + 物理 0 号卡上甩不掉的 ~450MB 上下文，RGB 路径的固定开销）。
- 踩的坑：record_sim 不钉布局（SETUP_NOTES「用代码搭建环境」末条，已修）。
- 下一步：无。重录同命令即可；想让目标也动起来改用 `run_tracking.py`。

**2026-10-13　视频返修：pan 开头猛甩 + 分割配色不稳　✅（用户审看视频提出 2 问，均修复重录）**

- 做了：`code/record_sim.py` 两处修复 ——
  1. `--pan` 加 `--settle`（默认 50 步）"到位再开拍"：出生偏航是随机 ±π/6，
     第 0 帧直接指到扫描起点 -π 会原地转 ~180°、开头甩 2 秒；
  2. 分割配色从"本帧名次 k 散列"改成 **id 本身散列**（物体进出画面时名次
     移位，同一物体会每帧变色）。视频同名重录覆盖。
- 验收数字：300 帧 1440×270 / 10.0 s 不变；第 0 帧已是稳态扫描画面；
  目标在 140/160 两帧分割色一致（粉帽+蓝杆），地面/墙色全程一致。
- 踩的坑：SETUP_NOTES「用代码搭建环境」新增 --pan 甩头条目，「分割图的
  显示坑」补正散列键。
- 下一步：无。答复"5 分钟 demo"之问 = 交接包 README 步骤 ③ 的冒烟测试
  （run_bench 60 步，输出 /outputs/runs/demo/ 三件套），文档已有，未改动。
