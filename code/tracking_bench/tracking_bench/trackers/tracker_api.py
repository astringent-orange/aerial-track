#!/usr/bin/env python3
"""P4：跟踪器的标准输入 / 输出接口（docs/BENCH_PLAN.md §2.2 / §2.3）。

三样东西：
    TrackerInput   每步喂给跟踪器的观测（字段和 §2.2 一字不差）
    TrackerAction  跟踪器每步的输出（§2.3）
    Tracker        跟踪器基类：reset(seed) + act(obs) -> action

⚠️ **这个模块不 import isaacgym / torch**，只有 dataclass + numpy。
   别人写自己的跟踪器时只需要 import 这一层，不用碰 IsaacGym —— 这是 P5 交接的核心。

字段标记沿用 §2.1：🔵 真实系统里也有；🟠 仿真真值（**用它做决策 = 作弊**，
用它做监督 / 评估 = 正常）。

坐标系 / 单位 / 四元数顺序见 `INTERFACE.md`，这里不重复。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

# 动作模式（docs/BENCH_PLAN.md §2.3）。**在配置里定死**，不每步给标签。
ACTION_MODES = ("position", "velocity", "acceleration")
DEFAULT_ACTION_MODE = "position"

# 跟踪器自己要满足的硬约束（§2.3）。超了 runner 会截断并记一次警告。
SPEED_LIMIT = 3.0          # m/s（位置设定点的变化率）
XY_LIMIT = 9.5             # |x|, |y| 上限
Z_MIN, Z_MAX = 0.5, 4.5    # m


def _arr(x, n=None, dtype=np.float64):
    """统一转成 numpy。给 None 就返回 None（可选字段）。"""
    if x is None:
        return None
    a = np.asarray(x, dtype=dtype).ravel()
    if n is not None and a.size != n:
        raise ValueError(f"期望 {n} 个元素，拿到 {a.size}")
    return a


# --------------------------------------------------------------------------- #
#  §2.2  环境上下文里的障碍物
# --------------------------------------------------------------------------- #
@dataclass
class Obstacle:
    """一个静态障碍（墙 / 圆柱）的外接盒。供避障用，跟踪器可以完全忽略它。"""

    pos: Tuple[float, float]        # (x, y) 地面投影，底面中心
    size: Tuple[float, float, float]  # (sx, sy, sz)
    yaw: float                      # rad，绕 z
    kind: str = "wall"              # "wall" | "cylinder"

    def as_dict(self):
        return {
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "size": [float(v) for v in self.size],
            "yaw": float(self.yaw),
            "kind": self.kind,
        }


# --------------------------------------------------------------------------- #
#  §2.2  TrackerInput
# --------------------------------------------------------------------------- #
@dataclass
class TrackerInput:
    """跟踪器每步拿到的观测。字段分组和 docs/BENCH_PLAN.md §2.2 完全对应。"""

    # ---- 本机状态 🔵 ----
    t: float = 0.0                      # 仿真时刻 [s]
    step: int = 0                       # 步号
    dt: float = 0.1                     # 控制周期 [s]
    chaser_pos: np.ndarray = None       # (3,) 世界
    chaser_vel: np.ndarray = None       # (3,) 世界
    chaser_attitude: np.ndarray = None  # (3,) (roll, pitch, yaw) rad
    chaser_angvel: np.ndarray = None    # (3,) 机体系 rad/s，可选

    # ---- 传感器 🔵 ----
    depth: np.ndarray = None            # (270, 480) float32，值域 [-1] ∪ [0,1]
    rgb: np.ndarray = None              # (270, 480, 3) uint8，可选（占 2 GPU，默认关）
    seg: np.ndarray = None              # (270, 480) int32，🟠 可选，目标 = 100
    K: np.ndarray = None                # (4,) fx, fy, cx, cy
    T_body_cam: np.ndarray = None       # (4, 4) 机体系 -> 相机系

    # ---- 目标真值（状态域）🟠 ----
    target_pos: np.ndarray = None       # (3,) 世界
    target_vel: np.ndarray = None       # (3,) 世界
    target_yaw: float = 0.0             # 世界系 rad
    target_valid: bool = True           # 目标是否存在

    # ---- 目标观测（相机域）🟠 ----
    in_fov: bool = False                # 几何：中心投影在图内。**不管挡没挡**
    visible: bool = False               # 渲染：分割图里有目标像素
    occluded: bool = False              # in_fov and not visible
    bbox: Optional[Tuple[int, int, int, int]] = None      # (u0,v0,u1,v1) | None
    center_pixel: Optional[Tuple[float, float]] = None    # 掩码质心
    projected_pixel: Optional[Tuple[float, float]] = None  # 几何投影点
    target_pixel_area: int = 0          # 掩码像素数
    target_depth_mean: Optional[float] = None  # ⚠️ 混合均值，不能当距离用
    distance: float = 0.0               # 两机欧氏距离

    # ---- 环境上下文 🟠 ----
    arena_bounds: Tuple[np.ndarray, np.ndarray] = None    # (lo(3), hi(3))
    obstacles: List[Obstacle] = field(default_factory=list)
    wall_layout_id: str = "arena_20x20"

    def __post_init__(self):
        self.chaser_pos = _arr(self.chaser_pos, 3)
        self.chaser_vel = _arr(self.chaser_vel, 3)
        self.chaser_attitude = _arr(self.chaser_attitude, 3)
        self.chaser_angvel = _arr(self.chaser_angvel, 3)
        self.target_pos = _arr(self.target_pos, 3)
        self.target_vel = _arr(self.target_vel, 3)
        self.K = _arr(self.K, 4)
        if self.T_body_cam is not None:
            self.T_body_cam = np.asarray(self.T_body_cam, dtype=np.float64).reshape(4, 4)
        if self.depth is not None:
            self.depth = np.asarray(self.depth)
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb)
        if self.seg is not None:
            self.seg = np.asarray(self.seg)
        if self.arena_bounds is not None:
            lo, hi = self.arena_bounds
            self.arena_bounds = (_arr(lo, 3), _arr(hi, 3))

    # 便捷方法：跟踪器多半只用得上这几个
    def cam_center_err(self):
        """`center_pixel` 相对图像中心的 (du, dv)。没有掩码就退回 `projected_pixel`。

        返回 None 表示两个点都没有（理论上不该发生，projected_pixel 总有）。
        """
        p = self.center_pixel if self.center_pixel is not None else self.projected_pixel
        if p is None or self.K is None:
            return None
        return (float(p[0] - self.K[2]), float(p[1] - self.K[3]))


# --------------------------------------------------------------------------- #
#  §2.3  TrackerAction
# --------------------------------------------------------------------------- #
@dataclass
class TrackerAction:
    """跟踪器的输出。动作模式在配置里定死，这里只填用到的那几个字段。

    | 模式 | 用的字段 |
    |---|---|
    | `position`（默认） | `position_setpoint`、`yaw_setpoint`、`velocity_ff`（可选） |
    | `velocity`         | `velocity_setpoint`、`yaw_rate` |
    | `acceleration`     | `accel`、`yaw_rate` —— **本期不做**，接口预留 |

    ⚠️ `velocity_ff` 现在**传不进控制器**：`LeePositionController.update()` 把
       `setpoint_velocity` 写死成 `zeros_like(...)`（`position_control.py:30`）。
       字段留着是为了接口完整，runner 会收下并忽略，同时记一次警告。
       要真吃到前馈得改 aerial_gym 那一行，本期不动它。
    """

    # position 模式
    position_setpoint: np.ndarray = None   # (3,) 世界
    yaw_setpoint: float = 0.0              # rad
    velocity_ff: np.ndarray = None         # (3,) | None，见上

    # velocity 模式
    velocity_setpoint: np.ndarray = None   # (3,) 世界
    yaw_rate: float = 0.0                  # rad/s

    # acceleration 模式（预留）
    accel: np.ndarray = None               # (3,) 世界

    def __post_init__(self):
        self.position_setpoint = _arr(self.position_setpoint, 3)
        self.velocity_ff = _arr(self.velocity_ff, 3)
        self.velocity_setpoint = _arr(self.velocity_setpoint, 3)
        self.accel = _arr(self.accel, 3)

    @classmethod
    def position(cls, pos, yaw=0.0, vel_ff=None):
        return cls(position_setpoint=pos, yaw_setpoint=float(yaw), velocity_ff=vel_ff)

    @classmethod
    def velocity(cls, vel, yaw_rate=0.0):
        return cls(velocity_setpoint=vel, yaw_rate=float(yaw_rate))


# --------------------------------------------------------------------------- #
#  Tracker 基类
# --------------------------------------------------------------------------- #
class Tracker:
    """跟踪器基类。别人只要继承它、实现 `act()`，就能接进整套测试环境。

    最小用法（完整可运行版见 `INTERFACE.md`）::

        class MyTracker(Tracker):
            name = "my_tracker"
            def act(self, obs):
                return TrackerAction.position(obs.target_pos, 0.0)

    约定：
      * `reset(seed)` 在每个 episode 开头调用一次，让跟踪器清状态、定随机性。
      * `act(obs)` **每步**调用一次，返回一个 `TrackerAction`。
      * 跟踪器**不应该**去碰仿真内部（`env_manager` 之类）—— 那是 runner 的事。
      * 用 🟠 字段做决策 = 作弊。真值是给监督 / 评估用的。
    """

    #: 给报告里用的名字
    name = "tracker"

    def reset(self, seed=None):
        """episode 开始时调用。默认什么都不做。"""
        return None

    def act(self, obs: TrackerInput) -> TrackerAction:
        """每步调用。必须实现。"""
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name!r}>"


def action_to_dict(a: TrackerAction):
    """给日志 / 复现清单用的可序列化 dict。numpy 转 list。"""
    d = asdict(a)
    out = {}
    for k, v in d.items():
        out[k] = None if v is None else (v.tolist() if isinstance(v, np.ndarray) else v)
    return out
