"""跟踪闭环用的几何与控制，阶段 0/1（闭环）和阶段 2（测试环境）共用。

拆成一个库是因为几件事都要用同一批常量，写两遍必然会对不上：
  - 墙 / 圆柱固定布局（配置做不到 6 根不同位置，见 pin_static_layout）
  - 目标轨迹 / 追踪机跟随

槽位约定（由 tracking_env.py 的 asset_type_to_dict_map 顺序 + keep_in_env 的
appendleft 规则决定，本文件按 YAML 墙数自动推导）：

    槽 0      target
    槽 1      bottom_wall（地板 mesh）
    槽 2..    walls（N 面，**YAML 逆序**：槽 2 = wall_{N-1} ... 槽 N+1 = wall_00）
    之后      cylinders（6 根）

改 tracking_env 的 map 顺序或 configs/arena_20x20.yaml 的墙数就会改这些数字，
先跑 check_p1.py 打印 ASSET_SLOT_ORDER 核对。

⚠️ 本文件**不 import aerial_gym / isaacgym**（纯几何，谁都能先 import）。
   布局数据直接读 YAML。
"""

import math
import os

import torch
import yaml

# ---- 场地布局（configs/arena_20x20.yaml 是唯一来源）----
# 路径经 paths.find_config 统一解析 —— 打包后模块在子包里，按 `__file__` 直接
# 拼 `configs/` 会拼到 `tracking_bench/tracking_bench/configs/`（踩过）。
from tracking_bench.paths import find_config as _find_config

_CFG_PATH = _find_config("arena_20x20.yaml")
with open(_CFG_PATH, encoding="utf-8") as _f:
    ARENA = yaml.safe_load(_f)

WALL_LAYOUT = ARENA["walls"]  # [{size, pos, yaw}, ...]，YAML 顺序
N_WALLS = len(WALL_LAYOUT)

# 圆柱布局也在 YAML（`cylinders:` 一节）。见该节的 ⚠️ —— 圆柱必须同时进
# 自由空间检查，只查墙不查柱会让路点采到柱面上。
_CYL = ARENA.get("cylinders", {}) or {}
CYLINDER_SIZE = tuple(float(v) for v in _CYL.get("size", (0.6, 0.6, 3.0)))
CYLINDER_XY = [tuple(float(v) for v in p) for p in _CYL.get("positions", [])]
N_CYLINDERS = len(CYLINDER_XY)  # tracking_env._tracking_cylinders.num_assets

ARENA_BOUNDS_MIN = ARENA["arena"]["bounds_min"]
ARENA_BOUNDS_MAX = ARENA["arena"]["bounds_max"]
ARENA_XY = max(abs(ARENA_BOUNDS_MIN[0]), abs(ARENA_BOUNDS_MAX[0]))  # 10.0
ARENA_Z = ARENA_BOUNDS_MAX[2]  # 5.0

# ---- 槽位（按上面的规则推导，别硬编码）----
TARGET_SLOT = 0
BOTTOM_WALL_SLOT = 1


def wall_slot(i):
    """YAML 里第 i 面墙落在哪个槽。keep_in_env 组内是逆序的。"""
    return 2 + (N_WALLS - 1 - i)


CYLINDER_SLOT_START = 2 + N_WALLS
CYLINDER_SLOT_END = CYLINDER_SLOT_START + N_CYLINDERS  # 开区间

# ---- 目标轨迹（docs/TRACKING_PLAN.md §4.2）----
TARGET_RADIUS = 2.0  # m
TARGET_PERIOD = 12.0  # s，线速度 ≈ 1.05 m/s
TARGET_HEIGHT = 2.0  # m

# ---- 追踪机（docs/TRACKING_PLAN.md §4.3，2026-09-30 重算）----
#
# 追踪机恒在目标的**径向外侧**，所以扫掠半径 = 目标半径 + standoff = 2.0 + S。
# 原计划 S = 2.5（扫掠半径 4.5），但外围圆柱在半径 4.7，只剩 0.2 m 间隙，
# 会被撞到。改成 S = 2.0 -> 扫掠半径 4.0，到圆柱面留 0.3 m。
#
# 代价：两机距离从 2.5m 缩到 2.0m。目标 0.6m 宽在 2.0m 处张角 17°，
# 占 87° 视场的 20%，看得非常清楚，对闭环验证没有影响。
STANDOFF = 2.0  # m，期望的两机距离
CHASER_XY_LIMIT = 9.5  # |x|,|y| 上限。20x20 场地给到 9.5，留 0.5m 边距（docs/BENCH_PLAN.md §2.3）

# 追踪机要比目标**高**这么多，让它俯视目标。
#
# ⚠️ 2026-10-12 从 0.8 改成 0.30。0.8 是给**旧的绕圈几何**标的，前提是位置环有
#    约 0.5 m 的常值下沉（命令 z=2.0 实际停 z≈1.46，绕圈的向心加速度造成），
#    扣掉后真实相对高度只有约 0.3 m。**新几何没有这个下沉** —— 分段五次轨迹不
#    产生持续向心加速度，diag_truth 实测 `sp_z − target_z` 恒等于设定的偏移量、
#    追踪机跟设定点的偏差 ≤ 0.41 m（只在下降时滞后）。于是真实相对高度就是
#    0.8 m：2 m 站位俯角 atan(0.8/2) = 21.8°，把 28.1° 的垂直半视场吃掉 78%，
#    目标被顶在画面下缘（proj_v ≈ 250/270），距离一近到 1.5 m 以内就掉出图。
#
# 0.30 m 的账（相机是**刚性前视、无云台**，俯角只由相对高度决定）：
#   * 站位 2.0 m：俯角 8.5°，proj_v ≈ 173，目标在画面中偏下，上下都有余量
#   * 相对高度 ±0.5 m 的扰动：proj_v ∈ [110, 236]，稳在 [0,270] 内
#   * 目标 z 顶到 4.0 时 clamp 会把偏移压成 -0.3（追踪机反而低一点），
#     俯角 -8.5°，proj_v ≈ 97，照样在画内
#
# ⚠️ 改它不影响水平通道：`chaser_follow_behind` 的 `desired[2]` 不吃 pos_lead，
#    `v_hat` 也只用 xy 速度，所以 POS_LEAD_TIME / YAW_LEAD_TIME 不用重标。
CHASER_Z_OFFSET = 0.30  # m
CHASER_Z_MIN = 1.5
CHASER_Z_MAX = 4.0

# ---- 前馈超前量（2026-10-12 重标定：新跟随方位「沿目标速度反方向退」）----
#
# 理论没变：Lee 位置控制器是 PD 无积分项，跟**匀速斜坡**设定点会留下一个和速度
# 成正比的常值误差（稳态净力矩为零 -> e = (Kd/Kp)·v）。把设定点往前推 τ 就抵掉。
# xy 面 K_pos/K_vel = 2.0/2.0 = **1.0 s**（lee_controller_config，randomize 关），
# 所以 τ 的理论值是 1.0，实测值永远略小 —— 五次轨迹不是纯斜坡，拐弯时 v̂ 在转，
# limit_speed 还会额外吃掉一点。
#
# ⚠️ 2026-09-30 那版 0.83 是在**旧几何**（圆轨迹 + 追踪机在径向外侧）上标的，
#    跟随方位换成"沿目标速度反方向退"后前提没了，2026-10-12 全部重测。
#    三轮对照（check_follow.py，seed 20261012，400 步取后半段稳态）：
#
#   位置前馈 —— 判据是 `沿 v̂ = (追踪机 − 目标)·v̂` 的均值，理想 = −standoff = −2.00：
#     pos_lead  0.00    0.45    0.70    0.83    0.87    0.95    1.10    1.20
#     沿 v̂    -2.875  -2.422  -2.169  -2.037  -1.997  -1.917  -1.767  -1.667
#     对 pos_lead 是干净的直线，斜率 1.008 ≈ 轨迹平均 |v|，零点拟合在 **0.868**。
#     0.87 实测 −1.997，离理想只有 **3 mm** —— 拟合和实测对上了。
#     ⚠️ 别拿 `||d|−standoff|` 当位置前馈的判据：距离对了但方位歪了，它一样小，
#        而方位歪是偏航环的事。
#
#   偏航 —— 旧几何那个**减号已经作废，而且在新几何下有害**（8.53° vs 零超前 4.80°）。
#   新几何里正确形式是**预测视线**，不是挪瞄准点：
#       yaw_cmd = atan2( (target + v·τ) − (chaser + v_c·τ) )
#   （视线本身就在 v̂ 方向上，瞄准点沿 ±v 平移的一阶效果为零，只剩跟踪误差 e 透过
#    δθ = −|v|·τ·(v̂×e)/|LOS|² 的二阶效果 —— 所以 +0.50 挪瞄准点能从 4.80° 压到
#    2.44°、−0.50 却涨到 8.53°，但都不如直接用预测视线。这里要更正我先前
#    "沿 v 挪瞄准点在新几何下转不动视线"的说法：能转，只是不干净。）
#   视界 τ 的扫描是 U 形，谷底在 **0.30 s**：
#     τ        0.10   0.15   0.25   0.30   0.35   0.40   0.50   (零超前)
#     偏航°     3.18   2.43   0.92   0.22   0.60   1.33   2.79      4.80
#     0.30 s 的 0.22° 比零超前好 22 倍。
#
# 前两个数只对 dt=0.1 s、v_cruise=1.2 m/s、Lee 增益 (2,2)/(2,2) 这套配置有效；
# 换控制周期 / 换增益 / 换轨迹速度都要重测。
POS_LEAD_TIME = 0.87  # s，位置设定点沿速度方向超前
YAW_LEAD_TIME = 0.30  # s，**预测视线**的视界：atan2((目标+v·τ)−(追踪机+v_c·τ))

CHASER_SPEED_LIMIT = 3.0  # m/s
# ⚠️ 这个值不能照搬 docs/TRACKING_PLAN.md 的 2.0（"别超过目标的 1.05"）——
# 那条推理是错的：追踪机在半径 4.0 上，跟上目标 30°/s 的角速度需要
# 线速度 4.0 * (2π/12) ≈ 2.10 m/s，比目标本身（半径 2.0 -> 1.05 m/s）快一倍。
# 限成 2.0 会让追踪机永远追不上，实测角向滞后 43°、两机距离从 2.0 涨到 2.71。
# 3.0 留 43% 余量给暂态。

# ---- 圆柱固定布局 ----
# 布局数据在 `configs/arena_20x20.yaml` 的 `cylinders:` 一节（唯一来源），
# 文件开头已读进 CYLINDER_XY / CYLINDER_SIZE / N_CYLINDERS。
# 20x20 场地：薄墙已经在半径 5.5~6.4 围了一圈，圆柱放到更外圈当散点障碍，
# 不挡中心的飞行空间（半径 8.5，离墙环最近 3.0 m）。


def pin_static_layout(env_manager, device=None):
    """把墙、圆柱、目标按固定布局摆好。

    为什么必须这样做：`asset_loader.py:87` 把 min/max_state_ratio 存在**资产类**
    上，同一个类的所有实例共用同一对范围 —— 6 根圆柱共用一个类，配置本身做不到
    "6 个固定的不同位置"（min==max 会让 6 根叠在同一个点）。所以只能在 reset 之后
    覆写状态张量。

    墙是一面一个类、min==max，reset 本身就摆对了；这里再写一次是保险
    （也顺带把速度清零）。

    必须在**每次** reset 之后调用：`asset_manager.reset_idx()` 会把所有资产
    位置重新随机采样一遍，覆写会被冲掉。reset 发生在
    `env_manager.reset()` 和 `env_manager.reset_terminated_and_truncated_envs()` 里。

    顺带把线速度/角速度清零 —— reset 不动速度，上一段遗留的速度会让物体漂走。
    """
    gtd = env_manager.global_tensor_dict
    state = gtd["env_asset_state_tensor"]  # (num_envs, num_assets, 13)
    if device is None:
        device = state.device

    # ---- 圆柱 ----
    n = len(CYLINDER_XY)
    pos = torch.zeros((n, 3), device=device)
    for i, (x, y) in enumerate(CYLINDER_XY):
        # 圆柱 URDF 的原点在底面，z=0 就是立在地面上
        pos[i, 0] = x
        pos[i, 1] = y
        pos[i, 2] = 0.0

    sl = slice(CYLINDER_SLOT_START, CYLINDER_SLOT_END)
    state[:, sl, 0:3] = pos.unsqueeze(0)
    # 单位四元数。aerial_gym 用的是 (x, y, z, w) 顺序，
    # 见 asset_manager.reset_idx 里 quat_from_euler_xyz_tensor 写进 3:7
    state[:, sl, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    state[:, sl, 7:13] = 0.0

    # ---- 薄墙 ----
    for i, w in enumerate(WALL_LAYOUT):
        s = wall_slot(i)
        x, y = w["pos"]
        yaw = float(w["yaw"])
        # 墙 URDF 的原点也在底面（make_wall_assets.py 统一的约定）-> z=0
        state[:, s, 0] = x
        state[:, s, 1] = y
        state[:, s, 2] = 0.0
        # (x, y, z, w) = (0, 0, sin(yaw/2), cos(yaw/2))
        state[:, s, 3] = 0.0
        state[:, s, 4] = 0.0
        state[:, s, 5] = math.sin(yaw / 2.0)
        state[:, s, 6] = math.cos(yaw / 2.0)
        state[:, s, 7:13] = 0.0

    # ---- 目标钉在轨迹起点 ----
    # 配置里 min==max 已经保证了，这里再写一次是为了让
    # "起点 = CircleTrajectory(t=0)" 这件事显式成立。
    traj = CircleTrajectory()
    tpos, _, _ = traj.state_at(0.0)
    state[:, TARGET_SLOT, 0:3] = torch.tensor(tpos, device=device)
    state[:, TARGET_SLOT, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    state[:, TARGET_SLOT, 7:13] = 0.0

    # 覆写完必须写回 IsaacGym，否则物理里还是旧位置
    if hasattr(env_manager.IGE_env, "write_to_sim"):
        env_manager.IGE_env.write_to_sim()
    return state


# 旧名字，阶段 1 的脚本还在用。新代码请用 pin_static_layout。
pin_cylinder_layout = pin_static_layout


class CircleTrajectory:
    """t -> (pos, vel, yaw)。半径 2 m 的圆，周期 12 s，高度固定 2 m。

    yaw 朝运动方向（切线方向），这样目标的"机头"也对得上它飞的方向。
    """

    def __init__(
        self,
        radius=TARGET_RADIUS,
        period=TARGET_PERIOD,
        height=TARGET_HEIGHT,
        center=(0.0, 0.0),
    ):
        self.radius = radius
        self.omega = 2.0 * math.pi / period
        self.height = height
        self.cx, self.cy = center

    def state_at(self, t):
        w = self.omega
        r = self.radius
        pos = (
            self.cx + r * math.cos(w * t),
            self.cy + r * math.sin(w * t),
            self.height,
        )
        vel = (
            -r * w * math.sin(w * t),
            r * w * math.cos(w * t),
            0.0,
        )
        yaw = w * t + math.pi / 2.0  # 朝运动方向
        return pos, vel, yaw


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def chaser_follow(target_pos, target_vel, chaser_pos):
    """【旧几何，留作对照】停在目标**径向外侧** STANDOFF 米，机头指向目标。

    返回 (desired_pos, yaw_cmd)。desired_pos 是位置设定点，直接喂位置控制器。

    ⚠️ 这是阶段 0/1（圆轨迹 + 追踪机绕场地原点公转）那套几何专用的，
    里面的偏航减号**只在那个几何下成立**。docs/BENCH_PLAN.md 阶段 2 起跟随方位改成
    "沿目标速度反方向退"，用 `chaser_follow_behind`，别用这个。

    ⚠️ "外侧"的方向必须取**从场地原点指向目标的径向**，不能取
    `normalize(追踪机 - 目标)`。第一版用了后者，结果是：追踪机一旦跑到原点
    另一侧，"当前方向"就指向圆内，它就被锁死在内圈绕（实测轨迹半径 0.5~0.9m），
    距离倒是对的（2.55m），但相机只能从很别扭的角度看目标，可见率掉到 44%。
    径向外侧的写法正好落在 docs/TRACKING_PLAN.md §4.3 算好的半径 4.5m 扫掠盘上。
    """
    # 位置设定点往前推 POS_LEAD_TIME，抵掉位置环的匀速滞后
    # （`dt` 那一步太小，补不了 25° 的角向落后）
    p_pred = [
        target_pos[0] + target_vel[0] * POS_LEAD_TIME,
        target_pos[1] + target_vel[1] * POS_LEAD_TIME,
        target_pos[2] + target_vel[2] * POS_LEAD_TIME,
    ]

    # 径向朝外。目标离原点太近时方向没定义，退回"目标->追踪机"方向。
    out = [p_pred[0], p_pred[1]]  # 原点在 (0,0)
    norm = math.hypot(out[0], out[1])
    if norm < 0.5:
        out = [chaser_pos[0] - p_pred[0], chaser_pos[1] - p_pred[1]]
        norm = math.hypot(out[0], out[1])
        if norm < 1e-6:
            out = [1.0, 0.0]
            norm = 1.0
    out = [out[0] / norm, out[1] / norm]

    desired = [
        p_pred[0] + STANDOFF * out[0],
        p_pred[1] + STANDOFF * out[1],
        _clamp(p_pred[2] + CHASER_Z_OFFSET, CHASER_Z_MIN, CHASER_Z_MAX),
    ]
    desired[0] = _clamp(desired[0], -CHASER_XY_LIMIT, CHASER_XY_LIMIT)
    desired[1] = _clamp(desired[1], -CHASER_XY_LIMIT, CHASER_XY_LIMIT)

    # 机头指向目标。瞄准点往目标**身后**挪 YAW_LEAD_TIME，抵消偏航环的滞后。
    # ⚠️ 这个减号是**旧几何**（径向外侧）实测出来的，挪到新几何下会变有害
    # （8.53° vs 零超前 4.80°）—— 新几何看 `chaser_follow_behind`。
    # 相机朝机体正前方，不指向目标几步就出画。
    p_yaw = [
        target_pos[0] - target_vel[0] * YAW_LEAD_TIME,
        target_pos[1] - target_vel[1] * YAW_LEAD_TIME,
    ]
    yaw_cmd = math.atan2(p_yaw[1] - chaser_pos[1], p_yaw[0] - chaser_pos[0])
    return desired, yaw_cmd


def chaser_follow_behind(target_pos, target_vel, chaser_pos, chaser_vel=None,
                         standoff=STANDOFF, pos_lead=POS_LEAD_TIME,
                         yaw_lead=0.0):
    """跟在目标**身后** `standoff` 米（沿速度反方向），机头指向目标。

    这是 docs/BENCH_PLAN.md §3 P4 要的跟随方位。和 `chaser_follow`（径向外侧）的差别
    全在几何上，前馈符号因此要重推。**结论已用 check_follow.py 三轮对照实测
    定稿（2026-10-12），数字见 POS_LEAD_TIME 上方那段注释。**

    **位置前馈 `pos_lead` 的几何是独立的。** 它抵的是位置环跟**匀速设定点**的
    稳态滞后 `e = (Kd/Kp)·v`，和目标在哪、跟在哪一侧都无关。设定点先沿 `+v` 推
    `pos_lead` 再沿 `-v̂` 退 `standoff`，控制器再落后 `|v|·pos_lead`，落完
    正好停在 `target − standoff·v̂`。实测最优 **0.87 s**（`沿 v̂` = −1.997，
    离理想 −2.00 只差 3 mm）。

    **偏航那路：旧几何的"减号"在新几何下有害**（8.53°，比零超前的 4.80° 还差），
    定稿形式是**预测视线**：

        yaw_cmd = atan2( (target + v·τ) − (chaser + v_c·τ) )

    实测 τ 的扫描是 U 形，谷底 **τ = 0.30 s → 0.22°**（零超前是 4.80°）。

    ⚠️ 要拿到这个效果**必须传 `chaser_vel`**，并且 `yaw_lead = YAW_LEAD_TIME`。
    不传 `chaser_vel` 时右边退化成"沿 +v 挪瞄准点"，那是**另一条路**，实测
    （+0.50 s）只有 2.44°，明显不如预测视线；`yaw_lead=0` 则是零超前基线 4.80°。
    所以这里的默认值故意留成 0 —— 宁可退回测过的基线，也不静默走一条没测过的路。
    """
    v = [float(target_vel[0]), float(target_vel[1]), float(target_vel[2])]
    speed_xy = math.hypot(v[0], v[1])
    v_hat = [1.0, 0.0] if speed_xy < 1e-6 else [v[0] / speed_xy, v[1] / speed_xy]

    desired = [
        target_pos[0] + v[0] * pos_lead - standoff * v_hat[0],
        target_pos[1] + v[1] * pos_lead - standoff * v_hat[1],
        _clamp(target_pos[2] + CHASER_Z_OFFSET, CHASER_Z_MIN, CHASER_Z_MAX),
    ]
    desired[0] = _clamp(desired[0], -CHASER_XY_LIMIT, CHASER_XY_LIMIT)
    desired[1] = _clamp(desired[1], -CHASER_XY_LIMIT, CHASER_XY_LIMIT)

    ax = target_pos[0] + v[0] * yaw_lead
    ay = target_pos[1] + v[1] * yaw_lead
    if chaser_vel is not None:
        ax -= float(chaser_vel[0]) * yaw_lead
        ay -= float(chaser_vel[1]) * yaw_lead
    yaw_cmd = math.atan2(ay - chaser_pos[1], ax - chaser_pos[0])
    return desired, yaw_cmd


def limit_speed(prev_pos, desired_pos, dt, limit=CHASER_SPEED_LIMIT):
    """把位置设定点的变化率限幅到 limit（m/s），防止追踪机猛冲超调。"""
    dx = desired_pos[0] - prev_pos[0]
    dy = desired_pos[1] - prev_pos[1]
    dz = desired_pos[2] - prev_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_step = limit * dt
    if dist <= max_step or dist < 1e-9:
        return list(desired_pos)
    s = max_step / dist
    return [prev_pos[0] + dx * s, prev_pos[1] + dy * s, prev_pos[2] + dz * s]


# --------------------------------------------------------------------------- #
#  避障几何（BaselineTracker 用；纯数学）
# --------------------------------------------------------------------------- #
#
# 为什么 baseline 需要避障：`chaser_follow_behind` 的期望点是
# `target + v·pos_lead − standoff·v̂`，目标自己在自由空间里飞不代表这个偏移点
# 也在自由空间里 —— 实测 600 步里有 3 步期望点整个落进圆柱。目标是运动学驱动的、
# 穿过去没事，追踪机是物理体，撞上就判失败。
#
# 这里只做**设定点级**的几何避障（不做路径规划），够让 baseline 当参考实现用。
# 更好的避障是各家跟踪器自己的事 —— 那正是 benchmark 要比的东西。

def obstacle_hit(p, obstacles, margin=0.0):
    """点 p 是否落进某个障碍的外扩 footprint。返回命中障碍的下标列表。

    `obstacles` 是 `[{pos, size, yaw}, ...]`（`tracker_api.Obstacle` 的字典形式）。
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    hits = []
    for i, w in enumerate(obstacles):
        sx, sy, sz = w["size"]
        yaw = float(w["yaw"])
        dx, dy = x - float(w["pos"][0]), y - float(w["pos"][1])
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        if (abs(lx) <= sx / 2.0 + margin
                and abs(ly) <= sy / 2.0 + margin
                and -margin <= z <= sz + margin):
            hits.append(i)
    return hits


def segment_clear(a, b, obstacles, margin=0.0, step=0.10):
    """线段 a→b 是否全程在障碍外扩 margin 之外。按 step 采样。"""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    dx = float(b[0]) - ax
    dy = float(b[1]) - ay
    dz = float(b[2]) - az
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    n = max(2, int(math.ceil(dist / step)) + 1)
    for k in range(n + 1):
        t = k / float(n)
        if obstacle_hit((ax + dx * t, ay + dy * t, az + dz * t), obstacles, margin):
            return False
    return True


# 绕追踪机旋转的搜索角（度）。先小后大，找到第一个自由且可达的就用。
AVOID_SEARCH_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90,
                    120, -120, 150, -150, 180)


def avoid_setpoint(chaser, desired, obstacles, margin=0.30):
    """在 desired 附近找一个自由、且从 chaser 直线可达的设定点。

    顺序：desired 本身 → 绕 chaser 在**水平面**里转（距离不变）→ 全挡死返回 None。
    返回 None 时调用方应该原地不动（但偏航照旧指向目标 —— 至少别跟丢）。

    转水平面就够了：z 是 `target_z + CHASER_Z_OFFSET`，和水平通道解耦
    （见 CHASER_Z_OFFSET 的注释），改 z 反而会破坏跟随几何。
    """
    if not obstacle_hit(desired, obstacles, margin) and \
            segment_clear(chaser, desired, obstacles, margin):
        return list(desired)

    cx, cy = float(chaser[0]), float(chaser[1])
    ux = float(desired[0]) - cx
    uy = float(desired[1]) - cy
    r = math.hypot(ux, uy)
    if r < 1e-6:
        return None
    for deg in AVOID_SEARCH_DEG:
        if deg == 0:
            continue
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        cand = [cx + ux * c - uy * s, cy + ux * s + uy * c, float(desired[2])]
        if not obstacle_hit(cand, obstacles, margin) and \
                segment_clear(chaser, cand, obstacles, margin):
            return cand
    return None
