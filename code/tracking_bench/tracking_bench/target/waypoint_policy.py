#!/usr/bin/env python3
"""目标运动模型：随机路点 + 分段五次多项式平滑轨迹（docs/BENCH_PLAN.md §3 P2）。

三层结构，各管各的：

    L0 路点层   按 seed 生成 P0..Pn，自由空间拒绝采样
    L1 轨迹层   分段五次多项式 Pi -> Pi+1，p/v 连续、两端 a=0
    L2 采样层   state_at(t) -> (pos, vel, yaw)

### 为什么是五次多项式 + 零加速度边界

边界条件

    p(0)=Pi,   v(0)=vi,   a(0)=0
    p(1)=Pi+1, v(1)=vi+1, a(1)=0

里 `v` 连续 ⇒ **目标在路点上不停**（需求：连续飞行），`a=0` ⇒ 段间没有
加速度阶跃。切向速度取 Catmull-Rom 式 `vi = v_cruise · normalize(P_{i+1} − P_{i−1})`，
速度剖面比"两端减速到 0"的最小-jerk 更接近匀速巡航。

### ⚠️ 段是 append-only 的，定稿就不改

Catmull-Rom 的 `vi` 要看**前后两个**邻点，所以新路点天然会改上一段的末端切向。
第一版每加一个路点就整体 `_rebuild()`，结果是播放途中 `state_at(t)` 会变：
仿真里目标明明精确写在 `state_at` 给的位置上（偏差 6e-7 m），却会突然跳
0.50 m（4.95 m/s）、还会跳进墙里 —— 因为事后重算出来的轨迹和当时飞的那条
已经对不上了。

现在的做法：**段一旦定稿就锁死**。
  · 段的起始切向 = 上一段的末端切向 ⇒ v 天然连续
  · 段的末端切向 = Catmull-Rom，而它要等下一个路点出现才算得出
  · 所以段表比路点表**滞后一段**：`len(segments) == len(pts) − 2`
只有 `set_waypoints`（用户把路点给全了）才用后向差分收尾，走 `_flush_tail`。

后果是 `state_at(t)` 对已经查过的时间**永远不变**，"无瞬移"才有意义。
回归检查见 `check_p2.py` 的 1e。

归一化时间 `τ = t/Ti ∈ [0,1]` 下（`p` 按分量算，`d = P_{i+1} − Pi`）：

    a0 = Pi
    a1 = vi · Ti
    a2 = 0
    a3 = 10·d − (6·vi + 4·v_{i+1})·Ti
    a4 = −15·d + (8·vi + 7·v_{i+1})·Ti
    a5 = 6·d − (3·vi + 3·v_{i+1})·Ti

（代入 τ=0/1 核对过：p(0)=Pi、p(1)=P_{i+1}、dp/dt(0)=vi、dp/dt(1)=v_{i+1}、
d²p/dt²(0)=d²p/dt²(1)=0。`check_p2.py` 里还有数值差分验收。）

⚠️ `a1 = vi · Ti` 不是 `vi` —— 多项式是对 **τ** 求导，真实速度要再除一个 `Ti`。
   写漏这一步轨迹会快/慢 `Ti` 倍，且 v 不连续。`eval_quintic` 把这件事做掉了。

### 段时长

初值是 §3 P2 给的 `Ti = max(T_min, |Δp|/v_max, sqrt(2·|Δp|/a_max))`。
**光有这个不够**：五次曲线在弯道上会往弦外鼓，真实峰值速度/加速度高于
`|Δp|/Ti` 这个匀速估计（弦和切向不共线时更明显）。所以 `_make_segment`
拿初值起跑之后，按**实测峰值**把 `Ti` 放大到满足 `v_max` / `a_max`，
否则 `峰值 ≤ v_max` 这条验收是**静默**超速的。

### 自由空间检查

路点和**实际飞出来的五次曲线**都要查 —— 只查折线不够。`|vi·Ti| = v_cruise·Ti ≥ |Δp|`
意味着 Hermite 切向**不短于弦长**，弯道上曲线会往弦外鼓。实测（L=4 m 段）：

    拐角  0°   20°   40°   57°   75°   90°  120°  150°  180°
    鼓肚  0.00  0.03  0.07  0.09  0.12  0.14  0.17  0.19  0.20  × |Δp|

**鼓肚有上界（约 0.2·|Δp|）**，所以不会发散，但对 8 m 的段就是 1.6 m，
远超 `wall_clearance = 0.5`，照样能从墙里穿过去。`_append_point_checked`
因此在生成时就查曲线，不过就换一个路点重采。

路点采样用**锥形采样**：先在当前航向 ±57° 的锥里取（拐弯小、鼓肚小），
采不到再放宽到 ±114°，最后全向。不写死"拐弯角上限"是因为随机游走会把
序列带进场地角落，硬限角之后从角落里采不出下一个点、直接卡死。

### 用法

    from tracking_bench.target.waypoint_policy import WaypointPolicy
    pol = WaypointPolicy(seed=20261012)
    pos, vel, yaw = pol.state_at(t)

纯几何，**不 import aerial_gym / isaacgym**，布局从 configs/arena_20x20.yaml 读。
"""

import math
import os

import numpy as np
import yaml

# 路径经 paths.find_config 统一解析（打包后模块在 target/ 子包里，按 `__file__`
# 直接拼 `configs/` 会拼到 target/configs/，踩过）。
from tracking_bench.paths import find_config as _find_config

_CFG_PATH = _find_config("arena_20x20.yaml")

# ---- 默认参数（§2.5 target_policy；可被 __init__ 覆盖）----
V_MAX = 1.2          # m/s，段内峰值速度上限
A_MAX = 1.5          # m/s^2，段内峰值加速度上限
T_MIN = 1.0          # s，段时长下限
MIN_SEGMENT = 2.5    # m，相邻路点最小间距（太近会抖）
MAX_SEGMENT = 8.0    # m，最大间距（太远目标会一次横穿全场）
MAX_TURN = 1.0       # rad（≈57°），下一节**首选**的拐弯半角，卡住会自动放宽
SAMPLE_STEP = 0.05   # m，自由空间检查的采样步长
CURVE_CLEARANCE = 0.05  # m，**曲线**检查用的净距，见下（别和 wall_clearance 混）


# ---------------------------------------------------------------------------
# 五次多项式
# ---------------------------------------------------------------------------
def quintic_coeffs(p0, p1, v0, v1, T):
    """返回 (a0..a5)，p(τ) = Σ ak·τ^k，τ = t/T ∈ [0,1]。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    d = p1 - p0
    a0 = p0
    a1 = v0 * T
    a2 = np.zeros(3)
    a3 = 10.0 * d - (6.0 * v0 + 4.0 * v1) * T
    a4 = -15.0 * d + (8.0 * v0 + 7.0 * v1) * T
    a5 = 6.0 * d - (3.0 * v0 + 3.0 * v1) * T
    return (a0, a1, a2, a3, a4, a5)


def eval_quintic(coeffs, tau, T):
    """τ -> (pos, vel, acc)。vel/acc 是对**真实时间 t** 的导数。"""
    a0, a1, a2, a3, a4, a5 = coeffs
    pos = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
    dp_dtau = a1 + 2 * a2 * tau + 3 * a3 * tau**2 + 4 * a4 * tau**3 + 5 * a5 * tau**4
    d2p_dtau2 = 2 * a2 + 6 * a3 * tau + 12 * a4 * tau**2 + 20 * a5 * tau**3
    return pos, dp_dtau / T, d2p_dtau2 / (T * T)


def peak_speed_accel(coeffs, T, n=201):
    """沿段采样求峰值速度 / 峰值加速度（含两端点）。"""
    pv = 0.0
    pa = 0.0
    for tau in np.linspace(0.0, 1.0, n):
        _, v, a = eval_quintic(coeffs, tau, T)
        pv = max(pv, float(np.linalg.norm(v)))
        pa = max(pa, float(np.linalg.norm(a)))
    return pv, pa


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
def point_clear(p, walls, wall_clearance, xy_min, xy_max, z_range,
                boundary_margin):
    """点 p 是否在自由空间（不在任何扩边墙内，也在场地内）。"""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if not (xy_min[0] + boundary_margin <= x <= xy_max[0] - boundary_margin):
        return False
    if not (xy_min[1] + boundary_margin <= y <= xy_max[1] - boundary_margin):
        return False
    if not (z_range[0] <= z <= z_range[1]):
        return False
    for w in walls:
        sx, sy, sz = w["size"]
        yaw = float(w["yaw"])
        dx = x - w["pos"][0]
        dy = y - w["pos"][1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        if (abs(lx) <= sx / 2.0 + wall_clearance
                and abs(ly) <= sy / 2.0 + wall_clearance
                and -wall_clearance <= z <= sz + wall_clearance):
            return False
    return True


def curve_clear(coeffs, T, clear_fn, step=SAMPLE_STEP):
    """沿五次曲线按 step 采样，逐点做自由空间检查。"""
    pts = [eval_quintic(coeffs, tau, T)[0] for tau in np.linspace(0.0, 1.0, 64)]
    length = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
    n = max(16, int(math.ceil(length / step)) + 1)
    for tau in np.linspace(0.0, 1.0, n):
        if not clear_fn(eval_quintic(coeffs, tau, T)[0]):
            return False
    return True


# ---------------------------------------------------------------------------
# WaypointPolicy
# ---------------------------------------------------------------------------
class WaypointPolicy:
    """随机路点 + 五次多项式轨迹。实现 docs/BENCH_PLAN.md §2.4 的 TargetPolicy 接口。"""

    def __init__(self, seed=None, arena_yaml=_CFG_PATH,
                 v_cruise=V_MAX, v_max=V_MAX, a_max=A_MAX, t_min=T_MIN,
                 min_segment=MIN_SEGMENT, max_segment=MAX_SEGMENT,
                 max_turn=MAX_TURN, n_waypoints=6, speed_scale=1.0):
        # ⚠️ 必须显式 encoding：`import isaacgym` 之后进程的
        #    locale.getpreferredencoding() 会变成 ASCII，YAML 里的中文注释
        #    会直接 UnicodeDecodeError（踩过：'ascii' codec can't decode 0xc3）。
        with open(arena_yaml, encoding="utf-8") as f:
            arena = yaml.safe_load(f)
        self._arena = arena
        # **所有**要躲的障碍都进自由空间检查：墙 + 圆柱。
        # ⚠️ 只查 `arena["walls"]` 是踩过的坑 —— 圆柱不查，路点能采到柱面上
        #    （实测离柱心 0.46 m，柱面只剩 0.16 m），目标贴着飞过去、
        #    追踪机跟在后面撞上去（check_p4 全量 baseline 第 161 步）。
        self.obstacles = list(arena["walls"])
        cyl = arena.get("cylinders", {}) or {}
        csz = list(cyl.get("size", (0.6, 0.6, 3.0)))
        for p in cyl.get("positions", []):
            self.obstacles.append({"size": csz, "pos": list(p), "yaw": 0.0})
        self.walls = self.obstacles  # 旧名字，外部可能还在用
        wp = arena.get("waypoints", {})
        self.xy_min = tuple(wp.get("xy_min", (-9.0, -9.0)))
        self.xy_max = tuple(wp.get("xy_max", (9.0, 9.0)))
        self.z_range = tuple(wp.get("z_range", (1.5, 4.0)))
        self.wall_clearance = float(wp.get("wall_clearance", 0.5))
        self.boundary_margin = float(wp.get("boundary_margin", 0.5))
        self.curve_clearance = float(wp.get("curve_clearance", CURVE_CLEARANCE))

        self.v_cruise = float(v_cruise)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.t_min = float(t_min)
        self.min_segment = float(min_segment)
        self.max_segment = float(max_segment)
        self.max_turn = float(max_turn)
        self.n_waypoints = int(n_waypoints)
        self.speed_scale = float(speed_scale)

        self._manual_state = None
        self._frozen_t = 0.0
        self._last_t = 0.0
        self._paused = False
        self._cache_i = 0
        self._pts = []
        self._segments = []
        self._cum_t = [0.0]
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.reset(None)

    # ------------------------------------------------------------------ 接口
    def reset(self, seed=None):
        """重新生成路点序列。传 seed 就重开随机流，不传就接着上次的走。"""
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        self._manual_state = None
        self._paused = False
        self._frozen_t = 0.0
        self._last_t = 0.0
        self._pts = [self._sample_point(None)]
        self._segments = []
        self._cum_t = [0.0]
        self._cache_i = 0
        self._extend(self.n_waypoints)

    @property
    def waypoints(self):
        """当前路点序列（含还没飞到的），供跟踪器前瞻。"""
        return [tuple(float(v) for v in p) for p in self._pts]

    @property
    def duration(self):
        """已生成轨迹的总时长。往后调用 state_at 会自动续段。"""
        return self._cum_t[-1]

    def set_waypoints(self, waypoints):
        """手动指定路点（脚本化场景）。路点、折线、**曲线**都要过自由空间检查。"""
        pts = [np.asarray(p, dtype=np.float64) for p in waypoints]
        if len(pts) < 2:
            raise ValueError("至少要 2 个路点")
        for p in pts:
            if not self._point_ok(p):
                raise ValueError(f"路点 {tuple(float(v) for v in p)} 不在自由空间里")
        for a, b in zip(pts[:-1], pts[1:]):
            if not self._segment_ok(a, b):
                raise ValueError(f"线段 {tuple(a)} -> {tuple(b)} 穿墙")
        old = (list(self._pts), list(self._segments), list(self._cum_t), self._cache_i)
        try:
            self._pts = pts
            self._segments = []
            self._cum_t = [0.0]
            self._cache_i = 0
            # 用户给全了 -> 末段用后向差分收尾（见 _flush_tail）
            segs = self._flush_tail()
        except Exception:
            (self._pts, self._segments, self._cum_t, self._cache_i) = old
            raise
        bad = [i for i, s in enumerate(segs) if not self._curve_ok(s)]
        if bad:
            (self._pts, self._segments, self._cum_t, self._cache_i) = old
            raise ValueError(
                f"第 {bad} 段的五次曲线穿墙（折线本身是通的，是弯道鼓肚造成的）。"
                f"把拐弯处的路点挪得缓一点，或调小 v_cruise"
            )
        self._manual_state = None

    def set_state(self, pos, vel=(0.0, 0.0, 0.0)):
        """直接钉住状态（调试用）。要恢复轨迹调 `clear_state()`。"""
        self._manual_state = (
            tuple(float(v) for v in pos), tuple(float(v) for v in vel), 0.0
        )

    def clear_state(self):
        self._manual_state = None

    def pause(self):
        """冻结在最后一次 state_at 的时刻（一次都没查过就冻在 t=0）。"""
        self._frozen_t = self._last_t
        self._paused = True

    def resume(self):
        self._paused = False

    def set_speed_scale(self, k):
        """整体调速，k>1 加速。重建轨迹（段时长按 1/k 缩）。

        ⚠️ 重建会改**已定稿**的段（T 变了五次曲线的鼓肚也变），所以只在开跑前调。
        """
        if k <= 0:
            raise ValueError("speed_scale 必须 > 0")
        old = self.speed_scale
        self.speed_scale = float(k)
        try:
            self._rebuild()
        except Exception:
            self.speed_scale = old
            raise

    def state_at(self, t):
        """t -> (pos, vel, yaw)。查询，不推进内部时钟。"""
        t = float(t)
        self._last_t = t
        if self._manual_state is not None:
            return self._manual_state
        if self._paused:
            t = self._frozen_t
        if t <= 0.0:
            t = 0.0
        # 轨迹不够长就往后接（保持目标"不停"，不掉进"停在最后一个路点"）
        while t > self._cum_t[-1] - 1e-9:
            self._extend(1)
        t = min(t, self._cum_t[-1])

        i = self._seg_index_at(t)
        seg = self._segments[i]
        tau = min(max((t - self._cum_t[i]) / seg.T, 0.0), 1.0)
        pos, vel, _ = eval_quintic(seg.coeffs, tau, seg.T)
        if abs(vel[0]) + abs(vel[1]) > 1e-9:
            yaw = math.atan2(vel[1], vel[0])
        else:
            yaw = seg.yaw_end
        return (tuple(float(v) for v in pos),
                tuple(float(v) for v in vel),
                float(yaw))

    # ------------------------------------------------------------------ 生成
    def _point_ok(self, p):
        return point_clear(p, self.walls, self.wall_clearance,
                           self.xy_min, self.xy_max, self.z_range,
                           self.boundary_margin)

    def _segment_ok(self, a, b):
        """折线 a->b 不能穿墙（SAMPLE_STEP 采样）。"""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / SAMPLE_STEP)) + 1)
        for s in np.linspace(0.0, 1.0, n):
            if not self._point_ok(a + s * (b - a)):
                return False
        return True

    def _curve_ok(self, seg):
        """一段五次曲线不能**进墙/出界** —— 比折线严格，但净距比路点松。

        两种要求别混（混了就采不出点，踩过）：
          · 路点 + 折线：`wall_clearance`（0.5 m）—— docs/BENCH_PLAN.md §3 P2 的原话
          · 飞出来的曲线：只求**不进墙体**，用 `curve_clearance`（0.05 m）

        因为鼓肚是往弦的**内侧**去的，而弦本身已经离墙 0.5 m 了；
        曲线只要求不越过那 0.5 m 缓冲的**里圈**。用同一个 0.5 m 去卡曲线，
        实测在场地角落 300 次重采全灭（上一段被新点改了切向，鼓肚 0.5~1.1 m）。
        """
        return curve_clear(seg.coeffs, seg.T, self._curve_point_ok)

    def _curve_point_ok(self, p):
        """曲线上的点：不进墙（净距 curve_clearance），但允许进到**真实场地边界**。

        路点被 `xy_min/xy_max`（±9）+ `boundary_margin` 圈在 ±8.5 里，
        曲线鼓肚可能鼓到 ±9 之外 —— 那仍然是场地内（±10），
        不该算出界。所以这里拿 `arena.bounds_*` 当界。
        """
        bmin = self._arena["arena"]["bounds_min"]
        bmax = self._arena["arena"]["bounds_max"]
        return point_clear(p, self.walls, self.curve_clearance,
                           (bmin[0], bmin[1]), (bmax[0], bmax[1]),
                           (bmin[2], bmax[2]), 0.0)

    def _dir_in_cone(self, heading, half_angle):
        """在以 heading 为轴、半角 half_angle 的锥里**均匀**取一个单位方向。"""
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(e1, heading))) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - float(np.dot(e1, heading)) * heading
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(heading, e1)
        az = float(self._rng.uniform(0.0, 2.0 * math.pi))
        # sqrt 让方向在锥**体内**均匀，而不是都挤在轴上
        alpha = half_angle * math.sqrt(float(self._rng.uniform(0.0, 1.0)))
        return (math.cos(alpha) * heading
                + math.sin(alpha) * (math.cos(az) * e1 + math.sin(az) * e2))

    def _sample_point(self, prev):
        """采下一个路点：优先小拐弯，采不到再逐级放宽锥角，最后全向。

        为什么不写死"拐弯角 ≤ N°"：随机游走会把路点序列带进场地角落，
        硬限角之后从角落里**采不出**下一个点，直接卡死（实测 max_turn=57°
        时在 (8.5, 4.9) 这种靠边位置 4000 次全灭）。锥形采样按"最想要的先试"
        排序，既保住了小拐弯的偏好，又不会死锁。

        鼓肚本身是有界的：拐 180° 也只往弦外鼓 0.2·|Δp|（见模块头），
        所以真正兜底的是 `_append_point_checked` 的曲线检查。
        """
        prev2 = self._pts[-2] if len(self._pts) >= 2 else None
        heading = None
        if prev is not None and prev2 is not None:
            d = np.asarray(prev, dtype=np.float64) - np.asarray(prev2, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                heading = d / n

        cones = [self.max_turn, 2.0 * self.max_turn, math.pi] \
            if heading is not None else [math.pi]
        tries_each = 2500 if heading is not None else 8000
        for cone in cones:
            for _ in range(tries_each):
                if prev is None:
                    lo = np.array([self.xy_min[0], self.xy_min[1], self.z_range[0]])
                    hi = np.array([self.xy_max[0], self.xy_max[1], self.z_range[1]])
                    p = self._rng.uniform(lo, hi)
                else:
                    d = float(self._rng.uniform(self.min_segment, self.max_segment))
                    v = self._dir_in_cone(heading, cone) if heading is not None \
                        else self._dir_in_cone(np.array([1.0, 0.0, 0.0]), math.pi)
                    p = np.asarray(prev, dtype=np.float64) + d * v
                if not self._point_ok(p):
                    continue
                if prev is not None:
                    if not self._segment_ok(prev, p):
                        continue
                return p
        raise RuntimeError(
            f"锥形采样试完都没找到下一个路点（prev={prev}）。"
            f"场地是不是被墙堵死了？或 min_segment={self.min_segment} / "
            f"max_segment={self.max_segment} 不合适"
        )

    def _extend(self, n):
        for _ in range(n):
            self._append_point_checked()

    def _tangent(self, i):
        """pts[i] 处的切向（模长 v_cruise）。

        首尾用前向/后向差分，中间用 Catmull-Rom `normalize(P_{i+1} − P_{i-1})` ——
        就是设计文档写的那条。⚠️ 只在 `i` 是**内部点**时才拿到 Catmull-Rom；
        末点的切向要等下一个路点出现才定得下来，见 `_finalize_next`。
        """
        n = len(self._pts)
        if i <= 0:
            d = self._pts[1] - self._pts[0]
        elif i >= n - 1:
            d = self._pts[-1] - self._pts[-2]
        else:
            d = self._pts[i + 1] - self._pts[i - 1]
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            d = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        return self.v_cruise * d / norm

    def _finalize_next(self, allow_edge_tangent=False):
        """定稿**下一段** `pts[m] -> pts[m+1]`（`m = len(self._segments)`）。

        「定稿」= **从此不再改**（append-only）。这是"跑 600 步无瞬移"的前提：

        ⚠️ 踩过的坑：第一版每加一个路点就 `_rebuild()` 重算**全部**段的切向
        （Catmull-Rom 看前后邻点，新点会改上一段的末端切向），连段时长 `_cum_t`
        一起挪。于是播放中 `state_at(t)` 前后不一致 —— 仿真里目标明明精确写在
        `state_at` 给的位置上（偏差 6e-7 m），却会突然跳 0.50 m（4.95 m/s），
        还会跳进墙里。事后重算的轨迹和当时飞的那条对不上，看着像目标抽风。

        现在的规则：段的**起始**切向沿用上一段的末端切向（⇒ v 天然连续），
        **末端**切向用 Catmull-Rom。而 Catmull-Rom 要看 `pts[m+2]`，所以得等
        下一个路点出现才定稿 —— 这就是"路点表比段表多一项"
        （`len(segments) == len(pts) − 2`）的由来。

        `allow_edge_tangent=True` 时允许在没有 `pts[m+2]` 的情况下用**后向差分**
        定稿末段（`set_waypoints` 这种"用户已经给全了"的场合）。
        """
        m = len(self._segments)
        if m + 1 >= len(self._pts):
            return None                       # 连终点都还没有
        if not allow_edge_tangent and m + 2 >= len(self._pts):
            return None                       # pts[m+2] 没到，末端切向定不下来
        v0 = self._segments[-1].v1 if self._segments else self._tangent(0)
        v1 = self._tangent(m + 1)             # 有 pts[m+2] 就是 Catmull-Rom
        seg = self._make_segment(self._pts[m], self._pts[m + 1], v0, v1)
        self._segments.append(seg)
        self._cum_t.append(self._cum_t[-1] + seg.T)
        self._cache_i = 0
        return seg

    def _flush_tail(self):
        """把还没定稿的段补完（**路点表已经给全**时才用）。

        末段的末端切向退化成后向差分 —— 设计文档说的「首尾用前向/后向差分」。
        自动续段的路径**不走这里**，所以巡航中定稿的段全是 Catmull-Rom。
        """
        out = []
        while True:
            seg = self._finalize_next(allow_edge_tangent=True)
            if seg is None:
                return out
            out.append(seg)

    def _append_point_checked(self, max_tries=300):
        """追加一个路点，把**新定稿的那一段**曲线一起验；穿墙就换一个再试。

        为什么要连曲线一起验：切向速度 |vi| = v_cruise，而段时长 T ≥ |Δp|/v_max，
        于是 |vi·Ti| ≥ |Δp| —— Hermite 切向**不短于弦长**，弯道上五次曲线会往弦外
        鼓出去。只查折线的话，曲线照样能从墙里穿过去（check_p2 第一版实测
        13 段里 7 段穿墙，最大离墙净距 0.00 m）。

        只需要查**一段**：定稿是 append-only 的，早先那些段不会被新点改写
        （旧版要连末两段一起查，就是因为 `_rebuild` 会追溯改上一段）。
        """
        for _ in range(max_tries):
            p = self._sample_point(self._pts[-1])
            self._pts.append(p)
            seg = self._finalize_next()
            if seg is None or self._curve_ok(seg):
                return
            self._pts.pop()
            self._segments.pop()
            self._cum_t.pop()
        self._cache_i = 0
        raise RuntimeError(
            f"{max_tries} 次都没采到「曲线也不穿墙」的路点"
            f"（prev={tuple(float(v) for v in self._pts[-1])}）。"
            f"弯道鼓肚盖住了净距，试着调小 v_cruise 或调大 wall_clearance"
        )

    def _rebuild(self):
        """⚠️ 只给 `set_speed_scale` 这种**故意改轨迹**的调用用。

        会从 `self._pts` 重算全部段，所以已定稿的段也会变 —— 别在播放中途调。
        """
        old = (list(self._pts), list(self._segments), list(self._cum_t), self._cache_i)
        try:
            self._segments = []
            self._cum_t = [0.0]
            self._cache_i = 0
            segs = self._flush_tail()
        except Exception:
            (self._pts, self._segments, self._cum_t, self._cache_i) = old
            raise
        bad = [i for i, s in enumerate(segs) if not self._curve_ok(s)]
        if bad:
            (self._pts, self._segments, self._cum_t, self._cache_i) = old
            raise ValueError(
                f"调速后第 {bad} 段的五次曲线穿墙（T 变了鼓肚也变）。"
                f"换个小一点的 speed_scale，或调 v_cruise"
            )

    def _make_segment(self, p0, p1, v0, v1):
        dist = float(np.linalg.norm(p1 - p0))
        T = max(self.t_min,
                dist / self.v_max if self.v_max > 0 else 0.0,
                math.sqrt(2.0 * dist / self.a_max) if self.a_max > 0 else 0.0)
        T /= self.speed_scale

        # 初值只是估计，真实峰值要看曲线。按实测峰值把 T 放大到满足约束。
        v_lim = self.v_max * self.speed_scale
        a_lim = self.a_max * self.speed_scale * self.speed_scale
        for _ in range(60):
            coeffs = quintic_coeffs(p0, p1, v0, v1, T)
            pv, pa = peak_speed_accel(coeffs, T)
            need = 1.0
            if pv > v_lim:
                need = max(need, pv / v_lim)
            if pa > a_lim:
                need = max(need, math.sqrt(pa / a_lim))
            if need <= 1.001:
                break
            T *= min(need * 1.05, 3.0)
        else:
            raise RuntimeError(f"段时长放大 60 轮仍不满足约束（T={T:.2f}s）")

        d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
        yaw_end = math.atan2(d[1], d[0]) if abs(d[0]) + abs(d[1]) > 1e-9 else 0.0
        return _Segment(p0, p1, v0, v1, coeffs, T, yaw_end)

    def _seg_index_at(self, t):
        i = self._cache_i
        if not (0 <= i < len(self._segments)):
            i = 0
        while i > 0 and t < self._cum_t[i]:
            i -= 1
        while i < len(self._segments) - 1 and t >= self._cum_t[i + 1]:
            i += 1
        self._cache_i = i
        return i


class _Segment:
    __slots__ = ("p0", "p1", "v0", "v1", "coeffs", "T", "yaw_end")

    def __init__(self, p0, p1, v0, v1, coeffs, T, yaw_end):
        self.p0, self.p1, self.v0, self.v1 = p0, p1, v0, v1
        self.coeffs, self.T, self.yaw_end = coeffs, T, yaw_end
