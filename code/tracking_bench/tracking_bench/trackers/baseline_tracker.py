#!/usr/bin/env python3
"""P4：baseline 跟踪器 + 一个"什么都不做"的对照。

`BaselineTracker` 的跟随方位是 docs/BENCH_PLAN.md §3 P4 定的：**沿目标速度反方向退
`standoff` 米**，机头指向目标。控制律就是 `tracking_lib.chaser_follow_behind`，
那两个前馈超前量已经在 P2 末尾用 `check_follow.py` 三轮对照实测定稿：

    POS_LEAD_TIME = 0.87 s   （沿 v̂ 的稳态偏差 −1.997，理想 −2.000）
    YAW_LEAD_TIME = 0.30 s   （预测视线，偏航误差 0.22°）

换控制周期 / 换 Lee 增益 / 换轨迹速度都要重测那两个数（见 tracking_lib 注释）。

`DoNothingTracker` 是验收用的对照：**把起始位姿原样锁住**，不转向、不追。
它存在的意义是让 §2.6 的 `lost_timeout` 能被严格验到。
"""

import math

import numpy as np

from tracking_bench import tracking_lib as tl
from tracking_bench.trackers.tracker_api import Tracker, TrackerAction, XY_LIMIT, Z_MIN, Z_MAX


class BaselineTracker(Tracker):
    """沿目标速度反方向退 `standoff`，机头指向目标（预测视线）。

    在 `chaser_follow_behind` 之上加一层**设定点级**的几何避障（`tl.avoid_setpoint`）。
    为什么必须加：期望点是 `target + v·pos_lead − standoff·v̂`，目标自己在自由空间
    里飞不代表这个偏移点也在 —— 实测 600 步里有 3 步期望点整个落进圆柱。目标是
    运动学驱动的穿过去没事，追踪机是物理体，撞上就判失败（check_p4 全量的
    baseline 就是这么在第 161 步挂的）。
    """

    name = "baseline"

    def __init__(self, standoff=tl.STANDOFF, pos_lead=tl.POS_LEAD_TIME,
                 yaw_lead=tl.YAW_LEAD_TIME, speed_limit=tl.CHASER_SPEED_LIMIT,
                 avoid_margin=0.30, **_ignored):
        self.standoff = float(standoff)
        self.pos_lead = float(pos_lead)
        self.yaw_lead = float(yaw_lead)
        self.speed_limit = float(speed_limit)
        self.avoid_margin = float(avoid_margin)
        self._prev_sp = None      # 上一步的位置设定点，给限速用
        self._obst = None         # 障碍的字典形式，第一次 act 时缓存（静态）
        self._warned_ff = False

    def reset(self, seed=None):
        self._prev_sp = None
        self._obst = None
        self._warned_ff = False

    def _obstacles(self, obs):
        if self._obst is None:
            self._obst = [o.as_dict() if hasattr(o, "as_dict") else dict(o)
                          for o in (obs.obstacles or [])]
        return self._obst

    def act(self, obs):
        # 目标速度是轨迹的解析导数。chaser_follow_behind 靠它定义"身后"的方位。
        tvel = obs.target_vel if obs.target_vel is not None else (0.0, 0.0, 0.0)
        desired, yaw_cmd = tl.chaser_follow_behind(
            obs.target_pos,
            tvel,
            obs.chaser_pos,
            chaser_vel=obs.chaser_vel,
            standoff=self.standoff,
            pos_lead=self.pos_lead,
            yaw_lead=self.yaw_lead,
        )
        # 限速：位置设定点的变化率不超过 speed_limit，防止猛冲超调
        if self._prev_sp is not None:
            desired = tl.limit_speed(self._prev_sp, desired, obs.dt, self.speed_limit)

        # 避障：期望点进了障碍 / 通往期望点的直线被挡，就绕追踪机水平转一个角。
        # ⚠️ **偏航不受避障影响**，始终指向目标 —— 位置躲障碍躲到跟丢就本末倒置了。
        # ⚠️ 全挡死时原地不动（沿用上一步设定点），这是故意的：宁可停下来等缺口，
        #    也不要穿墙。停下来目标会飞走，但那会如实记成 lost_timeout。
        safe = tl.avoid_setpoint(obs.chaser_pos, desired, self._obstacles(obs),
                                 margin=self.avoid_margin)
        if safe is not None:
            desired = safe
        elif self._prev_sp is not None:
            desired = list(self._prev_sp)
        # 避障改写是**跳**过去的（实测一步最多跳 0.4 m），再限一次速，
        # 否则设定点会以超 speed_limit 的速度甩过去。两次限速之间差最多
        # limit*dt = 0.3 m，还在 avoid 的 0.30 m 安全边距里，不会因此撞上。
        if self._prev_sp is not None:
            desired = tl.limit_speed(self._prev_sp, desired, obs.dt, self.speed_limit)
        self._prev_sp = list(desired)

        # 兜底夹一下（chaser_follow_behind 内部已经夹过，这里防的是限速把点推歪）
        desired = [
            float(np.clip(desired[0], -XY_LIMIT, XY_LIMIT)),
            float(np.clip(desired[1], -XY_LIMIT, XY_LIMIT)),
            float(np.clip(desired[2], Z_MIN, Z_MAX)),
        ]
        return TrackerAction.position(desired, yaw_cmd)


class DoNothingTracker(Tracker):
    """验收对照：**锁在起始位姿**，不转向、不追、不动。

    第一次 `act()` 把当时的位置和偏航一起定死，之后整局只输出这一个设定点。
    runner 会先 settle 到 `episode.start_pose`，所以"不动"是物理意义上的不动。

    ⚠️ 位置和偏航**都要**从 settle 结果里锁，别只锁位置。踩过：`hold_yaw`
       默认写死 0.0，`scene_occluded` 的 start_pose 偏航是 π/2（朝 +y），
       结果追踪机站在 (0,3,2.5) 却朝 +x 看，目标在 69°~111° 方位上、
       远超 43.5° 半视场 —— `in_fov_ratio` 只有 0.09，直接判了 lost_timeout。
       `scene_lost` 的 start_pose 偏航恰好是 0 才侥幸没暴露。
    """

    name = "do_nothing"

    def __init__(self, hold_pos=None, hold_yaw=None, **_ignored):
        self.hold_pos = None if hold_pos is None else [float(v) for v in hold_pos]
        # None = 跟着 settle 结果锁；给浮点数就用给定的
        self.hold_yaw = None if hold_yaw is None else float(hold_yaw)

    def reset(self, seed=None):
        return None

    def act(self, obs):
        # 第一次调用就把当前实际**位姿**（位置 + 偏航）定死 —— 这样起始位姿由
        # runner 决定，跟踪器只是拒绝动它。
        if self.hold_pos is None:
            self.hold_pos = [float(obs.chaser_pos[0]), float(obs.chaser_pos[1]),
                             float(obs.chaser_pos[2])]
        yaw = self.hold_yaw if self.hold_yaw is not None else float(obs.chaser_attitude[2])
        return TrackerAction.position(self.hold_pos, yaw)


class ConstOffsetTracker(Tracker):
    """【调试用】停在目标正后方固定 offset，机头指向目标 —— 不用前馈。

    只用来在验收里制造"目标会走出去"的几何，别当 baseline 用。
    """

    name = "const_offset"

    def __init__(self, offset=(0.0, -4.0, 0.0), **_ignored):
        self.offset = [float(v) for v in offset]
        self._prev_sp = None

    def reset(self, seed=None):
        self._prev_sp = None

    def act(self, obs):
        tp = obs.target_pos
        desired = [
            float(np.clip(tp[0] + self.offset[0], -XY_LIMIT, XY_LIMIT)),
            float(np.clip(tp[1] + self.offset[1], -XY_LIMIT, XY_LIMIT)),
            float(np.clip(tp[2] + self.offset[2], Z_MIN, Z_MAX)),
        ]
        yaw = math.atan2(tp[1] - obs.chaser_pos[1], tp[0] - obs.chaser_pos[0])
        if self._prev_sp is not None:
            desired = tl.limit_speed(self._prev_sp, desired, obs.dt, tl.CHASER_SPEED_LIMIT)
        self._prev_sp = list(desired)
        return TrackerAction.position(desired, yaw)
