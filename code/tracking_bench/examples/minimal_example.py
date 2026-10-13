#!/usr/bin/env python3
"""`INTERFACE.md` 里那个"最小可运行示例"的原文件。

**这一份就是文档里那段代码**，一字不差 —— `check_p4` 会 import 并跑它，
保证文档里的例子不会烂掉。要改就两边一起改。

跑法（容器内）：
    cd /workspace/tracking_bench
    python scripts/run_bench.py --tracker-file examples/minimal_example.py --steps 60 --device cuda:1
"""

import math

from tracking_bench.trackers.tracker_api import Tracker, TrackerAction, XY_LIMIT, Z_MIN, Z_MAX


class MinimalTracker(Tracker):
    """最小跟踪器：飞到目标后方 2.5 m、上方 0.8 m，机头指向目标。

    想写自己的跟踪器，照这个样子继承 `Tracker`、实现 `act()` 就行 ——
    不用碰 IsaacGym，不用碰仿真内部。

    ⚠️ 这里用 `obs.target_pos` 是因为接口演示要短。它是仿真真值，
       **拿它做决策 = 作弊**。真算法请用 `obs.depth` / `obs.rgb`。
    """

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
