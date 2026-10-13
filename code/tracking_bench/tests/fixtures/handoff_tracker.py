#!/usr/bin/env python3
"""交接验证 fixture：**只依据 README + INTERFACE.md 写出来的 tracker**。

P5 验收有一条「不看 docs/BENCH_PLAN.md，单凭 README + INTERFACE.md 能写一个新的
tracker 并跑起来」。这个文件就是那次验证的产物：内容只用了两份文档里出现过的
东西（`Tracker` / `TrackerAction.position` / `XY_LIMIT, Z_MIN, Z_MAX` /
`obs.depth` 的值域说明 / §3 的位置约束），没有引用包内其他实现。

策略故意和 `minimal_example.py` 不同：拿深度图的近程信息把机身压低一点，
证明"用自己的传感器输入写控制律"这条路是通的。
"""

import math

from tracking_bench.trackers.tracker_api import (
    Tracker, TrackerAction, XY_LIMIT, Z_MIN, Z_MAX,
)


class HandoffTracker(Tracker):
    """跟在目标后方 2.5 m、上方 0.8 m（深度图近程太多就往下压）。"""

    name = "handoff"

    def reset(self, seed=None):
        return None

    def act(self, obs):
        tp = obs.target_pos
        z_off = 0.8
        # 深度图值域 [-1]∪[0,1]：负数 = 比 0.2 m 还近，不是"没打中"
        d = obs.depth
        near = float((d < 0).mean())
        if near > 0.05:            # 近程像素太多，压低一点别贴脸
            z_off = 0.3
        p = [
            max(-XY_LIMIT, min(XY_LIMIT, float(tp[0]))),
            max(-XY_LIMIT, min(XY_LIMIT, float(tp[1]) - 2.5)),
            max(Z_MIN, min(Z_MAX, float(tp[2]) + z_off)),
        ]
        yaw = math.atan2(tp[1] - obs.chaser_pos[1], tp[0] - obs.chaser_pos[0])
        return TrackerAction.position(p, yaw)
