#!/usr/bin/env python3
"""import 冒烟：确认整个包（含碰 IsaacGym 的 sim/）能加载、配置能解析。

不起仿真、不开渲染，几秒钟跑完。用法（任意目录）：
    PYTHONPATH=/workspace/tracking_bench python smoke_import.py
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)

from tracking_bench.paths import (  # noqa: E402
    package_root, find_config, aerial_gym_asset_dir,
)
from tracking_bench import bench_config  # noqa: E402
from tracking_bench import tracking_lib as tl  # noqa: E402
from tracking_bench.target import waypoint_policy as wp  # noqa: E402
from tracking_bench.eval import bench_eval  # noqa: E402
from tracking_bench.trackers import baseline_tracker  # noqa: E402

# 碰 IsaacGym 的部分最后 import（它会初始化 warp / 打印设备）
from tracking_bench.sim import session  # noqa: E402
from tracking_bench import runner, bench_env  # noqa: E402


def main():
    print("package_root :", package_root())
    assert os.path.isdir(os.path.join(package_root(), "configs"))

    cfg = bench_config.load_config()                 # 默认 bench.yaml
    assert os.path.isfile(cfg.arena_yaml), cfg.arena_yaml
    cfg2 = bench_config.load_config(find_config("scene_lost.yaml"))
    assert os.path.isfile(cfg2.arena_yaml), cfg2.arena_yaml

    assert tl.N_WALLS == 8 and tl.N_CYLINDERS == 8, (tl.N_WALLS, tl.N_CYLINDERS)
    assert tl.CYLINDER_SIZE == (0.6, 0.6, 3.0), tl.CYLINDER_SIZE
    assert os.path.isfile(wp._CFG_PATH)

    for sub in ("cylinders", "walls_thin", "target_drone"):
        d = aerial_gym_asset_dir(sub)
        assert "environment_assets" in d, d

    assert session.SimSession is not None
    assert runner.run_episode is not None and bench_env.TrackingEnv is not None
    assert bench_eval.EpisodeMetrics is not None
    assert baseline_tracker.BaselineTracker is not None
    print("OK smoke_import")


if __name__ == "__main__":
    main()
