#!/usr/bin/env python3
"""跑一次跟踪器测试。**对外唯一入口**。

    # 用配置里的 baseline 跑一个 episode
    python scripts/run_bench.py --config configs/bench.yaml --device cuda:1 --out /outputs/runs/run1

    # 用自己写的 tracker（一个 .py，里面有个 Tracker 子类）
    python scripts/run_bench.py --tracker-file my_tracker.py --device cuda:1 --out /outputs/runs/mine

    # 换 seed 连跑 3 个
    python scripts/run_bench.py --seeds 1,2,3 --device cuda:1 --out /outputs/runs/sweep

在哪个目录跑都行（配置路径会自己解析）；`--out` 的相对路径是相对**当前目录**的。

GPU：默认走 warp（depth/seg），**只占 1 张卡**。显式给 `--device cuda:N`，
**不要自己设 `CUDA_VISIBLE_DEVICES`**（`gpu_lock` 会自己锁到单卡并重 exec）。
⚠️ 共享服务器上一次最多用 2 张卡（AGENTS.md），本入口只占 1 张。
"""

import argparse
import importlib.util
import os
import sys

# 让 `import tracking_bench` 能用，不用把包装进 site-packages
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)

from tracking_bench.sim import gpu_lock  # noqa: E402

gpu_lock.confine_to_one_gpu()  # 必须在 import torch / isaacgym 之前

import isaacgym  # noqa: E402,F401
import numpy as np  # noqa: E402

from tracking_bench import bench_config  # noqa: E402
from tracking_bench import runner  # noqa: E402
from tracking_bench.trackers.tracker_api import Tracker  # noqa: E402


def load_tracker_from_file(path):
    """从一个 .py 文件里找出 `Tracker` 的子类并实例化。

    找多个就取第一个非 `Tracker` 自己的；找不到就报错并列出文件里的类。
    """
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_user_tracker", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cands = [v for v in vars(mod).values()
             if isinstance(v, type) and issubclass(v, Tracker) and v is not Tracker]
    if not cands:
        names = [k for k, v in vars(mod).items() if isinstance(v, type)]
        raise ValueError(
            f"{path} 里没有 Tracker 的子类。找到的类：{names}。"
            f"写一个 `class Xxx(Tracker)` 并实现 `act()`。"
        )
    return cands[0]()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=bench_config.DEFAULT_CONFIG_PATH)
    p.add_argument("--device", default="cuda:0", help="物理卡号，如 cuda:1")
    p.add_argument("--out", default="/outputs/runs/run", help="输出目录，约定写 /outputs/runs/<日期>-<名字>（机器上没挂 /outputs 时换成任意可写路径）")
    p.add_argument("--tracker-file", default=None,
                   help="自己写的 tracker 的 .py 文件（里面有 Tracker 子类）")
    p.add_argument("--tracker", default=None, help="覆盖 cfg.tracker.type")
    p.add_argument("--steps", type=int, default=None, help="覆盖 episode.max_steps")
    p.add_argument("--lost-timeout", type=float, default=None,
                   help="覆盖 episode.lost_timeout（秒）")
    p.add_argument("--seed", type=int, default=None, help="覆盖 seed")
    p.add_argument("--seeds", default=None,
                   help="逗号分隔的多个 seed，跑 sweep（会忽略 --seed）")
    p.add_argument("--start-pose", default=None,
                   help="起始位姿 'x,y,z,yaw_deg'，覆盖 episode.start_pose")
    p.add_argument("--settle", type=int, default=None, help="覆盖 episode.settle_steps")
    args = p.parse_args()

    overrides = {}
    if args.tracker:
        overrides.setdefault("tracker", {})["type"] = args.tracker
    if args.steps is not None:
        overrides.setdefault("episode", {})["max_steps"] = int(args.steps)
    if args.lost_timeout is not None:
        overrides.setdefault("episode", {})["lost_timeout"] = float(args.lost_timeout)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.settle is not None:
        overrides.setdefault("episode", {})["settle_steps"] = int(args.settle)
    if args.start_pose is not None:
        vals = [float(v) for v in args.start_pose.split(",")]
        if len(vals) != 4:
            raise SystemExit("--start-pose 要 'x,y,z,yaw_deg' 四个数")
        vals[3] = float(np.radians(vals[3]))
        overrides.setdefault("episode", {})["start_pose"] = vals

    cfg = bench_config.load_config(args.config, overrides=overrides or None)

    session = runner.SimSession(cfg, device=args.device)

    if args.tracker_file:
        def factory():
            return load_tracker_from_file(args.tracker_file)
    else:
        def factory():
            return runner.make_tracker(cfg)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        results = runner.run_suite(session, factory, cfg, seeds, args.out,
                                   tag="")
        n_ok = sum(1 for r in results if r["result"] == "success")
        print(f"\n{n_ok}/{len(results)} 个 episode 成功 -> {os.path.abspath(args.out)}")
    else:
        tracker = factory()
        m = runner.run_episode(session, tracker, cfg=cfg,
                               seed=cfg.seed, out_dir=args.out)
        print(f"\n{m['result']}"
              + (f" ({m['fail_reason']} @ step {m['fail_step']})" if m["fail_reason"] else "")
              + f", {m['steps_run']} 步 -> {os.path.abspath(args.out)}")
        print("  metrics.json / summary.md / config_snapshot.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
