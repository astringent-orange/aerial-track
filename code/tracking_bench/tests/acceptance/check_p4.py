#!/usr/bin/env python3
"""P4 验收：接口层 + 自动统计 + 文档。对照 docs/BENCH_PLAN.md §3 P4 的验收清单。

    1. baseline 跑 600 步，出 metrics.json + summary.md，数字合理
    2. "什么都不做"的 tracker：3 s 后判失败，fail_reason="lost_timeout"，fail_step=30
    3. lost_timeout 调大，同场景不再 30 步就停（跑得满）
    4. 遮挡 > 3 s 不算丢失（in_fov 仍为真）；换成 visible 判据同一场景应当失败
    5. INTERFACE.md 的最小示例能直接跑通
    6. 换 seed 跑两次，waypoints 不同但都能跑通

⚠️ 所有判据都**不依赖 `env_manager.reset()` 的随机初始位姿** ——
   每个场景都在配置里钉死 `episode.start_pose`（P1 那次假失败的教训）。

用法（容器内，先 cd /workspace）：
    python check_p4.py --device cuda:1
"""

import argparse
import copy
import math
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)                    # 让 `import tracking_bench` 能用
sys.path.insert(0, os.path.join(_PKG_ROOT, "examples"))   # minimal_example

from tracking_bench.sim import gpu_lock
gpu_lock.confine_to_one_gpu()  # 必须在 import torch / isaacgym 之前

import isaacgym  # noqa: E402,F401
import numpy as np  # noqa: E402

from tracking_bench import bench_config  # noqa: E402
from tracking_bench.eval import bench_eval  # noqa: E402
from tracking_bench import runner  # noqa: E402
from tracking_bench.trackers.tracker_api import Tracker  # noqa: E402

# 配置在包根的 configs/，不在 tests/acceptance/ 下（按 __file__ 拼会拼错）。
CFG_DIR = os.path.join(_PKG_ROOT, "configs")


class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, cond, detail=""):
        cond = bool(cond)
        self.results.append((name, cond))
        print(f"  {'✅' if cond else '❌'} {name}"
              + (f": {detail}" if detail else ""), flush=True)
        return cond

    @property
    def ok(self):
        return all(c for _, c in self.results)


def show(m, label=""):
    de, ye, pe = m["distance_err"], m["yaw_err_deg"], m["pixel_err"]
    print(f"  [{label}] result={m['result']}"
          + (f" reason={m['fail_reason']}@{m['fail_step']}" if m["fail_reason"] else "")
          + f" steps={m['steps_run']} t={m['time_run']:.1f}s", flush=True)
    print(f"    dist_err  mean={de['mean']} std={de['std']} max={de['max']} "
          f"bias={de['bias']}", flush=True)
    print(f"    yaw_err°  mean={ye['mean']} std={ye['std']} max={ye['max']} "
          f"bias={ye['bias']}", flush=True)
    print(f"    pixel_err u={pe['u_mean']}±{pe['u_std']} "
          f"v={pe['v_mean']}±{pe['v_std']}", flush=True)
    print(f"    in_fov={m['in_fov_ratio']} visible={m['visible_ratio']} "
          f"occluded_steps={m['occluded_steps']} collisions={m['collision_count']}",
          flush=True)


def load_cfg(name, **over):
    path = os.path.join(CFG_DIR, name)
    ov = {}
    for k, v in over.items():
        if k in ("max_steps", "lost_timeout", "lost_criterion", "settle_steps",
                 "start_pose", "collision_is_fail"):
            ov.setdefault("episode", {})[k] = v
        elif k in ("type", "standoff", "action_mode"):
            ov.setdefault("tracker", {})[k] = v
        elif k in ("seed", "num_envs", "use_warp", "arena_yaml"):
            ov[k] = v
        else:
            raise KeyError(k)
    return bench_config.load_config(path, overrides=ov or None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out-dir", default="p4_check")
    p.add_argument("--quick", action="store_true",
                   help="只跑最短的一轮（开发时用），不作正式验收")
    args = p.parse_args()

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    ck = Checker()
    print("=" * 92, flush=True)
    print("P4 验收：接口层 + 自动统计 + 文档", flush=True)
    print("=" * 92, flush=True)

    # 基础场景配置（baseline 的主场景）
    n_main = 120 if args.quick else 600
    cfg_main = load_cfg("bench.yaml", max_steps=n_main)
    session = runner.SimSession(cfg_main, device=args.device)
    print(f"  dt = {session.dt} s，T_body_cam 对角 = "
          f"{np.round(np.diag(session.T_body_cam)[:3], 3)}", flush=True)

    # ------------------------------------------------------------------ 1
    print("=" * 92, flush=True)
    print("1) baseline 跑完整 episode，出 metrics.json + summary.md", flush=True)
    print("=" * 92, flush=True)
    d1 = os.path.join(out_root, "baseline")
    tr1 = runner.make_tracker(cfg_main)
    m1 = runner.run_episode(session, tr1, cfg=cfg_main, seed=cfg_main.seed,
                            out_dir=d1, tag="baseline")
    show(m1, "baseline")
    ck.check("1a baseline 跑满 %d 步，result=success" % n_main,
             m1["result"] == "success" and m1["steps_run"] == n_main,
             f"result={m1['result']} steps_run={m1['steps_run']}")
    ck.check("1b metrics.json + summary.md + config_snapshot.yaml 都落盘了",
             all(os.path.getsize(os.path.join(d1, f)) > 0
                 for f in ("metrics.json", "summary.md", "config_snapshot.yaml")),
             f"{sorted(os.listdir(d1))}")
    de = m1["distance_err"]
    ck.check("1c 距离误差合理（mean |dist−standoff| < 1.0 m）",
             de["mean"] is not None and de["mean"] < 1.0,
             f"mean={de['mean']} max={de['max']}")
    ck.check("1d 可见率合理（in_fov_ratio > 0.8）",
             m1["in_fov_ratio"] is not None and m1["in_fov_ratio"] > 0.8,
             f"in_fov_ratio={m1['in_fov_ratio']} visible_ratio={m1['visible_ratio']}")
    ck.check("1e 没撞（collision_count == 0）",
             m1["collision_count"] == 0,
             f"collision_count={m1['collision_count']}")
    ck.check("1f 复现字段齐：seed / wall_layout_id / waypoints",
             m1["seed"] == cfg_main.seed
             and m1["wall_layout_id"] == cfg_main.wall_layout_id
             and len(m1["waypoints"]) >= 2,
             f"seed={m1['seed']} layout={m1['wall_layout_id']} "
             f"waypoints={len(m1['waypoints'])} 个")

    # ------------------------------------------------------------------ 2
    print("=" * 92, flush=True)
    print("2) DoNothing：连续丢失 3 s 判失败（scene_lost）", flush=True)
    print("=" * 92, flush=True)
    cfg_lost = load_cfg("scene_lost.yaml",
                        max_steps=(60 if args.quick else 600))
    n_lost = int(cfg_lost.episode.max_steps)   # 这一组的满程步数（quick 下 60）
    d2 = os.path.join(out_root, "do_nothing_lost")
    tr2 = runner.make_tracker(cfg_lost)
    m2 = runner.run_episode(session, tr2, cfg=cfg_lost, seed=cfg_lost.seed,
                            out_dir=d2, tag="do_nothing")
    show(m2, "do_nothing")
    ck.check("2a result=fail, fail_reason=lost_timeout",
             m2["result"] == "fail" and m2["fail_reason"] == "lost_timeout",
             f"result={m2['result']} fail_reason={m2['fail_reason']}")
    ck.check("2b fail_step == 30（连续丢失 3.0 s = 30 步）",
             m2["fail_step"] == 30 and m2["steps_run"] == 30,
             f"fail_step={m2['fail_step']} steps_run={m2['steps_run']}")
    ck.check("2c 失败的 episode 也把指标写出来了（result=fail 但 metrics 在）",
             m2["distance_err"]["mean"] is not None
             and os.path.exists(os.path.join(d2, "metrics.json")),
             f"已收集 {m2['steps_run']} 步，"
             f"metrics.json 在 {d2}/")

    # ------------------------------------------------------------------ 3
    print("=" * 92, flush=True)
    print("3) lost_timeout 可配（同场景，只改超时）", flush=True)
    print("=" * 92, flush=True)
    # 3a：30 s 超时 —— 场景里最长的连续丢失约 11.8 s，不该触发
    cfg_30 = copy.deepcopy(cfg_lost)
    cfg_30.episode.lost_timeout = 30.0
    d3a = os.path.join(out_root, "lost_timeout_30s")
    m3a = runner.run_episode(session, runner.make_tracker(cfg_30), cfg=cfg_30,
                             seed=cfg_30.seed, out_dir=d3a, tag="timeout30s")
    show(m3a, "timeout=30s")
    ck.check("3a lost_timeout=30 s：不在 30 步停，跑得远比 30 步长",
             m3a["steps_run"] > 30 and m3a["fail_step"] != 30,
             f"steps_run={m3a['steps_run']} fail_step={m3a['fail_step']} "
             f"result={m3a['result']}")
    # 3b：超时比 episode 时长还长 —— 一定跑满
    cfg_big = copy.deepcopy(cfg_lost)
    cfg_big.episode.lost_timeout = float(n_lost) * cfg_big.dt + 10.0
    d3b = os.path.join(out_root, "lost_timeout_huge")
    m3b = runner.run_episode(session, runner.make_tracker(cfg_big), cfg=cfg_big,
                             seed=cfg_big.seed, out_dir=d3b, tag="timeoutHuge")
    show(m3b, "timeout>episode")
    ck.check("3b lost_timeout > episode 时长：跑满 %d 步，result=success" % n_lost,
             m3b["steps_run"] == n_lost and m3b["result"] == "success",
             f"steps_run={m3b['steps_run']} result={m3b['result']}")
    ck.check("3c 超时确实是配置驱动的（同一场景 3 s / 30 s / 更大 三种结果都对）",
             m2["fail_step"] == 30 and m3a["steps_run"] > 30
             and m3b["steps_run"] == n_lost,
             f"3s->{m2['fail_step']} 步停；30s->{m3a['steps_run']} 步；"
             f"更大->{m3b['steps_run']} 步跑满")

    # ------------------------------------------------------------------ 4
    print("=" * 92, flush=True)
    print("4) 遮挡 > 3 s 不算丢失（scene_occluded）", flush=True)
    print("=" * 92, flush=True)
    cfg_occ = load_cfg("scene_occluded.yaml")
    d4 = os.path.join(out_root, "occluded_in_fov")
    m4 = runner.run_episode(session, runner.make_tracker(cfg_occ), cfg=cfg_occ,
                            seed=cfg_occ.seed, out_dir=d4, tag="occluded")
    show(m4, "lost_criterion=in_fov")
    ck.check("4a 目标确实被挡了（occluded_steps ≥ 30，即连续遮挡 ≥ 3 s）",
             m4["occluded_steps"] is not None and m4["occluded_steps"] >= 30,
             f"occluded_steps={m4['occluded_steps']}")
    ck.check("4b 遮挡不算丢失：in_fov_ratio 高，没因遮挡判失败",
             m4["fail_reason"] != "lost_timeout"
             and m4["in_fov_ratio"] is not None and m4["in_fov_ratio"] > 0.9,
             f"fail_reason={m4['fail_reason']} in_fov_ratio={m4['in_fov_ratio']} "
             f"visible_ratio={m4['visible_ratio']}")
    # 4c：对照 —— 判据换成 visible，同一场景应当判失败。
    #     这一条证明 `lost_criterion` 是真接上的，不是写在配置里没人读。
    cfg_vis = copy.deepcopy(cfg_occ)
    cfg_vis.episode.lost_criterion = "visible"
    d4c = os.path.join(out_root, "occluded_visible")
    m4c = runner.run_episode(session, runner.make_tracker(cfg_vis), cfg=cfg_vis,
                             seed=cfg_vis.seed, out_dir=d4c, tag="lost=visible")
    show(m4c, "lost_criterion=visible")
    ck.check("4c 对照：判据换成 visible，同一场景判 lost_timeout",
             m4c["fail_reason"] == "lost_timeout",
             f"fail_reason={m4c['fail_reason']} fail_step={m4c['fail_step']}")

    # ------------------------------------------------------------------ 5
    print("=" * 92, flush=True)
    print("5) INTERFACE.md 的最小示例能跑通", flush=True)
    print("=" * 92, flush=True)
    import minimal_example
    cfg_min = load_cfg("bench.yaml", max_steps=(40 if args.quick else 80))
    d5 = os.path.join(out_root, "minimal_example")
    m5 = runner.run_episode(session, minimal_example.MinimalTracker(),
                            cfg=cfg_min, seed=cfg_min.seed, out_dir=d5,
                            tag="minimal")
    show(m5, "MinimalTracker")
    ck.check("5a MinimalTracker 跑得通，没崩、没撞",
             m5["steps_run"] == cfg_min.episode.max_steps
             and m5["collision_count"] == 0,
             f"steps_run={m5['steps_run']} collisions={m5['collision_count']}")
    ck.check("5b 它确实是个能用的跟踪器（in_fov_ratio > 0.5）",
             m5["in_fov_ratio"] is not None and m5["in_fov_ratio"] > 0.5,
             f"in_fov_ratio={m5['in_fov_ratio']}")

    # ------------------------------------------------------------------ 6
    print("=" * 92, flush=True)
    print("6) 换 seed：waypoints 不同，但都能跑通", flush=True)
    print("=" * 92, flush=True)
    cfg_s = load_cfg("bench.yaml", max_steps=(60 if args.quick else 200))
    seeds = [20261012, 7]
    ms = []
    for s in seeds:
        d = os.path.join(out_root, f"seed_{s}")
        m = runner.run_episode(session, runner.make_tracker(cfg_s), cfg=cfg_s,
                               seed=s, out_dir=d, tag=f"seed={s}")
        ms.append(m)
    wp0 = ms[0]["waypoints"]
    wp1 = ms[1]["waypoints"]
    print(f"  seed={seeds[0]} 前 3 个路点：{[list(np.round(w, 2)) for w in wp0[:3]]}",
          flush=True)
    print(f"  seed={seeds[1]} 前 3 个路点：{[list(np.round(w, 2)) for w in wp1[:3]]}",
          flush=True)
    ck.check("6a 两个 seed 都跑通",
             all(m["result"] in ("success", "fail") and m["steps_run"] > 0
                 for m in ms),
             f"steps_run={[m['steps_run'] for m in ms]}")
    ck.check("6b 两个 seed 的 waypoints 不同",
             wp0 != wp1,
             f"P0: {wp0[0]} vs {wp1[0]}")
    ck.check("6c metrics.json 里都带了 seed 和完整路点",
             all(m["seed"] in seeds and len(m["waypoints"]) >= 2 for m in ms),
             f"seeds={[m['seed'] for m in ms]}")

    # ------------------------------------------------------------------
    print("=" * 92, flush=True)
    print("P4 验收汇总", flush=True)
    print("=" * 92, flush=True)
    for name, cond in ck.results:
        print(f"  {'✅' if cond else '❌'} {name}", flush=True)
    print(f"\n产物 -> {os.path.abspath(out_root)}/", flush=True)
    print("  baseline/  do_nothing_lost/  lost_timeout_30s/  lost_timeout_huge/",
          flush=True)
    print("  occluded_in_fov/  occluded_visible/  minimal_example/  seed_*/",
          flush=True)
    print(f"\n{'P4 通过' if ck.ok else 'P4 未通过'}", flush=True)
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main())
