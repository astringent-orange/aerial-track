#!/usr/bin/env python3
"""P4：把仿真、目标策略、跟踪器、指标串成一个可跑的 episode（docs/BENCH_PLAN.md §2.6）。

    SimSession    一次建环境，管张量 / 渲染 / 目标写入 / 碰撞读数
    run_episode   跑一个 episode，返回 metrics dict（并可选落盘）
    run_suite     换 seed 连跑多个

⚠️ 这是**唯一**碰 isaacgym / aerial_gym 的地方之一。跟踪器作者不该 import 它
   （P5 会把这一层收进 `tracking_bench/sim/`）。

---------------------------------------------------------------- 一步的时序
step n（1-based）覆盖控制周期 [(n-1)·dt, n·dt)，**记录的观测取在 t = (n-1)·dt**：

    1. 从"当前帧"（上一个周期结束时渲染的）+ 目标状态 (n-1)·dt 拼出 TrackerInput
    2. 把这一步记进 metrics（含碰撞标志：这一帧是哪个周期产生的）
    3. 判终止：连续丢失 ≥ lost_timeout  → fail("lost_timeout")
              发生碰撞且 collision_is_fail → fail("collision")
    4. tracker.act(obs) -> TrackerAction
    5. 写目标到 t = n·dt，应用动作，env_manager.step()
    6. 读 `gtd["crashes"]`（本周期的碰撞），渲染下一帧

终止检查在**记录之后**，所以 `fail_step == steps_run`：fail_step=30 就是
"跑了 30 步、第 30 步那帧判出来的"。

---------------------------------------------------------------- 碰撞怎么读
`env_manager.step()` 开头 `reset_tensors()` 清零 `gtd["crashes"]`，
每个物理子步的 `compute_observations()` 累加
`(‖robot_contact_force‖ > collision_force_threshold)`。所以 **step() 返回后
`gtd["crashes"][e] > 0` 就是"这个控制周期撞了"**。

⚠️ `tracking_env.py` 里 `reset_on_collision = False` 是**故意**的：撞了以后
   aerial_gym 不自动 reset，由我们自己判失败。要是改回 True，一次碰撞就会把
   布局和目标全重采样，episode 数据直接作废。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tracking_bench import bench_config
from tracking_bench.eval import bench_eval
from tracking_bench import tracking_lib as tl
from tracking_bench.sensors import truth as T
from tracking_bench.trackers.tracker_api import (
    TrackerInput, TrackerAction, Obstacle,
    XY_LIMIT, Z_MIN, Z_MAX, SPEED_LIMIT, DEFAULT_ACTION_MODE,
)



# --- 仿真会话在 sim/ 门面后面（全项目唯一碰 IsaacGym 的地方）---
from tracking_bench.sim.session import (
    SimSession, build_env_quiet, quat_to_euler, yaw_to_quat, quat_mul,
)

# --------------------------------------------------------------------------- #
#  TrackerInput 组装
# --------------------------------------------------------------------------- #
def build_obs(step_n, t, session: SimSession, policy, tr, target_pos, target_vel,
              target_yaw, cfg):
    pos, vel, rpy, ang = session.chaser_state()
    cam_p, cam_R = session.cam_pose()
    seg, depth = session.frame()
    lo = np.asarray(tl.ARENA_BOUNDS_MIN, dtype=np.float64)
    hi = np.asarray(tl.ARENA_BOUNDS_MAX, dtype=np.float64)
    return TrackerInput(
        t=t, step=step_n, dt=session.dt,
        chaser_pos=pos, chaser_vel=vel, chaser_attitude=rpy, chaser_angvel=ang,
        depth=depth,
        rgb=None,                      # use_warp=True 时没有 RGB（占 2 卡，默认关）
        seg=seg,
        K=np.asarray(T.K, dtype=np.float64),
        T_body_cam=session.T_body_cam,
        target_pos=np.asarray(target_pos, dtype=np.float64),
        target_vel=np.asarray(target_vel, dtype=np.float64),
        target_yaw=float(target_yaw),
        target_valid=True,
        in_fov=tr["in_fov"], visible=tr["visible"], occluded=tr["occluded"],
        bbox=tr["bbox"], center_pixel=tr["center_pixel"],
        projected_pixel=tr["projected_pixel"],
        target_pixel_area=tr["target_pixel_area"],
        target_depth_mean=tr["target_depth_mean"],
        distance=tr["distance"],
        arena_bounds=(lo, hi),
        obstacles=list(session.obstacles),
        wall_layout_id=cfg.wall_layout_id,
    )


# --------------------------------------------------------------------------- #
#  run_episode
# --------------------------------------------------------------------------- #
def run_episode(session: SimSession, tracker, cfg=None, seed=None, out_dir=None,
                policy=None, start_pose=None, tag="", verbose=True):
    """跑一个 episode，返回 metrics dict。

    参数
    ----
    session     已建好的仿真会话
    tracker     `tracker_api.Tracker` 实例
    cfg         `BenchConfig`，默认用 session.cfg
    seed        目标轨迹的 seed。None 就用 cfg.seed
    out_dir     给了就写 metrics.json / summary.md / config_snapshot.yaml
    policy      给了就复用同一个 `WaypointPolicy`（否则按 cfg 新建）
    start_pose  (x, y, z, yaw)。None 就用 baseline 几何从目标 t=0 算一个
    """
    cfg = cfg or session.cfg
    ep = cfg.episode
    dt = session.dt
    seed = cfg.seed if seed is None else int(seed)

    from tracking_bench.target import waypoint_policy as wp_mod
    if policy is None:
        tp = cfg.target_policy
        policy = wp_mod.WaypointPolicy(
            seed=seed, arena_yaml=cfg.arena_yaml,
            v_cruise=tp.v_cruise, v_max=tp.v_max,
            a_max=tp.a_max, t_min=tp.t_min,
            n_waypoints=tp.n_waypoints, speed_scale=tp.speed_scale,
        )
        if getattr(tp, "waypoints", None):
            policy.set_waypoints(tp.waypoints)
    else:
        policy.reset(seed)
    if cfg.target_policy.speed_scale != 1.0:
        policy.set_speed_scale(cfg.target_policy.speed_scale)

    waypoints = [list(map(float, w)) for w in policy.waypoints]

    # ---- 起始位姿 ----
    tpos0, tvel0, tyaw0 = policy.state_at(0.0)
    if start_pose is None:
        start_pose = getattr(ep, "start_pose", None)
    if start_pose is None:
        desired, yaw0 = tl.chaser_follow_behind(
            tpos0, tvel0, tpos0, chaser_vel=None, standoff=cfg.tracker.standoff)
        start_pose = (desired[0], desired[1], desired[2], yaw0)
    start_pose = tuple(float(v) for v in start_pose)

    # ---- 收集器 ----
    metrics = bench_eval.EpisodeMetrics(
        seed=seed, wall_layout_id=cfg.wall_layout_id,
        standoff=float(cfg.tracker.standoff),
        lost_criterion=ep.lost_criterion, dt=dt,
        waypoints=waypoints,
    )

    # ---- settle：目标钉在 t=0，追踪机飞到起始位姿 ----
    if verbose:
        print(f"[{tag}] settle {ep.settle_steps} 步 -> "
              f"pos=({start_pose[0]:.2f},{start_pose[1]:.2f},{start_pose[2]:.2f}) "
              f"yaw={math.degrees(start_pose[3]):.0f}°", flush=True)
    session.settle(start_pose[0:3], start_pose[3], ep.settle_steps,
                   target_pos=tpos0, target_yaw=tyaw0)

    tracker.reset(seed)

    lost_steps = 0
    collided = False          # 上一个控制周期撞了没有（settle 之后清零）
    result, fail_reason, fail_step = "success", None, None

    for k in range(int(ep.max_steps)):
        n = k + 1                      # 1-based 步号
        t = k * dt
        tpos, tvel, tyaw = policy.state_at(t)
        cam_p, cam_R = session.cam_pose()
        seg, depth = session.frame()
        tr = T.compute_truth(seg, depth, tpos, cam_p, cam_R=cam_R)
        obs = build_obs(n, t, session, policy, tr, tpos, tvel, tyaw, cfg)

        metrics.record(step=n, t=t, tr=tr,
                       chaser_yaw=float(obs.chaser_attitude[2]),
                       target_yaw=tyaw, collision=collided)

        # ---- 终止判定（记录之后，所以 fail_step == steps_run）----
        lost = not tr[ep.lost_criterion]
        lost_steps = lost_steps + 1 if lost else 0
        if collided and ep.collision_is_fail:
            result, fail_reason, fail_step = "fail", "collision", n
            break
        if lost_steps * dt >= ep.lost_timeout - 1e-9:
            result, fail_reason, fail_step = "fail", "lost_timeout", n
            break
        if k == int(ep.max_steps) - 1:
            break                        # 跑满，success

        # ---- 跟踪器决策 + 推进 ----
        action = tracker.act(obs)
        session.apply_action(action, mode=cfg.tracker.action_mode)
        npos, nvel, nyaw = policy.state_at(t + dt)
        session.write_target(npos, nyaw)
        collided = session.step()
        session.render()

    metrics.result = result
    metrics.fail_reason = fail_reason
    metrics.fail_step = fail_step

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        snap = bench_config.save_config_snapshot(cfg, out_dir)
        m = metrics.metrics()
        m["config_snapshot_path"] = snap
        m["config_snapshot"] = cfg.to_dict()
        bench_eval.write_metrics_json(m, out_dir)
        bench_eval.write_summary_md(
            m, out_dir,
            title=f"Episode 摘要 — {tracker.name} seed={seed} → **{result}**")
        if verbose:
            print(f"[{tag}] {result}"
                  + (f" ({fail_reason} @ step {fail_step})" if fail_reason else "")
                  + f", {metrics.steps_run} 步 -> {out_dir}", flush=True)
        return m

    m = metrics.metrics()
    m["config_snapshot"] = cfg.to_dict()
    if verbose:
        print(f"[{tag}] {result}"
              + (f" ({fail_reason} @ step {fail_step})" if fail_reason else "")
              + f", {metrics.steps_run} 步", flush=True)
    return m


# --------------------------------------------------------------------------- #
#  run_suite
# --------------------------------------------------------------------------- #
def run_suite(session, tracker_factory, cfg, seeds, out_root, tag=""):
    """换 seed 连跑。`tracker_factory()` 每个 episode 造一个新跟踪器。"""
    results = []
    for s in seeds:
        tracker = tracker_factory()
        out_dir = os.path.join(out_root, f"seed_{s}")
        m = run_episode(session, tracker, cfg=cfg, seed=s, out_dir=out_dir,
                        tag=f"{tag}seed={s}")
        results.append(m)
    return results


# --------------------------------------------------------------------------- #
#  跟踪器注册表（给别人接自己的 tracker 用）
# --------------------------------------------------------------------------- #
TRACKER_REGISTRY = {}


def register_tracker(name, cls):
    TRACKER_REGISTRY[name] = cls


def make_tracker(cfg: bench_config.BenchConfig):
    """按 `cfg.tracker.type` 造跟踪器。默认注册了 `baseline` / `do_nothing`。"""
    from tracking_bench.trackers.baseline_tracker import BaselineTracker, DoNothingTracker, ConstOffsetTracker
    if not TRACKER_REGISTRY:
        register_tracker("baseline", BaselineTracker)
        register_tracker("do_nothing", DoNothingTracker)
        register_tracker("const_offset", ConstOffsetTracker)
    name = cfg.tracker.type
    if name not in TRACKER_REGISTRY:
        raise KeyError(
            f"没注册的 tracker 类型 {name!r}。已注册：{sorted(TRACKER_REGISTRY)}。"
            f"用 runner.register_tracker('名字', MyTracker) 注册。"
        )
    kw = dict(cfg.tracker.extra)
    kw.setdefault("standoff", cfg.tracker.standoff)
    return TRACKER_REGISTRY[name](**kw)
