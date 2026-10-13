#!/usr/bin/env python3
"""P4：Gym 风格封装（docs/BENCH_PLAN.md §2.7），给训练用。

    env = TrackingEnv(cfg, device="cuda:1")
    obs = env.reset(seed=20261012)
    done = False
    while not done:
        action = my_tracker.act(obs)      # 或者 RL 策略(obs) -> action
        obs, done, info = env.step(action)

⚠️ 文件名刻意不叫 `tracking_env.py` —— 那个名字被 aerial_gym 的环境配置占了
   （`aerial_gym/config/env_config/tracking_env.py`），同名会互相踩。

`done` 由 §2.6 的终止条件驱动（连续丢失 / 碰撞 / 跑满）。
`info` 携带当步诊断量。可选 `reward_fn(obs, action) -> float` 覆盖默认实现。
"""

import numpy as np

from tracking_bench import bench_config
from tracking_bench.eval import bench_eval
from tracking_bench import runner
from tracking_bench.sensors import truth as T
from tracking_bench.trackers.tracker_api import TrackerInput, TrackerAction


def default_reward(obs: TrackerInput, action: TrackerAction) -> float:
    """默认 reward：负的距离误差 + 可见奖励。**训练时用你自己的覆盖它。**

        r = −|distance − standoff|  +  1.0·visible  − 0.5·(not in_fov)

    这个默认值只是让封装"能用"，不是设计出来的指标。
    """
    standoff = 2.0
    if obs.target_pos is not None and obs.chaser_pos is not None:
        d = float(np.linalg.norm(obs.target_pos - obs.chaser_pos))
    else:
        d = obs.distance
    r = -abs(d - standoff)
    if obs.visible:
        r += 1.0
    if not obs.in_fov:
        r -= 0.5
    return float(r)


class TrackingEnv:
    """一个 episode 的 Gym 风格视图。内部复用 `runner.SimSession`。"""

    metadata = {"render_modes": ["sensors"]}

    def __init__(self, cfg=None, device="cuda:0", reward_fn=None,
                 start_pose=None, session=None):
        self.cfg = cfg or bench_config.load_config(bench_config.DEFAULT_CONFIG_PATH)
        self.reward_fn = reward_fn or default_reward
        self._start_pose = start_pose
        self.session = session or runner.SimSession(self.cfg, device=device)
        self.policy = None
        self.metrics = None
        self._step_n = 0
        self._lost_steps = 0
        self._collided = False
        self._done = False
        self._last_obs = None
        self._last_action = None

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        """开一个新 episode，返回初始 `TrackerInput`。"""
        cfg = self.cfg
        ep = cfg.episode
        seed = cfg.seed if seed is None else int(seed)

        from tracking_bench.target import waypoint_policy as wp_mod
        tp = cfg.target_policy
        self.policy = wp_mod.WaypointPolicy(
            seed=seed, arena_yaml=cfg.arena_yaml,
            v_cruise=tp.v_cruise, v_max=tp.v_max,
            a_max=tp.a_max, t_min=tp.t_min,
            n_waypoints=tp.n_waypoints, speed_scale=tp.speed_scale,
        )
        if getattr(tp, "waypoints", None):
            self.policy.set_waypoints(tp.waypoints)
        if tp.speed_scale != 1.0:
            self.policy.set_speed_scale(tp.speed_scale)

        self.metrics = bench_eval.EpisodeMetrics(
            seed=seed, wall_layout_id=cfg.wall_layout_id,
            standoff=float(cfg.tracker.standoff),
            lost_criterion=ep.lost_criterion, dt=self.session.dt,
            waypoints=[list(map(float, w)) for w in self.policy.waypoints],
        )
        self._step_n = 0
        self._lost_steps = 0
        self._collided = False
        self._done = False

        tpos0, tvel0, tyaw0 = self.policy.state_at(0.0)
        start_pose = self._start_pose or getattr(ep, "start_pose", None)
        if start_pose is None:
            from tracking_bench import tracking_lib as tl
            desired, yaw0 = tl.chaser_follow_behind(
                tpos0, tvel0, tpos0, chaser_vel=None,
                standoff=cfg.tracker.standoff)
            start_pose = (desired[0], desired[1], desired[2], yaw0)
        self.session.settle(start_pose[0:3], start_pose[3], ep.settle_steps,
                            target_pos=tpos0, target_yaw=tyaw0)
        return self._observe(tpos0, tvel0, tyaw0, collision=False, record=False)

    def step(self, action: TrackerAction):
        """执行一个控制周期，返回 `(obs, done, info)`。

        ⚠️ 记录口径：**每个 `step()` 记一条，描述这一步结束时的状态**。
           `reset()` 返回的初始观测**不记录**。所以 `stats["steps_run"]` 等于
           `step()` 的调用次数，`fail_step` 也是。和 `runner.run_episode` 的
           "每周期记一条"对齐（只是那条记在周期开头）。
        """
        if self._done:
            raise RuntimeError("episode 已经结束了，先 reset()")

        cfg = self.cfg
        ep = cfg.episode
        dt = self.session.dt

        self._last_action = action
        self.session.apply_action(action, mode=cfg.tracker.action_mode)
        # 目标写到**本周期结束**的时刻（period k 覆盖 ((k-1)dt, k·dt]）
        npos, nvel, nyaw = self.policy.state_at((self._step_n + 1) * dt)
        self.session.write_target(npos, nyaw)
        self._collided = self.session.step()
        self.session.render()

        self._step_n += 1
        obs = self._observe(npos, nvel, nyaw, collision=self._collided, record=True)

        # ---- 终止（和 runner.run_episode 同一套判据）----
        lost = not obs[ep.lost_criterion]
        self._lost_steps = self._lost_steps + 1 if lost else 0
        reason = None
        if self._collided and ep.collision_is_fail:
            reason = "collision"
        elif self._lost_steps * dt >= ep.lost_timeout - 1e-9:
            reason = "lost_timeout"
        elif self._step_n >= int(ep.max_steps):
            reason = "max_steps"

        self._done = reason is not None
        if self._done:
            self.metrics.result = "success" if reason == "max_steps" else "fail"
            self.metrics.fail_reason = None if reason == "max_steps" else reason
            self.metrics.fail_step = self._step_n

        info = self._info(obs, reason)
        return obs, self._done, info

    def close(self):
        self.session.close()

    # ------------------------------------------------- 训练常用的小工具
    @property
    def observation(self):
        return self._last_obs

    @property
    def stats(self):
        """跑完之后拿 §2.6 那张表。"""
        return self.metrics.metrics() if self.metrics is not None else None

    def reward(self, obs=None, action=None):
        return float(self.reward_fn(obs or self._last_obs,
                                    action or self._last_action))

    # ------------------------------------------------- 内部
    def _observe(self, tpos, tvel, tyaw, collision, record=True):
        dt = self.session.dt
        cam_p, cam_R = self.session.cam_pose()
        seg, depth = self.session.frame()
        tr = T.compute_truth(seg, depth, tpos, cam_p, cam_R=cam_R)
        obs = runner.build_obs(self._step_n, self._step_n * dt, self.session,
                               self.policy, tr, tpos, tvel, tyaw, self.cfg)
        if record:
            self.metrics.record(step=self._step_n, t=self._step_n * dt, tr=tr,
                                chaser_yaw=float(obs.chaser_attitude[2]),
                                target_yaw=tyaw, collision=collision)
        self._last_obs = obs
        return obs

    def _info(self, obs, fail_reason):
        return {
            "step": self._step_n,
            "t": self._step_n * self.session.dt,
            "distance": obs.distance,
            "distance_err": obs.distance - self.cfg.tracker.standoff,
            "in_fov": obs.in_fov,
            "visible": obs.visible,
            "occluded": obs.occluded,
            "target_pixel_area": obs.target_pixel_area,
            "fail_reason": fail_reason,
            "lost_steps": self._lost_steps,
            "reward": self.reward_fn(obs, self._last_action),
        }
