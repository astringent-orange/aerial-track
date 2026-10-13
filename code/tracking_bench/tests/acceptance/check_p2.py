#!/usr/bin/env python3
"""P2 验收：路点 + 五次多项式目标轨迹。对照 docs/BENCH_PLAN.md §3 P2 的验收清单。

    1. 数值差分验证 p、v 连续，a 在段边界为 0
    2. 目标全程不出墙、不出场地（查的是**实际飞出来的曲线**，不是折线）
    3. 同 seed 轨迹完全一致，换 seed 轨迹不同
    4. 峰值速度 ≤ v_max、峰值加速度 ≤ a_max
    5. 跑 600 步目标全程在动，无瞬移（接进仿真写位置验证）

前 4 项是纯数值，不碰 IsaacGym；第 5 项才建环境。

用法（容器内，先 cd /workspace）：
    python check_p2.py                        # 只跑数值部分，秒级
    python check_p2.py --with-sim             # 加上仿真 600 步（要占 1 张 GPU）
    python check_p2.py --with-sim --device cuda:6
"""

import argparse
import math
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)                    # 让 `import tracking_bench` 能用
sys.path.insert(0, os.path.join(_PKG_ROOT, "examples"))   # minimal_example

import numpy as np

from tracking_bench.target.waypoint_policy import (
    WaypointPolicy,
    eval_quintic,
    peak_speed_accel,
)

DT = 0.1        # s/步，和 tracking_env 的 num_physics_steps_per_env_step_mean=10 对齐
N_STEPS = 600


def wall_surface_distance(p, walls):
    """点 p 到最近墙**表面**的距离（0 = 已经在墙里）。墙是 yaw 过的 OBB。"""
    best = float("inf")
    for w in walls:
        sx, sy, sz = w["size"]
        yaw = float(w["yaw"])
        dx, dy = p[0] - w["pos"][0], p[1] - w["pos"][1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        ox = max(abs(lx) - sx / 2.0, 0.0)
        oy = max(abs(ly) - sy / 2.0, 0.0)
        oz = max(-p[2], p[2] - sz, 0.0)
        best = min(best, math.sqrt(ox * ox + oy * oy + oz * oz))
    return best


def build_env_quiet(**kw):
    """建环境，期间把 `sys.argv` 藏起来，建完再还回去。

    ⚠️ 这不是洁癖，是踩过的坑：`aerial_gym.utils.helpers.get_args()` 会拿
    `argparse` 去解析 `sys.argv[1:]`，而 argparse **默认允许长选项缩写**。
    于是我们自己的 `--sim` 被当成 isaacgym 的 `--sim_device`，吞掉下一个参数后
    报 `argument --sim_device: expected one argument` 直接退出。
    凡是自己的参数别用 `--sim*` / `--headless` / `--num-envs` / `--pipeline` /
    `--flex` / `--physx` / `--use-warp` 这些 isaacgym 选项的前缀；更稳的是像这里
    一样把 argv 藏掉（`num_envs` / `headless` / `use_warp` 都是显式传参的，
    不依赖命令行）。
    """
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        from aerial_gym.sim.sim_builder import SimBuilder
        return SimBuilder().build_env(**kw)
    finally:
        sys.argv = saved


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-sim", dest="sim", action="store_true",
                   help="跑仿真 600 步那一项（别叫 --sim：会撞 isaacgym 的 --sim_device）")
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--out-dir", default="p2_check")
    args = p.parse_args()

    ck = Checker()
    pol = WaypointPolicy(seed=20261012)

    # ------------------------------------------------------------------ 1
    print("=" * 78, flush=True)
    print("1) 段边界连续性（数值差分）", flush=True)
    print("=" * 78, flush=True)
    # 先把轨迹续到够长，免得后面 state_at 又在续段
    pol.state_at(0.0)
    while pol.duration < 120.0:
        pol._extend(1)
    n_seg = len(pol._segments)
    print(f"  路点 {len(pol._pts)} 个，段 {n_seg} 段，总时长 {pol.duration:.1f} s",
          flush=True)

    # 段边界连续性：直接比**相邻两段多项式的端点值**。
    # ⚠️ 别用 state_at(t-h) vs state_at(t) 比距离 —— 那本来就是 h·v（≈1.2 mm @ h=1e-3），
    #    看着像跳变其实只是走了 h 秒。check_p2 第一版就是这么写错的。
    p_bad = v_bad = a_bad = 0
    worst_p = worst_v = worst_a = 0.0
    for i in range(1, n_seg):
        lo, hi = pol._segments[i - 1], pol._segments[i]
        p_end, v_end, a_end = eval_quintic(lo.coeffs, 1.0, lo.T)
        p_start, v_start, a_start = eval_quintic(hi.coeffs, 0.0, hi.T)

        dp = float(np.linalg.norm(p_end - p_start))
        dv = float(np.linalg.norm(v_end - v_start))
        da = max(float(np.linalg.norm(a_end)), float(np.linalg.norm(a_start)))
        worst_p = max(worst_p, dp)
        worst_v = max(worst_v, dv)
        worst_a = max(worst_a, da)
        if dp > 1e-9:
            p_bad += 1
        if dv > 1e-9:
            v_bad += 1
        if da > 1e-9:
            a_bad += 1

    ck.check("1a 位置连续", p_bad == 0,
             f"{n_seg - 1} 个段边界最大跳变 {worst_p:.2e} m")
    ck.check("1b 速度连续", v_bad == 0,
             f"最大跳变 {worst_v:.2e} m/s（vi = v_cruise·normalize(Pi+1−Pi−1)）")
    ck.check("1c 段边界加速度 = 0", a_bad == 0,
             f"最大 |a| = {worst_a:.2e} m/s²")

    # 再用差分交叉核对一次解析导数（防"公式对了但求导写错"）
    h = 1e-4
    worst_dv = 0.0
    for i in range(0, min(n_seg, 6)):
        seg = pol._segments[i]
        t0 = pol._cum_t[i] + 0.37 * seg.T
        _, v_an, _ = pol.state_at(t0)
        p_a, _, _ = pol.state_at(t0 - h)
        p_b, _, _ = pol.state_at(t0 + h)
        v_fd = [(b - a) / (2 * h) for a, b in zip(p_a, p_b)]
        worst_dv = max(worst_dv, math.dist(v_fd, v_an))
    ck.check("1d 解析速度 = 数值差分", worst_dv < 1e-5,
             f"最大偏差 {worst_dv:.2e} m/s（这一步专门抓 a1=vi·Ti 少乘/多乘 Ti）")

    # 1e 定稿的段不能被后续续段改写。
    # ⚠️ 这一条是"无瞬移"的前提，踩过：旧版每加一个路点就整体 _rebuild()，
    #    Catmull-Rom 把前面几段的形状和 _cum_t 一起挪了，播放中 state_at(t)
    #    前后不一致，仿真里目标跳 0.50 m（4.95 m/s）还会跳进墙。
    pol.state_at(0.0)
    while pol.duration < 30.0:
        pol._extend(1)
    probe = [i * DT for i in range(0, 120)]
    snap_a = [pol.state_at(t)[0] for t in probe]
    n_seg_before = len(pol._segments)
    while pol.duration < 200.0:
        pol._extend(1)
    snap_b = [pol.state_at(t)[0] for t in probe]
    drift = max(math.dist(a, b) for a, b in zip(snap_a, snap_b))
    ck.check("1e 定稿轨迹不被后续续段改写", drift == 0.0,
             f"续到 200 s（段数 {n_seg_before} -> {len(pol._segments)}）后，"
             f"前 12 s 的 {len(probe)} 个采样点最大位移 {drift:.2e} m")

    # ------------------------------------------------------------------ 2
    print("=" * 78, flush=True)
    print("2) 轨迹不出墙、不出场地", flush=True)
    print("=" * 78, flush=True)
    # 2a/2b 的判据是"不进墙、不离场"，用 policy 的曲线判据（净距 0.05 m），
    # 不是路点的 0.5 m —— 见 waypoint_policy._curve_ok 的说明。
    clear_fn = pol._curve_point_ok
    n_clear_fail = 0
    for seg in pol._segments:
        if not pol._curve_ok(seg):
            n_clear_fail += 1
    ck.check("2a 每一段五次曲线都不进墙不出界", n_clear_fail == 0,
             f"{len(pol._segments) - n_clear_fail}/{len(pol._segments)} 段通过"
             f"（0.05 m 采样，墙净距 {pol.curve_clearance} m）")

    # 600 步离散采样也要查（这才对应"跑 600 步目标全程不出墙"）
    n_hit = 0
    min_clear = 1e9
    for i in range(args.steps):
        pos, _, _ = pol.state_at(i * DT)
        if not clear_fn(pos):
            n_hit += 1
        min_clear = min(min_clear, wall_surface_distance(pos, pol.walls))
    ck.check("2b 600 步采样不出墙不出界", n_hit == 0,
             f"{args.steps - n_hit}/{args.steps} 步在自由空间，"
             f"离墙表面最近 {min_clear:.2f} m")

    # ------------------------------------------------------------------ 3
    print("=" * 78, flush=True)
    print("3) seed 可复现", flush=True)
    print("=" * 78, flush=True)
    a1 = WaypointPolicy(seed=7)
    a2 = WaypointPolicy(seed=7)
    b = WaypointPolicy(seed=8)
    ts = [i * DT for i in range(0, 200)]
    s1 = [a1.state_at(t) for t in ts]
    s2 = [a2.state_at(t) for t in ts]
    s3 = [b.state_at(t) for t in ts]
    same = all(
        math.dist(x[0], y[0]) == 0.0 and math.dist(x[1], y[1]) == 0.0
        for x, y in zip(s1, s2)
    )
    wp_same = a1.waypoints == a2.waypoints
    diff = any(math.dist(x[0], y[0]) > 1e-9 for x, y in zip(s1, s3))
    ck.check("3a 同 seed 轨迹逐点一致", same and wp_same,
             f"200 个采样点 + {len(a1.waypoints)} 个路点完全相同")
    ck.check("3b 换 seed 轨迹不同", diff,
             f"seed 7 vs 8 的 200 个采样点有差异")

    # ------------------------------------------------------------------ 4
    print("=" * 78, flush=True)
    print("4) 峰值速度 / 加速度不超限", flush=True)
    print("=" * 78, flush=True)
    pol2 = WaypointPolicy(seed=20261012)
    pol2.state_at(0.0)
    while pol2.duration < 120.0:
        pol2._extend(1)
    pv = pa = 0.0
    over_v = over_a = 0
    for seg in pol2._segments:
        v, a = peak_speed_accel(seg.coeffs, seg.T)
        pv = max(pv, v)
        pa = max(pa, a)
        if v > pol2.v_max * 1.001:
            over_v += 1
        if a > pol2.a_max * 1.001:
            over_a += 1
    ck.check("4a 峰值速度 ≤ v_max", over_v == 0,
             f"实测峰值 {pv:.3f} m/s，上限 {pol2.v_max} m/s，"
             f"{len(pol2._segments) - over_v}/{len(pol2._segments)} 段达标")
    ck.check("4b 峰值加速度 ≤ a_max", over_a == 0,
             f"实测峰值 {pa:.3f} m/s²，上限 {pol2.a_max} m/s²")

    # 段时长也要合理：过短的段说明 T_MIN 没生效
    Ts = [seg.T for seg in pol2._segments]
    ck.check("4c 段时长 ≥ T_MIN", min(Ts) >= pol2.t_min - 1e-9,
             f"T ∈ [{min(Ts):.2f}, {max(Ts):.2f}] s，T_MIN = {pol2.t_min}")

    # ------------------------------------------------------------------ 5
    if args.sim:
        print("=" * 78, flush=True)
        print(f"5) 仿真里跑 {args.steps} 步：目标动不动、瞬移不瞬移", flush=True)
        print("=" * 78, flush=True)
        ck_sim(ck, args)
    else:
        print("=" * 78, flush=True)
        print("5) 跳过（要 `--with-sim`）。数值部分已覆盖「动作连续」的等价条件",
              flush=True)
        print("=" * 78, flush=True)

    # ------------------------------------------------------------------
    print("=" * 78, flush=True)
    print("P2 验收汇总", flush=True)
    print("=" * 78, flush=True)
    for name, cond in ck.results:
        print(f"  {'✅' if cond else '❌'} {name}", flush=True)
    print(f"\n{'P2 通过' if ck.ok else 'P2 未通过'}", flush=True)
    return 0 if ck.ok else 1


def ck_sim(ck, args):
    """把 WaypointPolicy 接进仿真，写目标位置，跑 N 步。"""
    from tracking_bench.sim import gpu_lock
    gpu_lock.confine_to_one_gpu()  # noqa: F841  -- 必须在 torch 之前

    import isaacgym  # noqa: E402,F401
    import torch  # noqa: E402

    from tracking_bench import tracking_lib as tl  # noqa: E402

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":")[1]))

    env_manager = build_env_quiet(
        sim_name="base_sim",
        env_name="tracking_env",
        robot_name="base_quadrotor_with_stereo_camera",
        controller_name="lee_position_control",
        args=None,
        device=args.device,
        num_envs=1,
        headless=True,
        use_warp=True,
    )
    gtd = env_manager.global_tensor_dict
    state = gtd["env_asset_state_tensor"]
    n_env_assets = state.shape[1]

    # ⚠️ 必须 reset，否则 warp 相机位姿是 NaN（见 docs/SETUP_NOTES.md「阶段 2 踩过的坑」）
    env_manager.reset()
    tl.pin_static_layout(env_manager)

    pol = WaypointPolicy(seed=20261012)
    # ⚠️ 判据要和 2a/2b 一致：曲线只保证 `curve_clearance`（0.05 m）不进墙，
    #    不是路点的 `wall_clearance`（0.5 m）。拿 0.5 去卡曲线会**误报**穿墙 ——
    #    鼓肚本来就允许顶到 0.05 m。
    clear_fn = pol._curve_point_ok

    n_phys = env_manager.cfg.env.num_physics_steps_per_env_step_mean
    dt = n_phys * float(gtd["dt"])
    # 轨迹一次生成够整场再开播。append-only 之后中途续段也不会改旧段，
    # 但预生成让"飞的那条 == 校验那条"这件事变成同一条，省得再对时序。
    pol.state_at(0.0)
    while pol.duration < (args.steps + 2) * dt:
        pol._extend(1)

    actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
    tw = torch.zeros((env_manager.num_envs, n_env_assets, 6),
                     device=env_manager.device)

    # ⚠️ 追踪机要**钉住不动**，这一项只验目标的运动模型。
    #    actions 全 0 等于命令追踪机飞到 (0,0,0) —— 地面在 z=0，它会撞地 ->
    #    collision -> reset_terminated_and_truncated_envs() -> asset_manager.reset_idx()
    #    把目标位置重采样（min_state_ratio==max_state_ratio，所以是**瞬移到固定点**）。
    #    踩过的现象：5a 偏差 0.39 m、5c 最大步长 0.50 m（4.95 m/s），看着像目标抽风，
    #    其实是被 reset 挪走的。check_p1 的写法（actions[:, 0:3] = cpos0）才对。
    cpos0 = gtd["robot_position"][0].detach().clone()
    actions[:, 0:3] = cpos0
    actions[:, 3] = 0.0

    log = []
    cmd = []
    reset_steps = []
    for i in range(args.steps):
        t = i * dt
        tpos, tvel, _ = pol.state_at(t)
        cmd.append(list(tpos))
        state[:, tl.TARGET_SLOT, 0] = tpos[0]
        state[:, tl.TARGET_SLOT, 1] = tpos[1]
        state[:, tl.TARGET_SLOT, 2] = tpos[2]
        # 目标是 disable_gravity=True + fix_base_link=False。清速度是为了让物理
        # 在这一步里对它零作用 —— 脚本化轨迹就该位置说了算，别让积分去凑。
        state[:, tl.TARGET_SLOT, 7:13] = 0.0

        tw.zero_()
        env_manager.step(actions=actions, env_actions=tw)
        rids = env_manager.reset_terminated_and_truncated_envs()
        if len(rids) > 0:
            reset_steps.append(i)
            # reset_idx 会把资产位置重采样，必须重新钉，否则后面全是瞬移
            tl.pin_static_layout(env_manager)
        log.append(state[0, tl.TARGET_SLOT, 0:3].cpu().tolist())

    ck.check("5e 全程没有触发 env reset", len(reset_steps) == 0,
             f"reset 发生在第 {reset_steps[:10]} 步"
             f"{' 等' if len(reset_steps) > 10 else ''}"
             if reset_steps else "600 步无 reset（追踪机钉住 → 不撞地 → 不 reset）")

    steps = [math.dist(a, b) for a, b in zip(log[:-1], log[1:])]
    expected = [math.dist(pol.state_at(i * dt)[0],
                          pol.state_at((i + 1) * dt)[0])
                for i in range(args.steps - 1)]
    err = [abs(a - b) for a, b in zip(steps, expected)]
    pos_err = [math.dist(a, b) for a, b in zip(cmd, log)]
    worst = int(np.argmax(pos_err)) if pos_err else -1
    ck.check("5a 仿真里的目标位置 = state_at 的值",
             max(pos_err) < 1e-3,
             f"{args.steps} 步最大偏差 {max(pos_err):.2e} m"
             + (f"（最差在第 {worst} 步：cmd={cmd[worst]}, 实际={log[worst]}）"
                if worst >= 0 else ""))
    ck.check("5a' 步长也和 state_at 一致",
             (not err) or max(err) < 1e-3,
             f"最大步长偏差 {max(err) if err else 0.0:.2e} m")
    ck.check("5b 目标全程在动（没有停住）",
             min(steps) > 0.01,
             f"步长 ∈ [{min(steps):.3f}, {max(steps):.3f}] m"
             f"（dt={dt:.2f} s → {min(steps)/dt:.2f}~{max(steps)/dt:.2f} m/s）")
    ck.check("5c 无瞬移（步长不超过 v_max·dt 的 3 倍）",
             max(steps) < 3.0 * 1.2 * dt,
             f"最大步长 {max(steps):.3f} m，阈值 {3.0*1.2*dt:.3f} m")
    bad = [i for i, p in enumerate(log) if not clear_fn(p)]
    min_clear = min(wall_surface_distance(p, pol.walls) for p in log)
    ck.check("5d 目标不出墙不出界", not bad,
             f"{args.steps - len(bad)}/{args.steps} 步在自由空间，"
             f"离墙表面最近 {min_clear:.2f} m"
             + (f"，出界步 {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    sys.exit(main())
