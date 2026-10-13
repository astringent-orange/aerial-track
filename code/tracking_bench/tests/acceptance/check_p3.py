#!/usr/bin/env python3
"""P3 验收：目标真值输出。对照 docs/BENCH_PLAN.md §3 P3 的验收清单。

    1. 静止场景手工核对 bbox / center_pixel / projected_pixel，
       确认掩码质心和**针孔**几何投影点对齐（残差 < 3 px）
       —— docs/BENCH_PLAN.md 原写"确认 +16px u / +2px v 偏置仍在"，那条是误记，
       那个"偏置"是等距柱状公式和针孔渲染器之间的差，见 docs/SETUP_NOTES.md
    2. 目标放到墙后：in_fov=True, visible=False, occluded=True
    3. 目标转出画面：in_fov=False
    4. 目标在近处（< 0.2m）时 target_depth_mean 反映 `-1.0` 语义，不报错

每个场景都是"把追踪机命令到一个固定位姿 + 把目标写到一个固定位置"，
然后渲染一帧、读 `warp_sensor` 的**真实**相机位姿、交给 `truth.compute_truth`。

⚠️ 投影是拿真实相机位姿算的，所以追踪机 settle 得准不准都不影响判据 ——
   这也正是 `truth.py` 该有的用法。

用法（容器内，先 cd /workspace）：
    python check_p3.py --device cuda:2
"""

import argparse
import math
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)                    # 让 `import tracking_bench` 能用
sys.path.insert(0, os.path.join(_PKG_ROOT, "examples"))   # minimal_example

from tracking_bench.sim import gpu_lock
gpu_lock.confine_to_one_gpu()  # 必须在 import torch / aerial_gym 之前

import isaacgym  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aerial_gym.sim.sim_builder import SimBuilder  # noqa: E402
from aerial_gym.utils.math import get_euler_xyz_tensor  # noqa: E402

from tracking_bench import tracking_lib as tl  # noqa: E402
from tracking_bench.sensors import truth as T  # noqa: E402


def build_env_quiet(**kw):
    """建环境，期间把 `sys.argv` 藏起来。原因见 check_p2.build_env_quiet。"""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
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


def settle_and_render(ctx, chaser_pos, chaser_yaw, target_pos, n_steps=60,
                      target_rel_to_cam=None):
    """把追踪机命令到 (chaser_pos, chaser_yaw)，目标写到 target_pos，跑 n_steps，
    最后渲染一帧并算真值。

    target_rel_to_cam: 给了就忽略 target_pos，改成"相对**真实**相机位姿"的偏移
    （场景 D 用 —— 目标要贴着相机，必须跟着相机走，不然会被推开）。
    """
    env_manager, gtd, state, actions = (
        ctx["env_manager"], ctx["gtd"], ctx["state"], ctx["actions"])
    n_phys = env_manager.cfg.env.num_physics_steps_per_env_step_mean
    dt = n_phys * float(gtd["dt"])

    actions[:, 0] = chaser_pos[0]
    actions[:, 1] = chaser_pos[1]
    actions[:, 2] = chaser_pos[2]
    actions[:, 3] = chaser_yaw

    for _ in range(n_steps):
        if target_rel_to_cam is None:
            tpos = target_pos
        else:
            ws = env_manager.robot_manager.warp_sensor
            cp = ws.sensor_position[0, 0].detach().cpu().numpy()
            R = T.quat_to_mat(ws.sensor_orientation[0, 0])
            tpos = cp + R @ np.asarray(target_rel_to_cam, dtype=np.float64)
        state[:, tl.TARGET_SLOT, 0] = float(tpos[0])
        state[:, tl.TARGET_SLOT, 1] = float(tpos[1])
        state[:, tl.TARGET_SLOT, 2] = float(tpos[2])
        # 清速度：目标是 disable_gravity=True + fix_base_link=False，
        # 位置说了算，别让积分去凑（同 check_p2）
        state[:, tl.TARGET_SLOT, 7:13] = 0.0
        env_manager.step(actions=actions, env_actions=ctx["tw"])
        # ⚠️ 这里**故意不调** reset_terminated_and_truncated_envs：
        #    场景 D 的目标贴着相机会撞，reset_idx 会把全部资产重采样
        #    （见 docs/SETUP_NOTES.md「别把追踪机指到 (0,0,0)」）。每步重写目标就够了。

    env_manager.render(render_components="sensors")
    seg = gtd["segmentation_pixels"][0, 0].detach().cpu().numpy()
    depth = gtd["depth_range_pixels"][0, 0].detach().cpu().numpy()
    ws = env_manager.robot_manager.warp_sensor
    cam_pos = ws.sensor_position[0, 0].detach().cpu().numpy()
    cam_R = T.quat_to_mat(ws.sensor_orientation[0, 0])
    tpos = state[0, tl.TARGET_SLOT, 0:3].detach().cpu().numpy()

    tr = T.compute_truth(seg, depth, tpos, cam_pos, cam_R=cam_R)
    return tr, seg, depth, cam_pos, cam_R, tpos, dt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--settle", type=int, default=60, help="每个场景前稳多少步")
    p.add_argument("--out-dir", default="p3_check")
    args = p.parse_args()

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":")[1]))

    os.makedirs(args.out_dir, exist_ok=True)

    env_manager = build_env_quiet(
        sim_name="base_sim",
        env_name="tracking_env",
        robot_name="base_quadrotor_with_stereo_camera",
        controller_name="lee_position_control",
        args=None,
        device=args.device,
        num_envs=1,
        headless=True,
        use_warp=True,  # depth/seg，单卡
    )
    gtd = env_manager.global_tensor_dict
    state = gtd["env_asset_state_tensor"]
    n_env_assets = state.shape[1]

    # ⚠️ 必须 reset，否则 warp 相机位姿是 NaN（见 docs/SETUP_NOTES.md）
    env_manager.reset()
    tl.pin_static_layout(env_manager)

    actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
    tw = torch.zeros((env_manager.num_envs, n_env_assets, 6),
                     device=env_manager.device)
    ctx = dict(env_manager=env_manager, gtd=gtd, state=state,
               actions=actions, tw=tw)

    ck = Checker()
    print("=" * 92, flush=True)
    print("P3 验收：目标真值输出", flush=True)
    print("=" * 92, flush=True)

    # ------------------------------------------------------------------ 1
    # 静止场景：追踪机在原点上方、机头朝 +x，目标在正前方几个距离/方位上。
    print("=" * 92, flush=True)
    print("1) 静止场景：bbox / center_pixel / projected_pixel 手工核对 + 质心偏置",
          flush=True)
    print("=" * 92, flush=True)
    CH = (0.0, 0.0, 2.5)
    CH_YAW = 0.0     # 机头朝 +x，相机光轴也朝 +x
    # 目标放在视场中间偏几个位置，让偏置的稳定性看得出来
    targets = [
        ("正前方 3.0m",        (3.0, 0.0, 2.5)),
        ("正前方 5.0m",        (5.0, 0.0, 2.5)),
        ("偏左 1.2m @4.0m",    (4.0, 1.2, 2.3)),
    ]
    print(f"  追踪机固定在 {CH}，yaw={math.degrees(CH_YAW):.0f}°", flush=True)
    print(f"  {'场景':<18} {'距离':>6} {'面积':>6} {'bbox':>16} "
          f"{'针孔 u,v':>14} {'等距柱状 u':>11} {'质心 u,v':>14} "
          f"{'残差(针孔)':>12}", flush=True)
    rows = []
    last = None
    for label, tp in targets:
        tr, seg, depth, cam_pos, cam_R, tpos, dt = settle_and_render(
            ctx, CH, CH_YAW, tp, n_steps=args.settle)
        last = (tr, seg, depth, cam_pos, cam_R, tpos)
        u_eq, v_eq, _, _ = T.project_equirect(cam_pos, cam_R, tpos)
        du = dv = float("nan")
        if tr["center_pixel"] is not None:
            du = tr["center_pixel"][0] - tr["projected_pixel"][0]
            dv = tr["center_pixel"][1] - tr["projected_pixel"][1]
        rows.append((label, tr, du, dv, u_eq))
        bb = ("%d,%d,%d,%d" % tr["bbox"]) if tr["bbox"] else "-"
        cp = ("%.1f,%.1f" % tr["center_pixel"]) if tr["center_pixel"] else "-"
        pp = "%.1f,%.1f" % tr["projected_pixel"]
        print(f"  {label:<18} {tr['distance']:>6.2f} "
              f"{tr['target_pixel_area']:>6} {bb:>16} {pp:>14} {u_eq:>11.1f} "
              f"{cp:>14} {du:>6.2f},{dv:<5.2f}", flush=True)

    tr, seg, depth, cam_pos, cam_R, tpos = last
    ck.check("1a 静止场景目标 in_fov + visible（没被挡）",
             tr["in_fov"] and tr["visible"] and not tr["occluded"],
             f"in_fov={tr['in_fov']} visible={tr['visible']} "
             f"occluded={tr['occluded']}")
    ck.check("1b bbox / center_pixel / projected_pixel 都有值",
             tr["bbox"] is not None and tr["center_pixel"] is not None
             and tr["projected_pixel"] is not None,
             f"bbox={tr['bbox']} center={tr['center_pixel']} "
             f"proj={tr['projected_pixel']}")
    # bbox 必须把整块掩码包住，中心点必须落在 bbox 里
    if tr["bbox"] is not None and tr["center_pixel"] is not None:
        u0, v0, u1, v1 = tr["bbox"]
        cu, cv = tr["center_pixel"]
        ck.check("1c center_pixel 落在 bbox 内",
                 u0 <= cu <= u1 and v0 <= cv <= v1,
                 f"center=({cu:.1f},{cv:.1f}) bbox={tr['bbox']}")
    else:
        ck.check("1c center_pixel 落在 bbox 内", False, "掩码为空")

    # 1d：投影用**针孔**，掩码质心就该和它对齐到 1 px 以内。
    # ⚠️ docs/BENCH_PLAN.md 原来写的是"确认 +16px u / +2px v 偏置仍在"，那是**误记** ——
    #    那个"偏置"是等距柱状公式和针孔渲染器之间的差，离轴越大、光轴上为 0。
    #    换成针孔后残差应为 0 上下的网格噪声。上面那张表同时打出两条公式供核对。
    biases = [(r[2], r[3]) for r in rows if not math.isnan(r[2])]
    bu = float(np.mean([b[0] for b in biases])) if biases else float("nan")
    bv = float(np.mean([b[1] for b in biases])) if biases else float("nan")
    su = float(np.std([b[0] for b in biases])) if biases else float("nan")
    sv = float(np.std([b[1] for b in biases])) if biases else float("nan")
    worst_u = max(abs(b[0]) for b in biases) if biases else float("nan")
    worst_v = max(abs(b[1]) for b in biases) if biases else float("nan")
    ck.check("1d 掩码质心 = 针孔几何投影（残差 < 3 px）",
             len(biases) >= 2 and worst_u < T.CENTROID_RESIDUAL_TOL_U
             and worst_v < T.CENTROID_RESIDUAL_TOL_V,
             f"{len(biases)} 个场景残差均值 ({bu:+.2f}, {bv:+.2f}) px，"
             f"最差 ({worst_u:.2f}, {worst_v:.2f}) px，"
             f"std ({su:.2f}, {sv:.2f}) px —— "
             f"旧的 +16px u '偏置'是等距柱状公式用错了，不是渲染偏置")

    # ------------------------------------------------------------------ 2
    print("=" * 92, flush=True)
    print("2) 目标在墙后：in_fov=True, visible=False, occluded=True", flush=True)
    print("=" * 92, flush=True)
    # wall_04 在 (0, 5.5)，2.0 x 0.3 x 3.0，z∈[0,3]。追踪机在原点朝 +y，
    # 目标放到墙外 (0, 8, 2.5) —— 视线穿过墙体，被完全挡住。
    CH2 = (0.0, 0.0, 2.5)
    CH2_YAW = math.pi / 2.0      # 机头朝 +y
    TP2 = (0.0, 8.0, 2.5)
    tr2, seg2, depth2, cam_pos2, cam_R2, tpos2, _ = settle_and_render(
        ctx, CH2, CH2_YAW, TP2, n_steps=args.settle)
    print(f"  {T.summarize(tr2)}", flush=True)
    print(f"  掩码 {tr2['target_pixel_area']} px，"
          f"该帧里非 miss/非地板的像素 "
          f"{int(((seg2 != -2) & (seg2 != T.TARGET_SEG_ID) & (seg2 != 13)).sum())}",
          flush=True)
    ck.check("2a 墙后：in_fov=True（几何上在视场里）", tr2["in_fov"],
             f"in_fov={tr2['in_fov']}（目标方位 "
             f"{math.degrees(math.atan2(TP2[1] - cam_pos2[1], TP2[0] - cam_pos2[0])):.0f}°，"
             f"相机朝向 {math.degrees(math.atan2(cam_R2[0, 1], cam_R2[0, 0])):.0f}°）")
    ck.check("2b 墙后：visible=False（分割图里没目标像素）",
             not tr2["visible"] and tr2["target_pixel_area"] == 0,
             f"visible={tr2['visible']} area={tr2['target_pixel_area']}")
    ck.check("2c 墙后：occluded=True", tr2["occluded"],
             f"occluded={tr2['occluded']}（定义 = in_fov and not visible）")

    # ------------------------------------------------------------------ 3
    print("=" * 92, flush=True)
    print("3) 目标转出画面：in_fov=False", flush=True)
    print("=" * 92, flush=True)
    # 同一个目标位置，但追踪机机头朝 +x，目标在它左侧 90°，远在 43.5° 半视场外
    CH3 = (0.0, 0.0, 2.5)
    CH3_YAW = 0.0                 # 机头朝 +x
    TP3 = (0.0, 8.0, 2.5)         # 在 +y 方向，方位 90°
    tr3, seg3, depth3, cam_pos3, cam_R3, tpos3, _ = settle_and_render(
        ctx, CH3, CH3_YAW, TP3, n_steps=args.settle)
    print(f"  {T.summarize(tr3)}", flush=True)
    ang = math.degrees(math.atan2(TP3[1] - cam_pos3[1], TP3[0] - cam_pos3[0]))
    cam_yaw = math.degrees(math.atan2(cam_R3[0, 1], cam_R3[0, 0]))
    rel = (ang - cam_yaw + 180.0) % 360.0 - 180.0
    ck.check("3a 出画：in_fov=False", not tr3["in_fov"],
             f"in_fov={tr3['in_fov']}（目标方位 {ang:.0f}°，相机朝向 {cam_yaw:.0f}°，"
             f"夹角 {rel:.0f}°，半视场 {T.HFOV_HALF}°）")
    ck.check("3b 出画：occluded 也不该为真（没进画就谈不上被挡）",
             not tr3["occluded"],
             f"occluded={tr3['occluded']}")

    # ------------------------------------------------------------------ 4
    print("=" * 92, flush=True)
    print("4) 目标贴着相机（< 0.2m）：target_depth_mean 反映 -1.0，不报错",
          flush=True)
    print("=" * 92, flush=True)
    # 目标是半径 0.30、高 0.35 的圆柱（原点在中心，轴沿 z）。相机光轴是水平的，
    # 所以沿光轴方向目标的**包络是直径 0.60**，不是高度 0.35 ——
    # 中心放到 0.42 m 处，朝相机的那块表面就在 0.42 - 0.30 = **0.12 m**
    # < MIN_RANGE(0.2)，于是那些像素写 -1.0。
    # ⚠️ 偏移是**相机系**的 (0, 0, d)：OpenCV 约定下光轴是 **cam-z**，
    #    不是 cam-x（第一版写成 (0.35, 0, 0)，目标跑到相机右边 90°，直接出画）。
    # ⚠️ 相对**真实相机位姿**放，每步重放一次，跟着相机走。
    D_NEAR = 0.42
    tr4, seg4, depth4, cam_pos4, cam_R4, tpos4, _ = settle_and_render(
        ctx, CH3, CH3_YAW, None, n_steps=args.settle,
        target_rel_to_cam=(0.0, 0.0, D_NEAR))
    near_surface = tr4["distance"] - 0.30   # 圆柱半径
    print(f"  {T.summarize(tr4)}", flush=True)
    print(f"  目标中心离相机 {tr4['distance']:.3f} m，"
          f"朝相机的表面离相机 {near_surface:.3f} m "
          f"（MIN_RANGE = {T.MIN_RANGE} m）", flush=True)
    if tr4["target_pixel_area"] > 0:
        m = seg4 == T.TARGET_SEG_ID
        dvals = depth4[m]
        n_neg = int((dvals < -0.5).sum())
        print(f"  掩码内深度：min={dvals.min():.3f} max={dvals.max():.3f} "
              f"mean={dvals.mean():.3f}，其中 {n_neg}/{dvals.size} 个是 -1.0",
              flush=True)
        ck.check("4a 近处不报错，且拿到了掩码和深度均值",
                 tr4["target_depth_mean"] is not None
                 and near_surface < T.MIN_RANGE,
                 f"target_depth_mean={tr4['target_depth_mean']:.3f}，"
                 f"表面距离 {near_surface:.3f} m < {T.MIN_RANGE} m")
        ck.check("4b target_depth_mean 反映 `-1.0` 语义（不是丢掉/裁成 0）",
                 tr4["target_depth_mean"] is not None
                 and tr4["target_depth_mean"] < -0.2,
                 f"{tr4['target_depth_mean']:.3f} < -0.2 —— "
                 f"掩码里 {n_neg}/{dvals.size} 个像素是 -1.0（比 {T.MIN_RANGE} m 还近）")
        ck.check("4c in_fov/visible 照常给，不因为深度是 -1 就崩",
                 tr4["in_fov"] and tr4["visible"],
                 f"in_fov={tr4['in_fov']} visible={tr4['visible']} "
                 f"area={tr4['target_pixel_area']}")
    else:
        ck.check("4a 近处不报错，且拿到了掩码和深度均值", False,
                 f"掩码为空（visible={tr4['visible']}），"
                 f"distance={tr4['distance']:.3f} m，表面 {near_surface:.3f} m")
        ck.check("4b target_depth_mean 反映 `-1.0` 语义（不是丢掉/裁成 0）",
                 False, "拿不到掩码，无从判断")
        ck.check("4c in_fov/visible 照常给，不因为深度是 -1 就崩",
                 tr4["in_fov"] and tr4["visible"],
                 f"in_fov={tr4['in_fov']} visible={tr4['visible']}")

    # ------------------------------------------------------------------
    print("=" * 92, flush=True)
    print("P3 验收汇总", flush=True)
    print("=" * 92, flush=True)
    for name, cond in ck.results:
        print(f"  {'✅' if cond else '❌'} {name}", flush=True)
    print(f"\n{'P3 通过' if ck.ok else 'P3 未通过'}", flush=True)
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main())
