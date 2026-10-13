#!/usr/bin/env python3
"""P1 验收：20x20 场地 + 薄墙。对照 docs/BENCH_PLAN.md §3 P1 的验收清单。

逐条检查：
    1. 资产数量对，槽位表和预期一致
    2. 墙立在地面上（z=0，原点在底面）
    3. warp 深度/分割图里能看到墙（不是隐形的）
    4. 跑 600 步墙一动不动
    5. 目标（现行圆轨迹，半径 2）不会穿墙

用法（容器内，先 cd /workspace）：
    python check_p1.py --device cuda:6
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

import isaacgym  # noqa: E402,F401  -- IsaacGym insists on being imported before torch
import logging  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aerial_gym.sim.sim_builder import SimBuilder  # noqa: E402
from aerial_gym.config.env_config.tracking_env import ASSET_SLOT_ORDER  # noqa: E402
from tracking_bench import tracking_lib as tl  # noqa: E402
from tracking_bench.sensors import truth  # noqa: E402  -- 投影公式统一走这里（针孔）

# 目标物的分割实例 id（槽 0、第一个被分配到的，见 tracking_lib 模块头）
TARGET_SEG_ID = 100
# 地板 bottom_wall 用的是固定 semantic id，不是递增的 101
FLOOR_SEG_ID = 13

IMG_W, IMG_H = 480, 270
HFOV_HALF = 43.5  # deg
VFOV_HALF = 28.1  # deg


def _quiet(name):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    for h in lg.handlers:
        h.setLevel(logging.ERROR)


def build_env_quiet(**kw):
    """建环境，期间把 `sys.argv` 藏起来。原因见 check_p2.build_env_quiet 的注释
    （isaacgym 的 gymutil 会解析我们的命令行，长选项缩写会撞它的 `--sim_device`）。"""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        return SimBuilder().build_env(**kw)
    finally:
        sys.argv = saved


def depth_to_uint8(frame):
    """深度帧 -> (H, W, 3) uint8，近黑远白。值域 [-1] ∪ [0,1]，clip 即可。"""
    d = np.clip(frame, 0.0, 1.0) * 255.0
    return np.repeat(d.astype(np.uint8)[..., None], 3, axis=2)


def quat_to_mat(q, device=None):
    """(x, y, z, w) -> 3x3，列向量是相机三根轴在世界系的表示。"""
    x, y, z, w = [float(v) for v in q]
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        device=device,
    )


def project(cam_pos, cam_R, p_world):
    """世界点 -> (u, v, depth_norm)。委托给 `truth.project`（**针孔**）。

    ⚠️ 这里以前是等距柱状 `u = 240 + angle/43.5*240`，和 warp 的针孔渲染器
    中间能差 16 px。现在统一走 `truth.project`，别再各写一份。
    """
    u, v, depth_norm, _cz = truth.project(cam_pos, cam_R, p_world)
    return u, v, depth_norm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0", help="物理卡号，如 cuda:6")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out-dir", default="p1_check", help="产物目录")
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
    _quiet("env_manager")

    state = gtd["env_asset_state_tensor"]
    # env_actions 的宽度 = env_asset_state_tensor 的资产数。
    # ⚠️ 这个张量是 vec_root_tensor[:, 1:, :]，**已经**排掉了机器人，
    # 所以不要再 -1（多减一次的症状是 obstacle_manager 里
    # "expanded size of the tensor (16) must match ... (15)"，不报警告）。
    n_env_assets = state.shape[1]
    results = []

    def check(name, cond, detail):
        results.append((name, bool(cond), detail))
        print(f"  {'✅' if cond else '❌'} {name}: {detail}", flush=True)

    # ---------- 1. 槽位 ----------
    print("=" * 78, flush=True)
    print("1) 槽位核对", flush=True)
    print("=" * 78, flush=True)
    n_assets = state.shape[1]
    expected = 1 + 1 + tl.N_WALLS + tl.N_CYLINDERS  # target + floor + walls + cylinders
    print(f"  num_obstacles_in_env = {gtd['num_obstacles_in_env']}", flush=True)
    print(f"  state 张量形状        = {tuple(state.shape)}", flush=True)
    print(f"  槽位表（前 {min(12, len(ASSET_SLOT_ORDER))} 项）:", flush=True)
    for slot, (key, inst) in enumerate(ASSET_SLOT_ORDER[:12]):
        mark = ""
        if key == "target":
            mark = "  <- 驱动代码写 twist[:, 0, :]"
        elif key == "bottom_wall":
            mark = "  <- 地板 mesh"
        print(f"    槽 {slot:2d}  {key}[{inst}]{mark}", flush=True)
    check("1a 资产总数", n_assets == expected,
          f"state.shape[1] = {n_assets}（要 {expected} = 1 目标 + 1 地板 + "
          f"{tl.N_WALLS} 墙 + {tl.N_CYLINDERS} 圆柱）")
    check("1b 目标在槽 0", tl.TARGET_SLOT == 0 and ASSET_SLOT_ORDER[0][0] == "target",
          f"ASSET_SLOT_ORDER[0] = {ASSET_SLOT_ORDER[0]}")
    check("1c 地板在槽 1", ASSET_SLOT_ORDER[1][0] == "bottom_wall",
          f"ASSET_SLOT_ORDER[1] = {ASSET_SLOT_ORDER[1]}")
    walls_ok = all(ASSET_SLOT_ORDER[tl.wall_slot(i)][0] == f"wall_{i:02d}"
                   for i in range(tl.N_WALLS))
    check("1d 墙槽位 = YAML 逆序",
          walls_ok,
          f"wall_slot(i) = 2 + ({tl.N_WALLS}-1-i)，槽 2..{1 + tl.N_WALLS}")

    # ---------- 2. 摆好 + 墙的位姿 ----------
    print("=" * 78, flush=True)
    print("2) 布局（pin_static_layout 之后）", flush=True)
    print("=" * 78, flush=True)
    # ⚠️ 必须先 reset() 再 pin。少了 reset() 的症状（踩过）：
    #    warp 相机的 sensor_position/orientation 是 NaN，分割图整帧 -2（没打中），
    #    而且**不报错** —— tf_apply 吃了未初始化的机器人四元数。
    #    run_tracking.py 里是 `env_manager.reset(); tl.pin_cylinder_layout(...)`。
    env_manager.reset()
    tl.pin_static_layout(env_manager)
    st = state[0].cpu().numpy()

    print(f"  {'#':>3} {'槽':>3} {'x':>7} {'y':>7} {'z':>6} {'yaw':>7} "
          f"{'|size|':>14}", flush=True)
    z_bad = 0
    pos_bad = 0
    for i, w in enumerate(tl.WALL_LAYOUT):
        s = tl.wall_slot(i)
        x, y, z = st[s, 0:3]
        # 四元数 (x,y,z,w) -> yaw
        qw, qx, qy, qz = st[s, 6], st[s, 3], st[s, 4], st[s, 5]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        sx, sy, sz = w["size"]
        print(f"  wall_{i:02d} {s:3d} {x:7.2f} {y:7.2f} {z:6.2f} {yaw:7.3f} "
              f"{sx:5.2f}x{sy:4.2f}x{sz:4.2f}", flush=True)
        if abs(z) > 1e-4:
            z_bad += 1
        if math.hypot(x - w["pos"][0], y - w["pos"][1]) > 1e-3 or \
           abs(yaw - float(w["yaw"])) > 1e-3:
            pos_bad += 1
    check("2a 墙 z=0（立在地面，不陷地/不悬空）", z_bad == 0,
          f"{tl.N_WALLS - z_bad}/{tl.N_WALLS} 面墙 z=0。"
          f"URDF 原点在底面（make_wall_assets.py），所以 z=0 就是贴地")
    check("2b 墙位姿 = YAML", pos_bad == 0,
          f"{tl.N_WALLS - pos_bad}/{tl.N_WALLS} 面墙 (x,y,yaw) 和 YAML 一致")

    cyl_sl = slice(tl.CYLINDER_SLOT_START, tl.CYLINDER_SLOT_END)
    cyl_ok = all(
        math.hypot(st[tl.CYLINDER_SLOT_START + i, 0] - xy[0],
                   st[tl.CYLINDER_SLOT_START + i, 1] - xy[1]) < 1e-3
        for i, xy in enumerate(tl.CYLINDER_XY)
    )
    check("2c 圆柱在外围固定布局", cyl_ok,
          f"6 根在半径 8.5 处，槽 {tl.CYLINDER_SLOT_START}..{tl.CYLINDER_SLOT_END - 1}")

    # ---------- 3. 渲染：墙看不看得见 ----------
    print("=" * 78, flush=True)
    print("3) warp 渲染：墙是不是隐形的", flush=True)
    print("=" * 78, flush=True)
    env_manager.step(
        actions=torch.zeros((env_manager.num_envs, 4), device=env_manager.device),
        env_actions=torch.zeros((env_manager.num_envs, n_env_assets, 6),
                                device=env_manager.device),
    )
    env_manager.render(render_components="sensors")

    depth = gtd["depth_range_pixels"][0, 0].cpu().numpy()
    seg = gtd["segmentation_pixels"][0, 0].cpu().numpy()

    import imageio as _io
    _io.imwrite(os.path.join(args.out_dir, "depth.png"), depth_to_uint8(depth))

    # 分割图着色：目标亮红、地板灰、墙/圆柱各一色
    lut = np.array(
        [[20, 20, 20], [255, 60, 60], [80, 80, 80], [70, 130, 220],
         [70, 200, 130], [230, 200, 70], [200, 90, 200], [90, 210, 210],
         [240, 140, 60]], dtype=np.uint8)
    ids = [int(s) for s in np.unique(seg)]
    cmap = {s: lut[k % len(lut)] for k, s in enumerate(ids)}
    if TARGET_SEG_ID in cmap:
        cmap[TARGET_SEG_ID] = np.array([255, 30, 30], dtype=np.uint8)
    if FLOOR_SEG_ID in cmap:
        cmap[FLOOR_SEG_ID] = np.array([70, 70, 70], dtype=np.uint8)
    rgb = np.zeros(seg.shape + (3,), dtype=np.uint8)
    for s, c in cmap.items():
        rgb[seg == s] = c
    _io.imwrite(os.path.join(args.out_dir, "seg.png"), rgb)

    hist = {s: int((seg == s).sum()) for s in ids}
    print(f"  seg id 直方图（像素数）：{ {k: hist[k] for k in sorted(hist)} }",
          flush=True)

    # ---- 3a：墙在 warp 分割图里不是隐形的 -----------------------------
    # ⚠️ 判据别做成"每面墙在自己投影窗口里取众数 id"。相机初始位姿是随机的，
    #    多数墙要么投影出画，要么超出 10 m 量程打成 miss（+1.0 分不开超程/未命中），
    #    窗口众数自然是 -2 —— 那量的是相机朝向，不是渲染。
    #    踩过：整帧明明有 3 个墙实例 id（103/104/107，共 15890 px），
    #    窗口法只认出 1 个 id 就判失败。真正要证的是"渲染看得见墙"，
    #    所以直接数**整帧里有几个墙实例 id**，和位姿无关。
    wall_inst_ids = {s for s in ids
                     if s not in (-2, TARGET_SEG_ID, FLOOR_SEG_ID) and hist[s] > 30}
    check("3a 至少 3 个墙实例在 warp 分割图里看得见",
          len(wall_inst_ids) >= 3,
          f"整帧里非目标/非地板/非 miss 的实例 id = {sorted(wall_inst_ids)}"
          f"（像素数 {[hist[s] for s in sorted(wall_inst_ids)]}）")

    # 每面墙：投影到像素，在一个小窗口里取众数 id —— **诊断表，不作验收**。
    # 这里只回答"我这次相机朝向看到了哪几面墙"，换一次 reset 结论就变。
    ws = env_manager.robot_manager.warp_sensor
    pcam = ws.sensor_position[0, 0].cpu().numpy()
    Rcam = quat_to_mat(ws.sensor_orientation[0, 0]).numpy()
    print(f"  {'墙':>8} {'期望 u,v':>12} {'期望深度':>8} {'窗口众数 id':>12} "
          f"{'实测深度':>8}  诊断", flush=True)
    in_view = 0
    for i, w in enumerate(tl.WALL_LAYOUT):
        sx, sy, sz = w["size"]
        x, y = w["pos"]
        # 墙的几何中心（原点在底面 -> 中心在 z = sz/2）
        p = [x, y, sz / 2.0]
        u, v, d_exp = project(pcam, Rcam, p)
        in_img = 0 <= u < IMG_W and 0 <= v < IMG_H
        in_range = 0.02 < d_exp < 0.95      # 在 10 m 量程内（超程会打成 miss）
        mode_id, mode_n, d_meas = -2, 0, float("nan")
        if in_img:
            u0, u1 = max(0, int(u) - 25), min(IMG_W, int(u) + 26)
            v0, v1 = max(0, int(v) - 25), min(IMG_H, int(v) + 26)
            patch = seg[v0:v1, u0:u1]
            vals, cnts = np.unique(patch, return_counts=True)
            k = int(np.argmax(cnts))
            mode_id, mode_n = int(vals[k]), int(cnts[k])
            dpatch = depth[v0:v1, u0:u1]
            hit = (patch == mode_id) & (dpatch > -0.5) & (dpatch < 0.999)
            d_meas = float(dpatch[hit].mean()) if hit.any() else float("nan")
        if not in_img:
            tag = "画外"
        elif not in_range:
            tag = "超程(>10m)" if d_exp >= 0.95 else "太近/背后"
        elif mode_id in (-2, TARGET_SEG_ID, FLOOR_SEG_ID):
            tag = "窗口被别的面盖住"
        else:
            tag = f"命中 id={mode_id}"
            in_view += 1
        print(f"  wall_{i:02d} {u:6.0f},{v:4.0f} {d_exp:8.2f} "
              f"{mode_id:8d}({mode_n:3d}) {d_meas:8.2f}  {tag}", flush=True)
    print(f"  （诊断）本次相机朝向下，画内且在量程内、窗口能对上实例 id 的墙 "
          f"{in_view} 面 —— 这一张表跟 reset 位姿走，不作验收", flush=True)
    near_floor = hist.get(FLOOR_SEG_ID, 0)
    check("3b 地板也在图里（warp 看得见 bottom_wall mesh）", near_floor > 1000,
          f"地板像素 {near_floor}")
    miss = hist.get(-2, 0)
    check("3c 画面不是全空", miss < IMG_W * IMG_H * 0.9,
          f"没打中的像素 {miss}/{IMG_W * IMG_H}（{miss / (IMG_W * IMG_H) * 100:.0f}%）")

    # ---------- 4. 跑 N 步，墙动不动 ----------
    print("=" * 78, flush=True)
    print(f"4) 跑 {args.steps} 步，墙/圆柱会不会漂", flush=True)
    print("=" * 78, flush=True)
    tl.pin_static_layout(env_manager)
    ref = state[0].cpu().numpy().copy()  # 只比墙和圆柱那几槽

    actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
    cpos0 = gtd["robot_position"][0].detach().clone().tolist()
    actions[:, 0:3] = torch.tensor(cpos0, device=env_manager.device)
    tw = torch.zeros((env_manager.num_envs, n_env_assets, 6),
                     device=env_manager.device)
    traj = tl.CircleTrajectory()
    n_phys = env_manager.cfg.env.num_physics_steps_per_env_step_mean
    dt = n_phys * float(gtd["dt"])

    cprev = list(cpos0)
    tgt_log = []
    for i in range(args.steps):
        t = i * dt
        tpos, tvel, _ = traj.state_at(t)
        state[:, tl.TARGET_SLOT, 0] = tpos[0]
        state[:, tl.TARGET_SLOT, 1] = tpos[1]
        state[:, tl.TARGET_SLOT, 2] = tpos[2]
        state[:, tl.TARGET_SLOT, 7:13] = 0.0

        cp = gtd["robot_position"][0].tolist()
        desired, yaw_cmd = tl.chaser_follow(tpos, tvel, cp)
        desired = tl.limit_speed(cprev, desired, dt)
        cprev = list(desired)
        actions[:, 0] = desired[0]
        actions[:, 1] = desired[1]
        actions[:, 2] = desired[2]
        actions[:, 3] = yaw_cmd

        tw.zero_()
        env_manager.step(actions=actions, env_actions=tw)
        if i % 50 == 0:
            env_manager.render(render_components="sensors")

        rids = env_manager.reset_terminated_and_truncated_envs()
        if len(rids) > 0:
            tl.pin_static_layout(env_manager)

        tgt_log.append(state[0, tl.TARGET_SLOT, 0:3].cpu().tolist())

    now = state[0].cpu().numpy()
    static_slots = list(range(tl.BOTTOM_WALL_SLOT, tl.CYLINDER_SLOT_END))
    drift = np.abs(now[static_slots, 0:3] - ref[static_slots, 0:3]).max()
    check("4a 墙 + 圆柱 + 地板全程不动", drift < 1e-3,
          f"{len(static_slots)} 个静态槽位的最大位移 = {drift:.2e} m")

    # ---------- 5. 目标穿不穿墙 ----------
    print("=" * 78, flush=True)
    print("5) 目标（圆轨迹 r=2）会不会穿墙", flush=True)
    print("=" * 78, flush=True)
    min_clear = 1e9
    worst = None
    for px, py, pz in tgt_log:
        for i, w in enumerate(tl.WALL_LAYOUT):
            sx, sy, sz = w["size"]
            yaw = float(w["yaw"])
            # 把目标点变换到墙的局部系（原点在底面中心）
            dx, dy = px - w["pos"][0], py - w["pos"][1]
            lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
            ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
            # 局部系里墙占 |x|<=sx/2, |y|<=sy/2, 0<=z<=sz
            ox = max(abs(lx) - sx / 2.0, 0.0)
            oy = max(abs(ly) - sy / 2.0, 0.0)
            oz = max(0.0 - pz, pz - sz, 0.0)
            d = math.sqrt(ox * ox + oy * oy + oz * oz)
            if d < min_clear:
                min_clear = d
                worst = (i, px, py, pz)
    check("5a 目标全程不进墙体", min_clear > 0.0,
          f"离墙表面最近 {min_clear:.2f} m"
          + (f"（wall_{worst[0]:02d} @ 目标 ({worst[1]:.2f}, {worst[2]:.2f}, {worst[3]:.2f}）"
             if worst else ""))

    # ---------- 产物 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        for i, w in enumerate(tl.WALL_LAYOUT):
            sx, sy, sz = w["size"]
            yaw = float(w["yaw"])
            cx, cy = w["pos"]
            corners = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2],
                                [sx / 2, sy / 2], [-sx / 2, sy / 2]])
            R = np.array([[math.cos(yaw), -math.sin(yaw)],
                          [math.sin(yaw), math.cos(yaw)]])
            pts = corners @ R.T + np.array([cx, cy])
            ax.fill(pts[:, 0], pts[:, 1], color="0.55", zorder=2)
            ax.text(cx, cy, f"w{i}", ha="center", va="center", fontsize=7, zorder=3)
        for x, y in tl.CYLINDER_XY:
            ax.add_patch(plt.Circle((x, y), 0.25, color="0.35", zorder=2))
        tx = [q[0] for q in tgt_log]
        ty = [q[1] for q in tgt_log]
        ax.plot(tx, ty, "C1-", lw=1.5, label="target (circle r=2)", zorder=4)
        ax.add_patch(plt.Rectangle((-10, -10), 20, 20, fill=False, ec="k", ls="--"))
        ax.set_aspect("equal")
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"P1 arena 20x20 — walls + cylinders ({args.steps} steps)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(args.out_dir, "topdown.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    except ImportError as e:
        print(f"俯视图跳过（matplotlib 不可用：{e}）", flush=True)

    # ---------- 汇总 ----------
    print("=" * 78, flush=True)
    print("P1 验收汇总", flush=True)
    print("=" * 78, flush=True)
    ok = True
    for name, cond, _ in results:
        if not cond:
            ok = False
        print(f"  {'✅' if cond else '❌'} {name}", flush=True)
    print(f"\n产物 -> {os.path.abspath(args.out_dir)}/", flush=True)
    print("  depth.png   深度帧（肉眼看墙有没有立在地上）", flush=True)
    print("  seg.png     分割帧（目标红、地板灰、墙/圆柱各一色）", flush=True)
    print("  topdown.png 俯视布局", flush=True)
    print(f"\n{'P1 通过' if ok else 'P1 未通过'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
