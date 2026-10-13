#!/usr/bin/env python3
"""`sim/` 门面 —— **全项目唯一碰 IsaacGym / aerial_gym 的地方**。

对外只暴露三样：`build_env_quiet`、`SimSession`，以及几个四元数工具。
跟踪器作者不该 import 这个模块；他们拿的是 `tracker_api.TrackerInput`。

⚠️ 里面对 aerial_gym 的**本地修改**（补丁）：
   `warp_env_manager.py` 的 `post_physics_step()` 里必须调 `update_meshes()`，
   否则 warp 渲染的网格不跟着物体动、移动物体会冻在出生点**且不报错**。
   `tests/test_sim_patch.py` 会断言这个补丁还在。

⚠️ 起始位姿的 z 必须在 [1.5, 4.0]，别指到 (0,0,0) —— 撞地会污染碰撞计数。
"""

import math
import os
import sys

import numpy as np

from tracking_bench import bench_config
from tracking_bench import tracking_lib as tl
from tracking_bench.sensors import truth as T
from tracking_bench.trackers.tracker_api import (
    Obstacle, TrackerAction, XY_LIMIT, Z_MIN, Z_MAX, DEFAULT_ACTION_MODE,
)


# --------------------------------------------------------------------------- #
#  小工具
# --------------------------------------------------------------------------- #
def quat_to_euler(q):
    """四元数 (x, y, z, w) -> (roll, pitch, yaw)。标准 XYZ 内旋。"""
    x, y, z, w = [float(v) for v in np.asarray(q).ravel()[:4]]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = 2.0 * (w * y - z * x)
    s = max(-1.0, min(1.0, s))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_to_quat(yaw):
    """绕 z 转 yaw 的四元数 (x, y, z, w)。"""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_mul(q1, q2):
    """四元数乘法 q1 ⊗ q2，**IsaacGym 的 (x, y, z, w) 顺序**，Hamilton 约定。"""
    x1, y1, z1, w1 = [float(v) for v in np.asarray(q1).ravel()[:4]]
    x2, y2, z2, w2 = [float(v) for v in np.asarray(q2).ravel()[:4]]
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def build_env_quiet(**kw):
    """建环境期间把 `sys.argv` 藏起来。

    isaacgym 的 gymutil 会解析我们的命令行，长选项缩写会撞它的 `--sim_device`
    （`--sim` 也能匹配上）。踩过：`check_p1 --sim-name xxx` 直接被它吃掉。
    """
    from aerial_gym.sim.sim_builder import SimBuilder

    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        return SimBuilder().build_env(**kw)
    finally:
        sys.argv = saved


# --------------------------------------------------------------------------- #
#  SimSession
# --------------------------------------------------------------------------- #
class SimSession:
    """一次建环境，供多个 episode 复用（换 seed 换 tracker 都不用重建）。"""

    def __init__(self, cfg: bench_config.BenchConfig, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        if cfg.num_envs != 1:
            raise NotImplementedError(
                f"P4 的 runner 只支持 num_envs=1（拿到 {cfg.num_envs}）。"
                f"多环境是锁步跑的，要等 P5 的 tracking_bench.sim 门面；"
                f"现在请把 num_envs 设成 1。"
            )
        self.env_manager = build_env_quiet(
            sim_name="base_sim",
            env_name="tracking_env",
            robot_name="base_quadrotor_with_stereo_camera",
            controller_name="lee_position_control",
            args=None,
            device=device,
            num_envs=cfg.num_envs,
            headless=True,
            use_warp=bool(cfg.use_warp),
        )
        self.gtd = self.env_manager.global_tensor_dict
        self.state = self.gtd["env_asset_state_tensor"]
        self.n_env_assets = self.state.shape[1]
        self.actions = np.zeros((cfg.num_envs, 4), dtype=np.float32)
        self.tw = None  # torch 张量，在 _init_torch 里建

        import torch
        self._torch = torch
        self.tw = torch.zeros((cfg.num_envs, self.n_env_assets, 6),
                              device=self.env_manager.device)

        # ⚠️ 必须先 reset()，否则 warp 相机的 sensor_position/orientation 是 NaN，
        #    渲染整帧 miss 且**不报错**。踩过两次了。
        self.env_manager.reset()
        tl.pin_static_layout(self.env_manager)

        n_phys = self.env_manager.cfg.env.num_physics_steps_per_env_step_mean
        self.dt = float(n_phys) * float(self.gtd["dt"])
        self.step_count = 0

        # T_body_cam：安装变换是**常量**（randomize_placement=False），直接从
        # warp_sensor 的安装参数构造。⚠️ 别去"实测一次" —— 早期版本拿
        # `sensor_position`/`sensor_orientation` 和机器人位姿去减，那时传感器位姿
        # 还没被 `update()` 同步，量到的是废值（平移 7.77/3.25/-1.32、旋转差 17°）。
        self.T_body_cam = self._mount_T_body_cam()

        # 障碍表（§2.2 的 `obstacles`）—— 从 arena YAML 来，跟踪器可完全忽略。
        # ⚠️ 圆柱尺寸取 `tl.CYLINDER_SIZE`，别手写 —— 曾经这里写死过
        #    (0.6, 0.6, 0.35)（把目标的扁圆柱尺寸抄过来了），避障的盒测试
        #    在飞行高度 z≈2 直接判不到柱子，静默失效。
        self.obstacles = [
            Obstacle(pos=tuple(w["pos"]), size=tuple(w["size"]),
                     yaw=float(w["yaw"]), kind="wall")
            for w in tl.WALL_LAYOUT
        ] + [
            Obstacle(pos=tuple(xy), size=tuple(tl.CYLINDER_SIZE), yaw=0.0,
                     kind="cylinder")
            for xy in tl.CYLINDER_XY
        ]

    # -------------------------------------------------------------- 内部
    def _mount_T_body_cam(self):
        """安装变换 `T_body_cam`：机体 -> 相机（齐次，`p_cam = T @ [p_body, 1]`）。

        warp_sensor 的定义（`warp_sensor.py:180-187`）：

            sensor_position    = tf_apply(body_q, body_p, sensor_local_position)
            sensor_orientation = body_q ⊗ (sensor_local_orientation ⊗ data_frame_quat)

        所以安装关系是 `p_body = R_mount @ p_cam + t_mount`，其中
        `t_mount = sensor_local_position`，
        `R_mount = quat_to_mat(local_q ⊗ data_frame_q)`。取逆即得 body -> cam：

            p_cam = R_mount.T @ p_body − R_mount.T @ t_mount

        ⚠️ 旋转块是 **body → cam**。相机三轴在**机体系**里的方向是
           `R_mount` 的**列**（= `T[:3,:3].T` 的列），别和旋转块本身搞混。
        ⚠️ `randomize_placement=False` 时这是常量；要是开了随机会每次 reset 变，
           那时要改成每次 reset 重算。
        """
        ws = self.env_manager.robot_manager.warp_sensor
        t_mount = ws.sensor_local_position[0, 0].detach().cpu().numpy().astype(np.float64)
        q_local = ws.sensor_local_orientation[0, 0].detach().cpu().numpy()
        q_data = ws.sensor_data_frame_quat[0, 0].detach().cpu().numpy()
        R_mount = T.quat_to_mat(quat_mul(q_local, q_data))
        R = R_mount.T                                   # body -> cam
        Tm = np.eye(4, dtype=np.float64)
        Tm[0:3, 0:3] = R
        Tm[0:3, 3] = -R @ t_mount
        return Tm

    def cam_pose(self):
        """真实相机位姿（世界系）。**必须读 warp_sensor，别用欧拉角猜**。"""
        ws = self.env_manager.robot_manager.warp_sensor
        p = ws.sensor_position[0, 0].detach().cpu().numpy()
        q = ws.sensor_orientation[0, 0].detach().cpu().numpy()
        return p, T.quat_to_mat(q)

    def chaser_state(self):
        torch = self._torch
        pos = self.gtd["robot_position"][0].detach().cpu().numpy().astype(np.float64)
        vel = self.gtd["robot_linvel"][0].detach().cpu().numpy().astype(np.float64)
        quat = self.gtd["robot_orientation"][0].detach().cpu().numpy()
        ang = self.gtd["robot_angvel"][0].detach().cpu().numpy().astype(np.float64)
        rpy = quat_to_euler(quat)
        return pos, vel, np.array(rpy, dtype=np.float64), ang

    def write_target(self, tpos, tyaw=0.0):
        """把目标钉到 (tpos, tyaw)，速度清零。

        ⚠️ 必须在 `env_manager.step()` **之前**写：`tracking_env` 开着
           `write_to_sim_at_every_timestep`，每次物理子步都会把状态张量写进 sim，
           所以提前写进张量就足够了。
        ⚠️ 速度清零是故意的：目标是 kinematically 驱动的（位置说了算），
           留着上一段的速度反而会让积分去凑。`target_vel` 报的是轨迹的解析导数。
        """
        torch = self._torch
        st = self.state
        st[:, tl.TARGET_SLOT, 0] = float(tpos[0])
        st[:, tl.TARGET_SLOT, 1] = float(tpos[1])
        st[:, tl.TARGET_SLOT, 2] = float(tpos[2])
        q = yaw_to_quat(tyaw)
        st[:, tl.TARGET_SLOT, 3] = q[0]
        st[:, tl.TARGET_SLOT, 4] = q[1]
        st[:, tl.TARGET_SLOT, 5] = q[2]
        st[:, tl.TARGET_SLOT, 6] = q[3]
        st[:, tl.TARGET_SLOT, 7:13] = 0.0

    def apply_action(self, action: TrackerAction, mode=DEFAULT_ACTION_MODE):
        """TrackerAction -> 控制器的 (num_envs, 4)。

        ⚠️ `LeePositionController.update()` 只吃 `[x, y, z, yaw]` 四个数，并且
           把 `setpoint_velocity` **写死成 0**（`position_control.py:30`）。
           所以 `velocity_ff` 收下但丢弃，`velocity` / `acceleration` 模式
           本期跑不了 —— 接口留着，等有人改控制器。
        """
        if mode != "position":
            raise NotImplementedError(
                f"action_mode={mode!r} 本期没实现：Lee 位置控制器的动作是 "
                f"(num_envs, 4) = [位置(3), 偏航]，吃不到速度/加速度设定点。"
            )
        if action.position_setpoint is None:
            raise ValueError("position 模式必须给 position_setpoint")
        p = np.asarray(action.position_setpoint, dtype=np.float64).ravel()
        # §2.3 的硬约束：超了就截断
        p = [float(np.clip(p[0], -XY_LIMIT, XY_LIMIT)),
             float(np.clip(p[1], -XY_LIMIT, XY_LIMIT)),
             float(np.clip(p[2], Z_MIN, Z_MAX))]
        self.actions[0, 0] = p[0]
        self.actions[0, 1] = p[1]
        self.actions[0, 2] = p[2]
        self.actions[0, 3] = float(action.yaw_setpoint)
        return p

    def step(self):
        """走一个控制周期。返回 True 表示这个周期里撞了。"""
        torch = self._torch
        act = torch.from_numpy(self.actions).to(self.env_manager.device)
        self.env_manager.step(actions=act, env_actions=self.tw)
        self.step_count += 1
        return int(self.gtd["crashes"][0].item()) > 0

    def render(self):
        self.env_manager.render(render_components="sensors")

    def frame(self):
        """(seg, depth) 的 numpy 副本。"""
        seg = self.gtd["segmentation_pixels"][0, 0].detach().cpu().numpy()
        depth = self.gtd["depth_range_pixels"][0, 0].detach().cpu().numpy()
        return seg, depth

    def teleport_robot(self, pos, yaw):
        """把机身**直接**搬到 (pos, yaw)，线速度/角速度/旋翼转速全清零。

        为什么必须有这个：撞进墙/圆柱之后无人机会卡在几何体里，Lee 位置控制器
        被接触力顶住，光发位置设定点飞不出来 —— 实测 setpoint 命令到 (0,0,2.5)，
        机身纹丝不动，后面每个 episode 都在第 2 步撞死、`dist_err≈12 m`。
        传送是唯一干净的复位手段。

        ⚠️ `robot_state_tensor` 是 `unfolded_vec_root_tensor[:, 0, :]` 的**视图**，
           所以写完调 `IGE_env.write_to_sim()` 就生效（它写的是整张 root 张量，
           含机器人）。别动 `env_asset_state_tensor`，那个是槽 1..N 的资产，
           不含机器人。
        """
        torch = self._torch
        dev = self.env_manager.device
        st = self.gtd["robot_state_tensor"]
        q = yaw_to_quat(float(yaw))
        st[0, 0:3] = torch.tensor([float(pos[0]), float(pos[1]), float(pos[2])], device=dev)
        st[0, 3:7] = torch.tensor(q, device=dev)
        st[0, 7:13] = 0.0
        dof = self.gtd.get("dof_state_tensor", None)
        if dof is not None:
            dof.zero_()
        self.env_manager.IGE_env.write_to_sim()

    def settle(self, pos, yaw, n_steps, target_pos=None, target_yaw=0.0):
        """把追踪机**放**到 (pos, yaw) 并稳住。目标默认钉在 target_pos（或不动）。

        开头先 `teleport_robot` —— 上一局撞卡了要能飞出来，settle 的位置设定点
        不足以把陷在几何体里的机身顶出来（见 teleport_robot 的注释）。

        ⚠️ 起始位姿的 z 必须在 [1.5, 4.0]，别指到 (0,0,0) —— 撞地会触发
           aerial_gym 的 reset（虽然 tracking_env 关了 reset_on_collision，
           但撞地的接触力会污染碰撞计数）。踩过。
        """
        pos = [float(np.clip(pos[0], -XY_LIMIT, XY_LIMIT)),
               float(np.clip(pos[1], -XY_LIMIT, XY_LIMIT)),
               float(np.clip(pos[2], Z_MIN, Z_MAX))]
        self.teleport_robot(pos, yaw)
        for _ in range(int(n_steps)):
            if target_pos is not None:
                self.write_target(target_pos, target_yaw)
            self.actions[0, 0:3] = pos
            self.actions[0, 3] = float(yaw)
            self.step()
        self.render()

    def close(self):
        try:
            self.env_manager.close() if hasattr(self.env_manager, "close") else None
        except Exception:
            pass


