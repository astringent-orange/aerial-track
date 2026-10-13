#!/usr/bin/env python3
"""P4：测试环境的配置（docs/BENCH_PLAN.md §2.5）。

YAML 驱动，跑起来之后**整个配置连同 seed 一起落盘到输出目录**（§2.6 的
`config_snapshot`），保证"跑起来看着对"也能复现。

⚠️ `import isaacgym` 之后默认文本编码会变成 ASCII，读 YAML **必须显式
   `encoding="utf-8"`** —— 这里所有 open() 都带上了。
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml

# 默认配置文件。路径经 paths.find_config 统一解析（打包后模块在子包里，
# 按 `__file__` 直接拼 `configs/` 会拼到子包下，踩过）。
from tracking_bench.paths import find_config as _find_config, package_root

DEFAULT_CONFIG_PATH = _find_config("bench.yaml")
DEFAULT_ARENA_PATH = _find_config("arena_20x20.yaml")


@dataclass
class CameraConfig:
    image: List[int] = field(default_factory=lambda: [270, 480])
    hfov_deg: float = 87.0
    max_range: float = 10.0
    min_range: float = 0.2


@dataclass
class TargetPolicyConfig:
    """目标运动模型的参数。

    ⚠️ **几何约束不在这儿**：`xy_min / xy_max / z_range / wall_clearance` 是
       `arena_yaml` 里 `waypoints:` 那一节的，由 `WaypointPolicy` 自己去读。
       两处都写会变成两个真相源，改一处忘一处。
    """

    type: str = "waypoint"
    v_cruise: float = 1.2          # m/s，切向速度（Catmull-Rom）
    v_max: float = 1.2             # m/s，段内峰值速度上限（**段时长按它反推**）
    a_max: float = 1.5             # m/s^2
    t_min: float = 1.0             # s，段时长下限
    n_waypoints: int = 6           # 预生成的路点个数
    speed_scale: float = 1.0       # set_speed_scale(k)，k>1 加速
    # 脚本化场景：给了就调 `TargetPolicy.set_waypoints()` 覆盖随机生成。
    # 位置是 [[x,y,z], ...]，走完会自动续随机段（WaypointPolicy.state_at 的行为）。
    waypoints: Optional[list] = None


@dataclass
class TrackerConfig:
    type: str = "baseline"         # 或别人注册的类名（见 runner.TRACKER_REGISTRY）
    action_mode: str = "position"  # position | velocity | acceleration
    standoff: float = 2.0          # baseline 专用
    # 额外 kwargs 直接透传给跟踪器构造函数
    extra: dict = field(default_factory=dict)


@dataclass
class EpisodeConfig:
    max_steps: int = 600
    lost_timeout: float = 3.0      # s，连续丢失这么长就判失败
    lost_criterion: str = "in_fov"  # in_fov | visible。默认 in_fov，见 §2.6
    collision_is_fail: bool = True
    settle_steps: int = 40         # episode 开头先把追踪机摆到起始位姿的步数
    # 追踪机起始位姿 (x, y, z, yaw)。None = 用 baseline 几何从目标 t=0 反推。
    # 验收脚本会钉死它 —— 判据不能依赖 `env_manager.reset()` 的随机初始位姿。
    start_pose: Optional[list] = None


@dataclass
class BenchConfig:
    arena_yaml: str = DEFAULT_ARENA_PATH
    camera: CameraConfig = field(default_factory=CameraConfig)
    target_policy: TargetPolicyConfig = field(default_factory=TargetPolicyConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    seed: int = 20261012
    num_envs: int = 1
    use_warp: bool = True          # True = depth/seg，单卡；False = RGB，两卡

    @property
    def dt(self):
        """控制周期。和 `gtd["dt"] * num_physics_steps_per_env_step_mean` 一致（0.1 s）。"""
        return 0.1

    def to_dict(self):
        return asdict(self)

    @property
    def wall_layout_id(self):
        """布局标识 = arena YAML 的文件名（不含扩展名）。"""
        return os.path.splitext(os.path.basename(self.arena_yaml))[0]


def _merge(dc, data):
    """把 dict 里的键覆盖到 dataclass 实例上（嵌套 dataclass 递归）。"""
    if not data:
        return dc
    for k, v in data.items():
        if not hasattr(dc, k):
            raise KeyError(f"配置里有未知字段 {k!r}（{type(dc).__name__} 不认识它）")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def _resolve_arena_yaml(value, base_dir):
    """把配置里的 `arena_yaml` 解析成绝对路径。

    ⚠️ 别用"相对当前目录"这一种解释 —— 换个 cwd 跑就静默读错/找不到（踩过：
       `arena_yaml: configs/arena_20x20.yaml` 写的是相对 /workspace，从别的目录
       跑必炸）。按下面顺序找，谁先存在用谁：

        1. 绝对路径，直接用
        2. 相对**这份配置 YAML 自己**的目录（`arena_yaml: arena_20x20.yaml`）
        3. 相对**包根**（`arena_yaml: configs/arena_20x20.yaml` 这种写法）
        4. `configs/` 下的同名文件（只写文件名也能找到）
        5. 相对当前目录（最后兜底，兼容老写法）
    """
    if value is None:
        return DEFAULT_ARENA_PATH
    value = str(value)
    if os.path.isabs(value):
        if os.path.isfile(value):
            return value
        raise FileNotFoundError(f"arena_yaml 指的文件不存在：{value}")
    root = package_root()
    candidates = [
        os.path.join(base_dir, value),
        os.path.join(root, value),
        os.path.join(root, "configs", os.path.basename(value)),
        value,
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"arena_yaml={value!r} 找不到文件。试过：{candidates}"
    )


def load_config(path=None, overrides=None):
    """读 YAML -> `BenchConfig`。`overrides` 是一个嵌套 dict，最后盖上去。

    YAML 里没写的字段用 dataclass 默认值，所以最小配置可以只有几行。
    """
    cfg = BenchConfig()
    base_dir = os.getcwd()
    if path is not None:
        path = os.path.abspath(path)
        base_dir = os.path.dirname(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    cfg.arena_yaml = _resolve_arena_yaml(cfg.arena_yaml, base_dir)
    # 归一化几个易错的类型
    cfg.episode.max_steps = int(cfg.episode.max_steps)
    cfg.episode.lost_timeout = float(cfg.episode.lost_timeout)
    cfg.seed = int(cfg.seed)
    cfg.num_envs = int(cfg.num_envs)
    cfg.use_warp = bool(cfg.use_warp)
    if cfg.episode.lost_criterion not in ("in_fov", "visible"):
        raise ValueError(
            f"episode.lost_criterion 只能是 'in_fov' 或 'visible'，拿到 "
            f"{cfg.episode.lost_criterion!r}"
        )
    return cfg


def save_config_snapshot(cfg, out_dir, name="config_snapshot.yaml"):
    """把**最终生效**的配置落盘，返回文件路径。§2.6 的 `config_snapshot`。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
    return path
