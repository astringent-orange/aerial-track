"""配置文件的定位 —— 唯一一处按目录结构找 `configs/` 的地方。

打包之后模块会挪进子包（`sim/` / `target/` / `sensors/` …），`__file__` 的深度
跟着变，各处各写一遍 `dirname(abspath(__file__))` 必然会对不上（踩过：`tracking_lib`
搬到 `tracking_bench/` 下后去找 `tracking_bench/tracking_bench/configs/`）。
所以收成一个函数，往**包根目录**找。

包根 = 含 `configs/` 的那一层，也就是 `tracking_bench/`（外层）：

    tracking_bench/            ← 包根（configs/ 在这）
    ├── configs/
    └── tracking_bench/        ← Python 包
        ├── paths.py           ← 本文件
        └── sim/ target/ …
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def package_root():
    """包根目录（含 `configs/` 的那一层）。"""
    # paths.py 在 <root>/tracking_bench/paths.py -> 往上一层
    root = os.path.dirname(_HERE)
    if os.path.isdir(os.path.join(root, "configs")):
        return root
    # 兜底：再往上（万一被拷成了 tracking_bench/tracking_bench/tracking_bench/）
    up = os.path.dirname(root)
    if os.path.isdir(os.path.join(up, "configs")):
        return up
    raise FileNotFoundError(
        f"找不到 configs/ 目录。从 {_HERE} 往上找过 "
        f"{root} 和 {up}。整个 tracking_bench/ 目录要一起拷，别只拷里面的包。"
    )


def find_config(*parts):
    """定位 `configs/` 下的文件，返回绝对路径。

    `find_config("arena_20x20.yaml")`、`find_config("bench.yaml")`。
    """
    p = os.path.join(package_root(), "configs", *parts)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"配置文件不存在：{p}")
    return p


def aerial_gym_asset_dir(*parts):
    """aerial_gym 的资产目录 `<AERIAL_GYM>/resources/models/environment_assets/...`。

    `AssetLoader` **只认这个位置**（`aerial_gym/__init__.py` 的
    `AERIAL_GYM_DIRECTORY = dirname(dirname(realpath(__file__)))`）。生成 URDF 的
    脚本把默认输出指到这里，别按 `__file__` 拼 —— 脚本挪进 `assets/` 就拼歪了。

    定位顺序：
      1. 已安装的 `aerial_gym` 包在哪，资产目录就在它的上一层下面
      2. 兜底：开发工作区 `<ws>/code/{tracking_bench, aerial_gym_simulator}`
    """
    import importlib.util
    root = None
    spec = importlib.util.find_spec("aerial_gym")
    if spec is not None and spec.origin:
        root = os.path.dirname(os.path.dirname(os.path.realpath(spec.origin)))
    else:
        guess = os.path.join(os.path.dirname(package_root()), "aerial_gym_simulator")
        if os.path.isdir(guess):
            root = guess
    if root is None:
        raise FileNotFoundError(
            "找不到 aerial_gym 的安装位置，定不了资产目录。"
            "装好 aerial_gym 再跑，或者用 --out 显式指定输出目录。"
        )
    return os.path.join(root, "resources", "models", "environment_assets", *parts)
