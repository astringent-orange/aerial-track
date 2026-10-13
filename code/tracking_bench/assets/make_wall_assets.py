#!/usr/bin/env python3
"""按 configs/arena_20x20.yaml 生成薄墙 URDF 资产。

和 make_cylinder_assets.py 是同一套思路（代码生成 URDF，aerial_gym 用配置引用），
区别在两点：

  1. **尺寸来自 YAML，不是随机**。YAML 是场地布局的唯一来源，
     改场地 = 改 YAML = 重跑本脚本，不用动 Python 逻辑。
  2. **一面墙一个 URDF 文件**（wall_00.urdf ... wall_07.urdf），
     哪个文件对应哪个位置是一一对应的。如果共用一个文件、靠 `file=None` 从
     目录里随机挑，同一个尺寸的几面墙就会互相换位置，槽位就对不上了。

⚠️ 关键约定：**几何原点放在底面**，不是几何中心。
URDF 的 box 默认以原点为中心，那样 3m 高的墙要写 z=1.5 才立在地面上，
而 0.5m 高的圆柱要写 z=0 —— 两套约定混用必然摆错。
把 visual/collision/inertial 的 origin 整体上移 height/2 之后，link 原点
落在底面，于是**所有尺寸都用 z=0 摆放**，和圆柱一致。

现成的 `resources/models/environment_assets/objects/1_x_1_wall.urdf` 是
"原点在几何中心"的写法，**不要复用**。

用法（容器内，先 cd /workspace）：
    python make_wall_assets.py --from-config configs/arena_20x20.yaml
    python make_wall_assets.py --from-config configs/arena_20x20.yaml --list
"""

import argparse
import os
import sys

import yaml

# 路径统一走 tracking_bench.paths —— 脚本在 assets/ 里，按 `__file__` 拼
# `configs/` / 资产目录都会拼歪（踩过）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking_bench.paths import aerial_gym_asset_dir, find_config  # noqa: E402

DEFAULT_CFG = find_config("arena_20x20.yaml")
DEFAULT_OUT = aerial_gym_asset_dir("walls_thin")

# 墙的颜色：低饱和灰蓝，在深度图/分割图里和橙色目标、彩色圆柱区分得开。
# 分割图是按实例 id 散列着色的（见 record_sim.py 的 to_uint8），RGB 才用得上这里。
WALL_RGBA = (0.62, 0.68, 0.74)


def box_urdf(name, sx, sy, sz, mass, rgba):
    """Return the URDF text for a box standing on the origin.

    The link origin sits at the *bottom face* of the box (see module docstring),
    so a caller only ever needs z=0 to stand it on the ground.
    """
    # 长方体绕三轴的转动惯量（以自身几何中心为原点）。
    # 墙默认是静态的（fix_base_link=True），惯量用不到，但算对了没坏处。
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    half_z = sz / 2.0
    r, g, b = rgba

    return f"""<?xml version='1.0' encoding='UTF-8'?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 {half_z:.6f}" rpy="0.0 0.0 0.0"/>
      <mass value="{mass:.3f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0.0" ixz="0.0" iyy="{iyy:.6f}" iyz="0.0" izz="{izz:.6f}"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry>
        <box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>
      </geometry>
      <origin xyz="0.0 0.0 {half_z:.6f}" rpy="0.0 0.0 0.0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry>
        <box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>
      </geometry>
      <origin xyz="0.0 0.0 {half_z:.6f}" rpy="0.0 0.0 0.0"/>
    </collision>
  </link>
  <material name="wall_material">
    <color rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>
  </material>
</robot>
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--from-config", default=DEFAULT_CFG, help="布局 YAML")
    p.add_argument("--out", default=DEFAULT_OUT, help="URDF 输出目录")
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--list", action="store_true", help="只打印不写文件")
    args = p.parse_args()

    if not os.path.isfile(args.from_config):
        sys.exit(f"找不到布局文件：{args.from_config}")

    with open(args.from_config, encoding="utf-8") as f:
        arena = yaml.safe_load(f)

    walls = arena.get("walls") or []
    if not walls:
        sys.exit(f"{args.from_config} 里没有 walls 条目")

    print(f"布局：{args.from_config}")
    print(f"{'#':>3} {'size (L x W x H)':^22} {'pos (x, y)':^18} {'yaw':>8}")
    for i, w in enumerate(walls):
        sx, sy, sz = w["size"]
        x, y = w["pos"]
        print(f"{i:3d} {sx:6.2f} x {sy:5.2f} x {sz:5.2f}    "
              f"({x:6.2f}, {y:6.2f})  {w['yaw']:8.4f}")

    if args.list:
        print(f"\n(--list：未写文件。目标目录 {args.out})")
        return

    os.makedirs(args.out, exist_ok=True)
    for i, w in enumerate(walls):
        sx, sy, sz = w["size"]
        name = f"wall_{i:02d}"
        path = os.path.join(args.out, f"{name}.urdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(box_urdf(name, sx, sy, sz, args.mass, WALL_RGBA))

    print(f"\n已写入 {len(walls)} 个 URDF -> {args.out}")
    print("提醒：改了布局要重跑本脚本，然后确认 wall_asset_config.py 能加载到这些文件。")


if __name__ == "__main__":
    main()
