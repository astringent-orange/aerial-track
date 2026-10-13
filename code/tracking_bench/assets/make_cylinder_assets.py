#!/usr/bin/env python3
"""生成圆柱体 URDF 资产，供 aerial_gym 的环境随机摆放。

背景：aerial_gym 里所有环境物体都是 URDF 文件，AssetLoader 用
`randomly_pick_assets_from_folder()` 从资产目录里随机挑 .urdf 加载。
所以"用代码搭建环境"的完整做法是两步：

    1. 用代码批量生成 URDF  ->  本脚本
    2. 写一个环境配置引用它们  ->  aerial_gym/config/env_config/empty_with_cylinders.py

关键设计：圆柱的**几何原点放在底面**，不是几何中心。
URDF 里圆柱的轴是 z 轴，默认原点在中心，那样不同高度的圆柱要配不同
的 z 高度才能立在地面上。把 visual/collision/inertial 的 origin 整体上移
length/2 之后，link 原点就落在底面 —— 于是不管多高的圆柱，
放在 z=0 就正好立在地面上，一个 z 值搞定所有尺寸。

用法：
    python make_cylinder_assets.py                 # 默认在包里生成 12 个
    python make_cylinder_assets.py --count 20      # 想要几个
    python make_cylinder_assets.py --list          # 只看看会生成什么，不写文件
"""

import argparse
import os
import random
import sys

# 路径统一走 tracking_bench.paths —— 脚本在 assets/ 里，按 `__file__` 拼
# 资产目录会拼歪（踩过）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking_bench.paths import aerial_gym_asset_dir  # noqa: E402

# 默认写到 aerial_gym 的资产目录里，AssetLoader 只认这个位置
DEFAULT_OUT = aerial_gym_asset_dir("cylinders")

# 挑几种区分度高的颜色，RGB 相机里一眼能分清是哪个圆柱。
# 对做跟踪实验有用：分割图按实例上色，RGB 里颜色不同也便于肉眼看。
PALETTE = [
    (0.85, 0.20, 0.20),  # 红
    (0.20, 0.60, 0.90),  # 蓝
    (0.95, 0.70, 0.15),  # 黄
    (0.25, 0.75, 0.35),  # 绿
    (0.70, 0.35, 0.85),  # 紫
    (0.95, 0.50, 0.20),  # 橙
    (0.30, 0.80, 0.80),  # 青
    (0.90, 0.45, 0.65),  # 粉
]


def cylinder_urdf(radius, length, mass, color):
    """Return the URDF text for one cylinder standing on the origin.

    The link origin sits at the *base* of the cylinder (see module docstring),
    so a caller only ever needs z=0 to stand it on the ground.
    """
    # 实心圆柱绕自身轴的转动惯量 Izz = m r^2 / 2；
    # 绕径向轴 Ixx = Iyy = m (3r^2 + h^2) / 12。
    # 这里的圆柱默认是静态的（fix_base_link=True），惯量其实用不到，
    # 但算对了没坏处 —— 万一以后要让它们受力/倒下，物理才是对的。
    half = length / 2.0
    izz = 0.5 * mass * radius**2
    ixx = mass * (3.0 * radius**2 + length**2) / 12.0
    rgba = " ".join(f"{c:.3f}" for c in color)

    return f"""<?xml version='1.0' encoding='UTF-8'?>
<robot name="cylinder_r{radius:.3f}_h{length:.3f}">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 {half:.6f}" rpy="0.0 0.0 0.0"/>
      <mass value="{mass:.3f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0.0" ixz="0.0" iyy="{ixx:.6f}" iyz="0.0" izz="{izz:.6f}"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry>
        <cylinder radius="{radius:.6f}" length="{length:.6f}"/>
      </geometry>
      <origin xyz="0.0 0.0 {half:.6f}" rpy="0.0 0.0 0.0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry>
        <cylinder radius="{radius:.6f}" length="{length:.6f}"/>
      </geometry>
      <origin xyz="0.0 0.0 {half:.6f}" rpy="0.0 0.0 0.0"/>
    </collision>
  </link>
  <material name="cylinder_material">
    <color rgba="{rgba} 1.0"/>
  </material>
</robot>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    p.add_argument("--count", type=int, default=12, help="生成几个圆柱")
    p.add_argument("--radius-min", type=float, default=0.08)
    p.add_argument("--radius-max", type=float, default=0.35)
    p.add_argument("--length-min", type=float, default=0.8)
    p.add_argument("--length-max", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="固定随机种子，方便复现")
    p.add_argument("--list", action="store_true", help="只打印不写文件")
    args = p.parse_args()

    rng = random.Random(args.seed)
    specs = []
    for i in range(args.count):
        r = rng.uniform(args.radius_min, args.radius_max)
        h = rng.uniform(args.length_min, args.length_max)
        specs.append((i, r, h, PALETTE[i % len(PALETTE)]))

    for i, r, h, color in specs:
        name = f"cylinder_{i:03d}"
        print(f"{name}  radius={r:.3f}m  length={h:.3f}m  "
              f"rgb=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")

    if args.list:
        print(f"\n(--list：未写文件。目标目录 {args.out})")
        return

    os.makedirs(args.out, exist_ok=True)
    for i, r, h, color in specs:
        path = os.path.join(args.out, f"cylinder_{i:03d}.urdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cylinder_urdf(r, h, args.mass, color))
    print(f"\n已写入 {len(specs)} 个 URDF -> {args.out}")


if __name__ == "__main__":
    main()
