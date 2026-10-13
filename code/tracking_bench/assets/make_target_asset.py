#!/usr/bin/env python3
"""生成跟踪实验用的「目标物」URDF —— 一个亮橙色的扁圆柱。

配套：
    aerial_gym/config/asset_config/target_asset_config.py   引用它
    aerial_gym/config/env_config/tracking_env.py            把它挂进场景

和 make_cylinder_assets.py 的**关键区别是原点位置**：
    圆柱障碍物的原点在**底面**，z=0 就是立在地面上。
    目标物是要飞的，原点放在**几何中心** —— 这样轨迹给的 (x, y, z)
    就等于它在画面上的中心位置，不用再补一个半长度的偏移。
    两者别搞混，混了以后目标会比你以为的低/高半个身位。

为什么是圆柱不是球：aerial_gym 现有 4600 多个几何体全是 `<box>` / `<cylinder>`
（实测 grep 统计），`<sphere>` 在 IsaacGym 里能不能加载没有验证过。冒烟测试的
目的是验证「目标能不能被驱动」，不是验证外观，所以先用有把握的形状。
真要换成球体，改本文件里的 geometry 三处 + 重跑即可，配置不用动。

用法：
    python make_target_asset.py            # 生成到 aerial_gym 的资产目录
    python make_target_asset.py --list     # 只看会写什么，不落盘
"""

import argparse
import os
import sys

# 路径统一走 tracking_bench.paths —— 脚本在 assets/ 里，按 `__file__` 拼
# 资产目录会拼歪（踩过）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking_bench.paths import aerial_gym_asset_dir  # noqa: E402

DEFAULT_OUT = aerial_gym_asset_dir("target_drone")

# 亮橙色。在深度图里它是个比圆柱粗短的团块，在 RGB 里一眼能认出来。
# 和 make_cylinder_assets 的调色板刻意错开（那边橙色是 0.95, 0.50, 0.20，
# 偏浅）—— 目标要能和障碍物区分开。
TARGET_COLOR = (0.95, 0.35, 0.05)

RADIUS = 0.30
LENGTH = 0.35
MASS = 0.20


def target_urdf(radius=RADIUS, length=LENGTH, mass=MASS, color=TARGET_COLOR):
    """Return the URDF text for the target blob, origin at its **center**."""
    half = length / 2.0
    # 实心圆柱：绕自身轴 Izz = m r^2 / 2，绕径向轴 Ixx = Iyy = m (3r^2 + h^2)/12。
    # 目标是 disable_gravity + 被驱动的，惯量基本用不到，但算对了没坏处。
    izz = 0.5 * mass * radius**2
    ixx = mass * (3.0 * radius**2 + length**2) / 12.0
    rgba = " ".join(f"{c:.3f}" for c in color)

    # 注意 origin 都是 0 —— 几何中心就是 link 原点（和障碍圆柱相反）。
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<robot name="target_orb">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <mass value="{mass:.3f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0.0" ixz="0.0" iyy="{ixx:.6f}" iyz="0.0" izz="{izz:.6f}"/>
    </inertial>
    <visual name="base_link_visual">
      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <geometry>
        <cylinder radius="{radius:.6f}" length="{length:.6f}"/>
      </geometry>
      <material name="target_orange">
        <color rgba="{rgba} 1.0"/>
      </material>
    </visual>
    <collision name="base_link_collision">
      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <geometry>
        <cylinder radius="{radius:.6f}" length="{length:.6f}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=DEFAULT_OUT, help="输出目录（默认写进 aerial_gym 资产目录）")
    p.add_argument("--list", action="store_true", help="只打印，不写文件")
    args = p.parse_args()

    filename = "target_orb.urdf"
    text = target_urdf()
    path = os.path.join(args.out, filename)

    if args.list:
        print(f"would write {path} ({len(text)} bytes)")
        return

    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
