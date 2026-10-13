#!/usr/bin/env python3
"""回归：**vendored 补丁还在** —— `warp_env_manager.post_physics_step()` 里
必须调 `update_meshes()`。

为什么钉这条：warp 渲染的网格只在 `reset_idx()` 里重算顶点 + `refit()`，
物体一移动就**冻结在出生点**（物理和状态张量都正常，不报任何异常）。
修法是把重算抽成 `update_meshes()`，并在 `post_physics_step()` 每步照做一次。
这是对 aerial_gym 的**本地修改**（见 README「对 aerial_gym 的本地修改」），
换机器 / 重装 aerial_gym 极易丢 —— 丢了以后深度图静默出错，比崩溃难查得多。

不用 GPU、不起仿真，纯读源码。
"""

import importlib.util
import os
import re
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)


def find_warp_env_manager():
    """定位 aerial_gym 的 warp_env_manager.py（不 import 它）。"""
    spec = importlib.util.find_spec("aerial_gym")
    if spec is None or not spec.origin:
        raise AssertionError(
            "找不到 aerial_gym 包。回归测试要对着**安装好的** aerial_gym 查补丁。"
        )
    pkg_dir = os.path.dirname(os.path.realpath(spec.origin))
    path = os.path.join(pkg_dir, "env_manager", "warp_env_manager.py")
    if not os.path.isfile(path):
        raise AssertionError(f"aerial_gym 里没有 {path}")
    return path


def method_source(src, name):
    """抠出 `    def name(self, ...):` 到下一个同级 def 之间的源码。"""
    m = re.search(rf"^    def {name}\(.*?\n(.*?)(?=^    def |\Z)",
                  src, flags=re.S | re.M)
    if not m:
        raise AssertionError(f"源码里找不到 `def {name}`")
    return m.group(0)


def main():
    path = find_warp_env_manager()
    with open(path, encoding="utf-8") as f:
        src = f.read()

    assert "def update_meshes" in src, (
        f"{path} 里没有 update_meshes() —— warp 网格不跟物体动的补丁整个丢了。"
    )
    body = method_source(src, "post_physics_step")
    assert "update_meshes" in body, (
        f"{path} 的 post_physics_step() 里没有调 update_meshes() ——\n"
        f"移动物体的渲染会冻在出生点，**不报任何异常**。补回一行\n"
        f"    self.update_meshes()\n"
        f"（补丁细节见 docs/SETUP_NOTES.md「跟踪闭环踩过的坑」第 3 条）。"
    )
    print(f"✅ warp 补丁在：{os.path.relpath(path, '/workspace')}"
          f" 的 post_physics_step() 调了 update_meshes()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
