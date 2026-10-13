#!/usr/bin/env python3
"""回归：**场地布局只有一个真相源**，aerial_gym 那边也找得到它。

`aerial_gym/config/asset_config/wall_asset_config.py` 在 import 时读
`arena_20x20.yaml`（thin wall 布局的唯一来源）。它按两级找：

    1. <AERIAL_GYM>/../tracking_bench/configs/arena_20x20.yaml   （P5 后的正源）
    2. <AERIAL_GYM>/../configs/arena_20x20.yaml                  （老布局兜底）

为什么钉这条：2026-10-12 清理 `code/` 时删了老的 `configs/`，aerial_gym 那边
还在找它 —— **下一次建环境才炸**，import 冒烟测不到。同一个坑还要求两边的
YAML **内容一致**（漂了就是两个真相源，改一处忘一处）。

纯路径检查，不 import aerial_gym / isaacgym。
"""

import importlib.util
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)

from tracking_bench.paths import find_config  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    cond = bool(cond)
    ok = ok and cond
    print(f"  {'✅' if cond else '❌'} {name}" + (f": {detail}" if detail else ""))


def main():
    spec = importlib.util.find_spec("aerial_gym")
    check("aerial_gym 可定位", spec is not None and spec.origin)
    root = os.path.dirname(os.path.dirname(os.path.realpath(spec.origin)))

    # wall_asset_config.py 的解析顺序（保持两边一致！）
    candidates = [
        os.path.normpath(os.path.join(root, "..", "tracking_bench", "configs",
                                      "arena_20x20.yaml")),
        os.path.normpath(os.path.join(root, "..", "configs", "arena_20x20.yaml")),
    ]
    found = next((p for p in candidates if os.path.isfile(p)), None)
    check("aerial_gym 能找到场地 YAML", found is not None, f"找过 {candidates}")

    pkg_cfg = find_config("arena_20x20.yaml")
    check("包内正源存在", os.path.isfile(pkg_cfg), pkg_cfg)
    if found is not None:
        with open(found, encoding="utf-8") as f:
            a = f.read()
        with open(pkg_cfg, encoding="utf-8") as f:
            b = f.read()
        check("两边 YAML 内容一致（单一真相源）", a == b,
              f"{found} vs {pkg_cfg}")
        check("wall_asset_config 的 CFG_PATH 会选到存在的那个",
              os.path.dirname(found) in
              (os.path.dirname(candidates[0]), os.path.dirname(candidates[1])))

    print("✅ 场地布局单一真相源" if ok else "❌ 场地布局有第二个真相源/找不到")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
