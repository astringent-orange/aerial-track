#!/usr/bin/env python3
"""离线回归全套：import 冒烟 + 三条回归（补丁 / 槽位 / 深度值域）。

**不用 GPU、不起仿真**，几十秒跑完。上 GPU 的验收在
`tests/acceptance/check_p{1..4}.py`（要 `--device cuda:N`）。

    python tests/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = [
    ("import 冒烟", os.path.join(HERE, "smoke_import.py")),
    ("warp 补丁还在", os.path.join(HERE, "regression", "test_warp_patch.py")),
    ("槽位顺序", os.path.join(HERE, "regression", "test_slots.py")),
    ("深度值域语义", os.path.join(HERE, "regression", "test_depth_domain.py")),
    ("场地布局单一真相源", os.path.join(HERE, "regression", "test_arena_single_source.py")),
]


def main():
    failed = []
    for name, path in TESTS:
        print(f"\n===== {name}  ({os.path.relpath(path, HERE)}) =====", flush=True)
        r = subprocess.run([sys.executable, path])
        if r.returncode != 0:
            failed.append(name)
    print("\n" + "=" * 60, flush=True)
    if failed:
        print(f"❌ {len(failed)}/{len(TESTS)} 项失败：{failed}", flush=True)
        return 1
    print(f"✅ 离线回归全过（{len(TESTS)}/{len(TESTS)}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
