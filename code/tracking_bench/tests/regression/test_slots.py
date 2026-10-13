#!/usr/bin/env python3
"""回归：**槽位顺序**没被改乱。

槽位由 `tracking_env.py` 的 `asset_type_to_dict_map` 顺序 + `keep_in_env` 的
appendleft 规则决定，`tracking_lib.py` 按 YAML 墙数自动推导：

    槽 0      target
    槽 1      bottom_wall（地板 mesh）
    槽 2..9   walls（8 面，**YAML 逆序**：槽 2 = wall_07 ... 槽 9 = wall_00）
    槽 10..15 cylinders（6 根）

改 tracking_env 的 map 顺序、改 arena YAML 的墙数/圆柱数，这里就该红 ——
槽位对不上意味着 `pin_static_layout()` 钉错物体、`write_target` 写错 actor，
**静默**把墙当目标挪走。纯 Python，不用 GPU。
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)

from tracking_bench import tracking_lib as tl  # noqa: E402


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        cond = bool(cond)
        ok = ok and cond
        print(f"  {'✅' if cond else '❌'} {name}" + (f": {detail}" if detail else ""))

    check("N_WALLS == 8", tl.N_WALLS == 8, str(tl.N_WALLS))
    check("N_CYLINDERS == 6", tl.N_CYLINDERS == 6, str(tl.N_CYLINDERS))
    check("CYLINDER_SIZE == (0.6, 0.6, 3.0)", tl.CYLINDER_SIZE == (0.6, 0.6, 3.0),
          str(tl.CYLINDER_SIZE))

    check("TARGET_SLOT == 0", tl.TARGET_SLOT == 0, str(tl.TARGET_SLOT))
    check("BOTTOM_WALL_SLOT == 1", tl.BOTTOM_WALL_SLOT == 1,
          str(tl.BOTTOM_WALL_SLOT))

    # 墙：YAML 逆序，第 i 面墙在 2 + (N_WALLS-1-i)
    got = [tl.wall_slot(i) for i in range(tl.N_WALLS)]
    want = [2 + (tl.N_WALLS - 1 - i) for i in range(tl.N_WALLS)]
    check("wall_slot 逆序 2..9", got == want, f"{got} vs {want}")
    check("wall_slot(0) == 9（YAML 第一面墙在最后一槽）",
          tl.wall_slot(0) == 2 + tl.N_WALLS - 1, str(tl.wall_slot(0)))
    check("wall_slot(7) == 2（YAML 最后一面墙在槽 2）",
          tl.wall_slot(tl.N_WALLS - 1) == 2, str(tl.wall_slot(tl.N_WALLS - 1)))

    check("CYLINDER_SLOT_START == 2 + N_WALLS == 10",
          tl.CYLINDER_SLOT_START == 2 + tl.N_WALLS and tl.CYLINDER_SLOT_START == 10,
          str(tl.CYLINDER_SLOT_START))
    check("CYLINDER_SLOT_END == 16", tl.CYLINDER_SLOT_END == 16,
          str(tl.CYLINDER_SLOT_END))

    # 槽位互不重叠、刚好铺满 0..15
    all_slots = ([tl.TARGET_SLOT, tl.BOTTOM_WALL_SLOT]
                 + [tl.wall_slot(i) for i in range(tl.N_WALLS)]
                 + list(range(tl.CYLINDER_SLOT_START, tl.CYLINDER_SLOT_END)))
    check("槽位 0..15 不重不漏", sorted(all_slots) == list(range(16)),
          str(sorted(all_slots)))

    print("✅ 槽位顺序正确" if ok else "❌ 槽位顺序有错")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
