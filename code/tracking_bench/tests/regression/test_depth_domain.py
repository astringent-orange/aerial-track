#!/usr/bin/env python3
"""回归：**深度图值域语义** `[-1] ∪ [0,1]`。

这套语义是踩坑大户，改坏了**不报异常**，只会让所有拿深度换算距离的代码算出
离谱的数：

    [0, 1]   = cam_z / MAX_RANGE（MAX_RANGE = 10 m）
    +1.0     = 超出 10 m **或**没打中（**分不开**，别试图分开）
    -1.0     = 比 MIN_RANGE（0.2 m）还近。**负数不是"没打中"**

钉三件事：
  1. 常量（10 m / 0.2 m / 480×270 / 针孔内参）没被改动
  2. `project()` 的几何深度 `cz / MAX_RANGE` 不带特殊值；目标在相机**背后**
     时返回 ±1e9 哨兵（不是像素坐标，绝不能平均进 pixel_err）
  3. `compute_truth()` 对混合深度的处理：`-1.0` **算进** `target_depth_mean`
     （太近时整块就是 -1.0），同时给 `target_depth_neg1_ratio`；
     `distance` 用欧氏距离，**不从深度图换算**（混合均值不能当距离用）

纯 numpy，不用 GPU、不起仿真。
"""

import os
import sys

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)

from tracking_bench.sensors import truth as T  # noqa: E402

H, W = 270, 480
ok = True


def check(name, cond, detail=""):
    global ok
    cond = bool(cond)
    ok = ok and cond
    print(f"  {'✅' if cond else '❌'} {name}" + (f": {detail}" if detail else ""))


def main():
    # ---- 1. 常量 ----
    check("MAX_RANGE == 10.0 m", T.MAX_RANGE == 10.0, str(T.MAX_RANGE))
    check("MIN_RANGE == 0.2 m", T.MIN_RANGE == 0.2, str(T.MIN_RANGE))
    check("图像 480x270", (T.IMG_W, T.IMG_H) == (W, H), f"{T.IMG_W}x{T.IMG_H}")
    check("针孔内参 f=252.908 cx=240 cy=135",
          abs(T.FX - 252.908) < 0.01 and T.CX == 240.0 and T.CY == 135.0,
          f"f={T.FX:.3f} cx={T.CX} cy={T.CY}")

    # ---- 2. project：几何深度无特殊值；背后是哨兵 ----
    cam_pos = np.zeros(3)
    cam_R = np.eye(3)                      # 相机轴 = 世界轴，光轴 = +z
    u, v, dn, cz = T.project(cam_pos, cam_R, np.array([0.3, -0.2, 5.0]))
    check("前方目标 depth_norm = cz/MAX_RANGE", abs(dn - 0.5) < 1e-9, f"{dn}")
    check("前方目标投影点在图内", T.in_image(u, v), f"({u:.1f},{v:.1f})")
    check("前方目标 cz 有限且 > 0", cz == 5.0, f"{cz}")

    ub, vb, dnb, czb = T.project(cam_pos, cam_R, np.array([0.0, 0.0, -3.0]))
    check("背后目标是 ±1e9 哨兵（不是像素坐标）",
          abs(ub) >= 1e6 and abs(vb) >= 1e6 and czb == -3.0,
          f"({ub:.0f},{vb:.0f}) cz={czb}")
    check("背后目标 in_image == False", not T.in_image(ub, vb))

    # ---- 3. compute_truth 对混合深度的处理 ----
    # 掩码里一半 -1.0（比 MIN_RANGE 近）、一半 +1.0（远/没打中），
    # 复刻"目标贴着相机"的实测形态（那时均值 ≈ -0.543）。
    seg = np.full((H, W), -2, np.int32)
    depth = np.full((H, W), 1.0, np.float32)      # 背景：远 / 没打中
    seg[100:110, 200:210] = 100                    # 100 个目标像素
    depth[100:105, 200:210] = -1.0                 # 一半超近程
    depth[105:110, 200:210] = 1.0                  # 一半远端
    tr = T.compute_truth(seg, depth, np.array([0.0, 0.0, 5.0]), cam_pos,
                         cam_R=cam_R)
    check("target_depth_mean 是 -1/+1 的混合均值（≈ -0.0）",
          tr["target_depth_mean"] is not None
          and abs(tr["target_depth_mean"] - 0.0) < 1e-6,
          f"{tr['target_depth_mean']}")
    check("target_depth_neg1_ratio == 0.5", abs(tr["target_depth_neg1_ratio"] - 0.5) < 1e-6,
          f"{tr['target_depth_neg1_ratio']}")
    check("distance 是欧氏距离 5.0，不从深度图换算",
          abs(tr["distance"] - 5.0) < 1e-9, f"{tr['distance']}")
    check("可见性：visible ⊆ in_fov",
          (not tr["visible"]) or tr["in_fov"],
          f"visible={tr['visible']} in_fov={tr['in_fov']}")
    check("100 个目标像素", tr["target_pixel_area"] == 100,
          str(tr["target_pixel_area"]))

    # 整块超近程：mean ≈ -1.0（这正是 -1.0 要算进均值的原因）
    depth[100:110, 200:210] = -1.0
    tr2 = T.compute_truth(seg, depth, np.array([0.0, 0.0, 5.0]), cam_pos,
                          cam_R=cam_R)
    check("整块 -1.0 时 mean ≈ -1.0（超近程语义不被抹平）",
          abs(tr2["target_depth_mean"] - (-1.0)) < 1e-6,
          f"{tr2['target_depth_mean']}")
    check("整块 -1.0 时 neg1_ratio == 1.0",
          abs(tr2["target_depth_neg1_ratio"] - 1.0) < 1e-6,
          f"{tr2['target_depth_neg1_ratio']}")

    print("✅ 深度值域语义正确" if ok else "❌ 深度值域语义有错")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
