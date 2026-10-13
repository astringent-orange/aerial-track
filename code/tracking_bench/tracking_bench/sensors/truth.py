#!/usr/bin/env python3
"""P3：目标真值输出 —— 相机域的那几个观测字段（docs/BENCH_PLAN.md §2.2「目标观测」）。

给定一帧分割图 + 一帧深度图 + 目标世界坐标 + 相机真实位姿，算出：

    in_fov / visible / occluded / bbox / center_pixel / projected_pixel /
    target_pixel_area / target_depth_mean / distance

**状态域**那几个字段（`target_pos` / `target_vel` / `target_yaw` / `target_valid`）
不在这儿 —— 它们直接来自目标策略和状态张量，不经成像。

⚠️ 相机位姿必须读 `warp_sensor.sensor_position` / `sensor_orientation`，
   **不要用欧拉角去猜**。相机装在机体原点（`randomize_placement=False`），
   但用真实位姿这条路径对安装位置变化免疫。

⚠️ 投影公式是**针孔**模型，和 warp 渲染器一致（`warp_cam.py` 里
   `f = (W/2)/tan(hfov/2)`、`vertical_fov = 2*atan(H/(2f))`，fx = fy = f，方像素）：

       cx, cy, cz = R_cam.T @ (target_pos - cam_pos)
       u = 240 + f * cx / cz
       v = 135 + f * cy / cz          # v 向下
       f = 240 / tan(radians(43.5)) = 252.908

   相机是 OpenCV 约定：x 右 / y 下 / z 前（光轴）。**v 向下增大。**

   ⚠️ 别用等距柱状的 `u = 240 + degrees(atan2(cx,cz))/43.5*240`。那条是**角度线性**，
   只在图心和图边与针孔重合，中间能差出 16 px（离轴 15° 处实测 15.5 px）。
   曾经把这 16 px 误记成"掩码质心的系统性偏置" —— 其实是公式用错了。
   详见 `project_equirect` 和 `docs/SETUP_NOTES.md`。

⚠️ `in_fov` 和 `visible` 必须分开。加了墙之后"目标丢了"分不清是**转头转丢了**
   （`in_fov=False`）还是**被墙挡了**（`in_fov=True, visible=False`）——
   这两种的处理策略完全不同。

   目标是**扩展物体**，所以 `in_fov` 不能只看中心投影：

       in_fov   = in_image(中心投影) or visible      # 有一部分在视场里（不管挡没挡）
       visible  = 掩码非空                           # 真的渲染出来了
       occluded = in_fov and not visible             # 中心在图内，但被挡住

   三条不变量：`visible ⊆ in_fov`；`occluded ⇒ 中心在图内`；
   丢失（`not in_fov`）= 中心出图 **且** 一个目标像素都没有。
"""

import math

import numpy as np

# ---- 相机内参（针孔，方像素；fx = fy = f）----
IMG_W = 480
IMG_H = 270
HFOV_HALF = 43.5  # deg，horizontal_fov_deg = 87.0 的一半
# f = (W/2) / tan(hfov/2)。这一条直接抄 warp_cam.py 的内参构造。
F_LEN = (IMG_W / 2.0) / math.tan(math.radians(HFOV_HALF))   # 252.908
FX = F_LEN
FY = F_LEN
CX = IMG_W / 2.0
CY = IMG_H / 2.0
K = (FX, FY, CX, CY)          # 对外暴露的内参（docs/BENCH_PLAN.md §2.2 的 `K` 字段）
# 垂直视场是从 f 和 H **推出来**的，不是独立参数 —— 推出来正好 28.10°，
# 和以前手标的 VFOV_HALF=28.1 对上，反过来印证 f 是对的。
VFOV_HALF = math.degrees(math.atan((IMG_H / 2.0) / F_LEN))  # 28.10 deg

# ---- 深度图语义（值域 [-1] ∪ [0,1]，别当 [0,1] 用）----
MAX_RANGE = 10.0   # m，超过就写 +1.0
MIN_RANGE = 0.2    # m，比这还近就写 -1.0
# ⚠️ +1.0 分不开"超程"和"没打中"；-1.0 **不是**"没打中"，是"太近了"。

# ---- 分割实例 id ----
TARGET_SEG_ID = 100

# 掩码质心 vs 几何投影点的**残差**。投影公式换成针孔之后，这个值实测
# 是 -0.7 ~ 0.0 px（见 check_p3 的 1d）—— 就是网格离散化噪声，**不是**系统性偏置。
# 早先记的 +16px u / +2px v 是等距柱状公式用错了造成的，别再当偏置去补。
# 保留这两个数只是为了让 check_p3 有容差可写；评估默认用 center_pixel，
# projected_pixel 用来定位"是跟踪器的问题还是成像的问题"。
CENTROID_RESIDUAL_TOL_U = 3.0   # px
CENTROID_RESIDUAL_TOL_V = 3.0   # px


def _to_np(x):
    """torch.Tensor / numpy / list -> float64 或 int 的 numpy 数组。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def quat_to_mat(q):
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵。

    列向量是相机三根轴在**世界系**的表示，所以世界 -> 相机用 `R.T @ d`。
    ⚠️ IsaacGym 的四元数顺序是 **(x, y, z, w)**，不是 (w, x, y, z)。
    """
    x, y, z, w = [float(v) for v in _to_np(q).ravel()[:4]]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project(cam_pos, cam_R, p_world):
    """世界点 -> 像素（**针孔**，和 warp 渲染器同一套内参）。

    返回 `(u, v, depth_norm, cz)`：
      u, v        像素坐标（浮点，**可以是负数或超出图幅**，用来判 in_fov）
      depth_norm  `cz / MAX_RANGE`，和深度图同一值域，但**不带** -1/+1 的特殊值
      cz          相机系 z（光轴方向），`cz <= 0` 表示在相机背后

    ⚠️ 改它 = 改成像模型。要改先回 docs/SETUP_NOTES.md 重标定。
    """
    d = _to_np(p_world).ravel()[:3] - _to_np(cam_pos).ravel()[:3]
    cx, cy, cz = _to_np(cam_R).T @ d
    cz = float(cz)
    if cz <= 1e-9:
        # 在相机背后（或恰好落在像平面上）：针孔投影无定义。
        # 给一个符号正确的超大值，保证 in_fov=False，又不产生 nan/inf。
        su = 1.0 if cx >= 0 else -1.0
        sv = 1.0 if cy >= 0 else -1.0
        return 1e9 * su, 1e9 * sv, cz / MAX_RANGE, cz
    u = CX + FX * cx / cz
    v = CY + FY * cy / cz
    return float(u), float(v), cz / MAX_RANGE, cz


def project_equirect(cam_pos, cam_R, p_world):
    """【仅留档，别用来对照掩码】旧的等距柱状投影。

        u = 240 + degrees(atan2(cx, cz)) / 43.5 * 240
        v = 135 + degrees(atan2(cy, cz)) / 28.1 * 135

    它把**角度**当像素线性刻度，而 warp 是针孔（像素 ∝ tan(角度)）。
    两者在图心（u=240）和图边重合，中间最大差 16 px：

        角度 15°  ->  等距柱状 240-82.8 = 157.2
                    针孔     240 - f·tan(15°)/1 = 240-67.7 = 172.3   差 15.1 px

    2026-09 那个"掩码质心相对投影点系统性偏 +16px u / +2px v"就是这么来的 ——
    不是渲染偏置，是公式用错了。留着这个函数是为了让那段记录能被复算。
    """
    d = _to_np(p_world).ravel()[:3] - _to_np(cam_pos).ravel()[:3]
    cx, cy, cz = _to_np(cam_R).T @ d
    u = CX + math.degrees(math.atan2(cx, cz)) / HFOV_HALF * CX
    v = CY + math.degrees(math.atan2(cy, cz)) / VFOV_HALF * CY
    return float(u), float(v), float(cz) / MAX_RANGE, float(cz)


def in_image(u, v):
    """投影点落在 480x270 图幅内（不含边界外）。"""
    return 0.0 <= u < IMG_W and 0.0 <= v < IMG_H


def compute_truth(seg, depth, target_pos, cam_pos, cam_quat=None, cam_R=None,
                  target_seg_id=TARGET_SEG_ID):
    """算出 docs/BENCH_PLAN.md §2.2「目标观测」那一整块字段。

    参数
    ----
    seg            (H, W) 分割图，逐实例 id
    depth          (H, W) 深度图，值域 `[-1] ∪ [0,1]`
    target_pos     (3,) 目标世界坐标
    cam_pos        (3,) 相机世界坐标 —— 读 `warp_sensor.sensor_position`
    cam_quat       (4,) 相机四元数 (x,y,z,w) —— 读 `warp_sensor.sensor_orientation`
    cam_R          (3,3) 或者直接给旋转矩阵（给了就忽略 cam_quat）
    target_seg_id  目标的分割实例 id，默认 100

    返回一个 dict，字段见模块头。
    """
    if cam_R is None:
        if cam_quat is None:
            raise ValueError("cam_quat 和 cam_R 至少给一个")
        cam_R = quat_to_mat(cam_quat)
    cam_R = _to_np(cam_R)

    seg = _to_np(seg)
    depth = _to_np(depth)
    tp = _to_np(target_pos).ravel()[:3]

    # ---- 几何投影（不管挡没挡）----
    u, v, depth_norm, cz = project(cam_pos, cam_R, tp)
    projected_pixel = (u, v)

    distance = float(np.linalg.norm(tp - _to_np(cam_pos).ravel()[:3]))

    # ---- 渲染判定：掩码 ----
    mask = seg == target_seg_id
    area = int(mask.sum())
    visible = area > 0

    # ---- in_fov：目标在不在视场里 ----
    # ⚠️ 目标是**扩展物体**（直径 0.6 m 的圆柱），不是点。只判"中心投影在图内"
    #    会让目标**中心刚出图、边缘还占半屏**这种帧被算成 in_fov=False ——
    #    check_p4 实测 visible=0.925 > in_fov=0.692，还把"目标明明在画面里"
    #    判成丢失。正确判据是**中心在图内 或 掩码非空**：
    #
    #        in_fov   = in_image(投影点) or visible
    #        occluded = in_fov and not visible = in_image(投影点) and not visible
    #        丢失     = not in_fov = 中心出图 **且** 一个像素都没有
    #
    #    这样三条不变量都成立：`visible ⊆ in_fov`、`occluded ⇒ 中心在图内`、
    #    "丢失"= 真的什么都看不见（而不是"没对准中心"）。
    in_fov = in_image(u, v) or visible

    bbox = center_pixel = target_depth_mean = None
    neg1_ratio = 0.0
    if visible:
        ys, xs = np.nonzero(mask)
        # 外接矩形，闭区间（和 PIL/slice 的写法对齐，直接拿来切片正好）
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        # ⚠️ 用**掩码质心**而不是外接矩形中心 —— 感知系统实际产出的是质心。
        #    换成针孔投影后它和 projected_pixel 对到亚像素（实测残差 < 1 px），
        #    评估默认用 center_pixel。
        center_pixel = (float(xs.mean()), float(ys.mean()))
        # ⚠️ 深度均值把 -1.0 **算进去**（太近时目标整块就是 -1.0），
        #    这样 target_depth_mean ≈ -1.0 才能反映"比 MIN_RANGE 还近"的语义。
        #    拿 mask 之外的值或者丢掉 -1 都会把这件事抹平。
        #
        #    ⚠️⚠️ 代价是这个均值可能是 -1.0 和有效值的**混合**，不能当距离用。
        #    实测目标贴着相机时 mean = -0.543，直接换算距离会得到 -5.43 m。
        #    所以下面同时给 `target_depth_neg1_ratio`：掩码里有多大比例是 -1.0。
        #    ratio 高 -> 整个目标都在 MIN_RANGE 里；ratio 低 -> 只有近端超近程。
        dvals = depth[mask]
        target_depth_mean = float(dvals.mean())
        neg1_ratio = float((dvals < -0.5).mean())

    return {
        "in_fov": bool(in_fov),
        "visible": bool(visible),
        "occluded": bool(in_fov and not visible),
        "bbox": bbox,
        "center_pixel": center_pixel,
        "projected_pixel": projected_pixel,
        "target_pixel_area": area,
        "target_depth_mean": target_depth_mean,
        "distance": distance,
        # 下面几个是给调试用的旁支量，不在 §2.2 的字段表里
        "target_depth_neg1_ratio": neg1_ratio,  # 掩码里 -1.0（比 MIN_RANGE 近）的比例
        "depth_norm_geom": float(depth_norm),   # 几何算出来的 cz/MAX_RANGE（无 -1/+1 特殊值）
        "cz": float(cz),                        # 相机系 z，<=0 表示在背后
    }


def summarize(tr):
    """一行可读的摘要，给日志和 summary.md 用。"""
    cp = tr["center_pixel"]
    pp = tr["projected_pixel"]
    cp_s = "(%.1f, %.1f)" % cp if cp is not None else "None"
    pp_s = "(%.1f, %.1f)" % pp
    bb_s = ("%d,%d,%d,%d" % tr["bbox"]) if tr["bbox"] is not None else "None"
    dm = tr["target_depth_mean"]
    dm_s = "%.3f" % dm if dm is not None else "None"
    if dm is not None and tr.get("target_depth_neg1_ratio", 0.0) > 0.0:
        # 混合均值不能当距离用，把 -1.0 的占比一起打出来
        dm_s += f"(近程占{tr['target_depth_neg1_ratio']*100:.0f}%)"
    state = ("IN_FOV" if tr["in_fov"] else "OUT")
    if tr["visible"]:
        state += "+VIS"
    elif tr["in_fov"]:
        state += "+OCC"
    return (
        f"[{state}] dist={tr['distance']:.2f}m "
        f"area={tr['target_pixel_area']}px "
        f"bbox={bb_s} center={cp_s} proj={pp_s} "
        f"depth={dm_s}"
    )
