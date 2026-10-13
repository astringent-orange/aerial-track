#!/usr/bin/env python3
"""P4：episode 的指标统计 + 报告（docs/BENCH_PLAN.md §2.6）。

`EpisodeMetrics` 每步收一条记录，跑完算出 §2.6 那张表，写两个文件：

    metrics.json   机器读的（含 config_snapshot / seed / waypoints）
    summary.md     人读的

⚠️ **失败的 episode 照常把已收集的指标写出来**，只是 `result = "fail"`。
   这样能看出来是差在哪一步。

指标口径（§2.6，写死在这里，别各处各算一套）：

| 字段 | 定义 |
|---|---|
| `distance_err.{mean,std,max}` | `abs(distance − standoff)` 的统计。`bias` 是带符号均值 |
| `los_err_deg.{mean,std,max}` | **视线误差**：相机光轴 与 相机→目标连线 的夹角（deg，恒 ≥ 0）。由 `projected_pixel` 用针孔模型反推。**这才是"相机指得准不准"** |
| `yaw_err_deg.{mean,std,max}` | `abs(wrapped(target_yaw − chaser_yaw))` 的统计，deg。`bias` 带符号均值。⚠️ 这是**航向对齐**误差，**不是**视线误差 —— 目标横穿画面时相机好好盯着，它一样能到 90° |
| `pixel_err.{u_mean,u_std,v_mean,v_std}` | `center_pixel − 图像中心`，**带符号**（要看系统性偏移） |
| `visible_ratio` / `in_fov_ratio` | `visible` / `in_fov` 为真的步数占比 |
| `occluded_steps` | `occluded` 为真的**步数**（不是占比） |
| `collision_count` | 发生过碰撞的**控制周期数** |

> 掩码质心（`center_pixel`）是感知系统实际产出的东西，所以评估默认用它；
> `projected_pixel` 只在排查"是跟踪器的问题还是成像的问题"时看。

> **`target_depth_mean` 不进指标**：它是 `-1.0` 和正值的混合均值，不能当距离用
> （见 docs/SETUP_NOTES.md）。距离一律用两机欧氏距离。
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from tracking_bench.sensors import truth as T   # 只为拿 K（纯 numpy，不碰 isaacgym）

# standoff 的来源：runner 会把 tracker 配置里的 standoff 传进来。
DEFAULT_STANDOFF = 2.0


def _wrap_deg(a):
    """把角度差折到 (-180, 180]。"""
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class StepRecord:
    step: int
    t: float
    distance: float
    standoff: float
    target_yaw: float
    chaser_yaw: float
    in_fov: bool
    visible: bool
    occluded: bool
    center_pixel: tuple       # (u, v) | None
    projected_pixel: tuple    # (u, v) | None
    target_pixel_area: int
    collision: bool


@dataclass
class EpisodeMetrics:
    """收集一个 episode 的逐步记录，最后算出 §2.6 的那张表。"""

    seed: int
    wall_layout_id: str
    standoff: float = DEFAULT_STANDOFF
    lost_criterion: str = "in_fov"
    dt: float = 0.1                 # 控制周期，给 time_run 用
    records: list = field(default_factory=list)

    # 终止信息（runner 写）
    result: str = "success"          # success | fail
    fail_reason: str = None          # lost_timeout | collision | None
    fail_step: int = None
    waypoints: list = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    config_snapshot_path: str = None

    # ---- 收集 ----
    def record(self, step, t, tr, chaser_yaw, collision=False,
               target_yaw=0.0, standoff=None):
        """`tr` 是 `truth.compute_truth()` 的返回值。"""
        sd = self.standoff if standoff is None else float(standoff)
        self.records.append(StepRecord(
            step=int(step),
            t=float(t),
            distance=float(tr["distance"]),
            standoff=sd,
            target_yaw=float(target_yaw),
            chaser_yaw=float(chaser_yaw),
            in_fov=bool(tr["in_fov"]),
            visible=bool(tr["visible"]),
            occluded=bool(tr["occluded"]),
            center_pixel=None if tr["center_pixel"] is None
            else (float(tr["center_pixel"][0]), float(tr["center_pixel"][1])),
            projected_pixel=(float(tr["projected_pixel"][0]),
                             float(tr["projected_pixel"][1])),
            target_pixel_area=int(tr["target_pixel_area"]),
            collision=bool(collision),
        ))

    # ---- 派生量 ----
    @property
    def steps_run(self):
        return len(self.records)

    @property
    def time_run(self):
        return self.steps_run * self.dt

    def _series(self):
        if not self.records:
            return None
        dist = np.array([r.distance for r in self.records], dtype=np.float64)
        sd = np.array([r.standoff for r in self.records], dtype=np.float64)
        dyaw = np.array([_wrap_deg(math.degrees(r.target_yaw - r.chaser_yaw))
                         for r in self.records], dtype=np.float64)
        return dist, sd, dyaw

    @staticmethod
    def _stats_abs_err(err):
        """|误差| 的 mean/std/max + 带符号均值 bias。"""
        if err.size == 0:
            return {"mean": None, "std": None, "max": None, "bias": None}
        a = np.abs(err)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "max": float(a.max()),
            "bias": float(err.mean()),
        }

    def metrics(self):
        """算出 §2.6 的整张表。没跑任何一步时各字段是 None，不抛异常。"""
        out = {
            "result": self.result,
            "fail_reason": self.fail_reason,
            "fail_step": self.fail_step,
            "steps_run": self.steps_run,
            "time_run": None,
            "dt": self.dt,
            "distance_err": {"mean": None, "std": None, "max": None, "bias": None},
            "los_err_deg": {"mean": None, "std": None, "max": None,
                            "n_samples": 0},
            "yaw_err_deg": {"mean": None, "std": None, "max": None, "bias": None},
            "pixel_err": {"u_mean": None, "u_std": None,
                          "v_mean": None, "v_std": None, "n_samples": 0},
            "visible_ratio": None,
            "occluded_steps": None,
            "in_fov_ratio": None,
            "collision_count": None,
            "lost_criterion": self.lost_criterion,
            "standoff": self.standoff,
            "seed": self.seed,
            "wall_layout_id": self.wall_layout_id,
            "waypoints": [list(map(float, w)) for w in self.waypoints],
            "config_snapshot": self.config_snapshot,
            "config_snapshot_path": self.config_snapshot_path,
        }
        if not self.records:
            return out

        dist, sd, dyaw = self._series()
        out["time_run"] = self.time_run
        out["distance_err"] = self._stats_abs_err(dist - sd)
        out["yaw_err_deg"] = self._stats_abs_err(dyaw)

        # 像素误差：center_pixel 没掩码时退回 projected_pixel（跟踪器没看见目标，
        # 但几何投影点仍然有意义 —— 它回答"相机指得对不对"）。
        #
        # ⚠️ 目标在相机**背后**时 `truth.project()` 返回 ±1e9 的哨兵（保证 in_fov
        #    是 False 且不产生 nan）。那不是像素坐标，绝不能平均进来 —— 踩过：
        #    do_nothing 场景的 pixel_err 打出 -1000000240，baseline 打出 5e7。
        #    只统计**真的在像平面前**的样本，并把用了几个样本报出来。
        du, dv = [], []
        for r in self.records:
            p = r.center_pixel if r.center_pixel is not None else r.projected_pixel
            if p is None:
                continue
            if not (abs(p[0]) < 1e6 and abs(p[1]) < 1e6):
                continue        # 哨兵：目标在背后，没有有意义的像素坐标
            du.append(p[0] - 240.0)   # 图像中心 u
            dv.append(p[1] - 135.0)   # 图像中心 v
        if du:
            out["pixel_err"] = {
                "u_mean": float(np.mean(du)), "u_std": float(np.std(du)),
                "v_mean": float(np.mean(dv)), "v_std": float(np.std(dv)),
                "n_samples": int(len(du)),
            }

        # 视线误差：相机光轴 与 相机→目标连线 的夹角。由**几何**投影点反推
        # （针孔模型，见 truth.project）：
        #     theta = atan2(hypot(u - cx, v - cy), f)
        # 用 projected_pixel 而不是 center_pixel —— 这里要的是"相机指得准不准"，
        # 掩码质心是感知的输出，会把检测抖动算进来。
        # ⚠️ 和 yaw_err_deg 是两回事：那个量航向对齐（target_yaw − chaser_yaw），
        #    目标横穿画面时相机好好盯着，它一样能到 90°。看相机指得准不准看这个。
        # ⚠️ 同 pixel_err：目标跑到相机背后时 projected_pixel 是 ±1e9 哨兵，剔掉。
        los = []
        for r in self.records:
            p = r.projected_pixel
            if p is None:
                continue
            if not (abs(p[0]) < 1e6 and abs(p[1]) < 1e6):
                continue
            los.append(math.degrees(math.atan2(
                math.hypot(p[0] - T.CX, p[1] - T.CY), T.FX)))
        if los:
            a = np.asarray(los, dtype=np.float64)
            out["los_err_deg"] = {
                "mean": float(a.mean()), "std": float(a.std()),
                "max": float(a.max()), "n_samples": int(a.size),
            }

        n = len(self.records)
        out["visible_ratio"] = float(sum(r.visible for r in self.records) / n)
        out["in_fov_ratio"] = float(sum(r.in_fov for r in self.records) / n)
        out["occluded_steps"] = int(sum(r.occluded for r in self.records))
        out["collision_count"] = int(sum(r.collision for r in self.records))
        return out


# --------------------------------------------------------------------------- #
#  报告
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def write_metrics_json(metrics, out_dir, name="metrics.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def render_summary_md(metrics, title=None):
    """把 metrics dict 渲染成 summary.md 的正文（纯字符串，方便测试）。"""
    m = metrics
    title = title or f"Episode 摘要（seed={m['seed']}）"
    de, ye, pe = m["distance_err"], m["yaw_err_deg"], m["pixel_err"]
    le = m.get("los_err_deg") or {"mean": None, "std": None, "max": None,
                                  "n_samples": 0}
    lines = [
        f"# {title}",
        "",
        "## 结果",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| result | **{m['result']}** |",
        f"| fail_reason | {m['fail_reason'] or '—'} |",
        f"| fail_step | {m['fail_step'] if m['fail_step'] is not None else '—'} |",
        f"| steps_run | {m['steps_run']} |",
        f"| time_run | {_fmt(m['time_run'], 2)} s |",
        "",
        "## 跟踪精度",
        "",
        "| 指标 | mean | std | max | bias（带符号均值） |",
        "|---|---|---|---|---|",
        f"| distance_err [m] (`abs(距离 − standoff)`) | {_fmt(de['mean'])} "
        f"| {_fmt(de['std'])} | {_fmt(de['max'])} | {_fmt(de['bias'])} |",
        f"| los_err_deg [deg] (**视线误差**：光轴↔相机→目标) | {_fmt(le['mean'], 2)} "
        f"| {_fmt(le['std'], 2)} | {_fmt(le['max'], 2)} | — |",
        f"| yaw_err_deg [deg] (`target_yaw − chaser_yaw`，**航向对齐**) "
        f"| {_fmt(ye['mean'], 2)} "
        f"| {_fmt(ye['std'], 2)} | {_fmt(ye['max'], 2)} | {_fmt(ye['bias'], 2)} |",
        "",
        f"`pixel_err`（掩码质心相对图像中心，**带符号**）："
        f"u = {_fmt(pe['u_mean'], 2)} ± {_fmt(pe['u_std'], 2)} px，"
        f"v = {_fmt(pe['v_mean'], 2)} ± {_fmt(pe['v_std'], 2)} px",
        "",
        "## 可见性 / 安全",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| in_fov_ratio | {_fmt(m['in_fov_ratio'], 3)} |",
        f"| visible_ratio | {_fmt(m['visible_ratio'], 3)} |",
        f"| occluded_steps | {m['occluded_steps']} |",
        f"| collision_count | {m['collision_count']} |",
        f"| lost_criterion | `{m['lost_criterion']}` |",
        "",
        "> 遮挡（`in_fov=True, visible=False`）**不算丢失**：墙挡住了不是跟踪器能控制的。",
        "> 丢失判据是 `in_fov`，也就是目标中心的几何投影落在图外 —— 那一项才是"
        "跟踪器能控制的（把相机指向目标）。",
        "",
        "## 复现",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| seed | `{m['seed']}` |",
        f"| wall_layout_id | `{m['wall_layout_id']}` |",
        f"| standoff | {m['standoff']} m |",
        f"| config_snapshot | `{m['config_snapshot_path'] or '（未落盘）'}` |",
        f"| waypoints | {len(m['waypoints'])} 个 |",
        "",
    ]
    if m["waypoints"]:
        lines += ["路点序列：", "", "```"]
        for i, w in enumerate(m["waypoints"]):
            lines.append(f"P{i}: ({w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f})")
        lines += ["```", ""]
    lines += [
        "---",
        "",
        "生成：`tracking_bench` 的 `bench_eval.write_summary_md`。"
        "字段口径见 `INTERFACE.md` §指标。",
        "",
    ]
    return "\n".join(lines)


def write_summary_md(metrics, out_dir, name="summary.md", title=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_summary_md(metrics, title=title))
    return path
