"""把进程锁到一张物理 GPU 上 —— 必须在 import torch / aerial_gym **之前**用。

为什么需要它（和 record_sim.py 里那段是同一个原因，这里抽出来复用）：

    aerial_gym/utils/math.py 里大量 @torch.jit.script 装饰器会在 **import 期**
    编译并初始化 CUDA，而且固定建在**物理 0 号卡**上（约 450MB）。
    torch.cuda.set_device() 改不动它。所以只要仿真不在 0 号卡，进程就会实打实
    占两张卡 —— 一张跑仿真，一张放这个甩不掉的上下文。

    唯一有效的办法是在 CUDA 初始化之前设 CUDA_VISIBLE_DEVICES：物理卡 N 被映射成
    逻辑 0 号卡，那个上下文和仿真就都落在 N 上。但环境变量对已加载 CUDA 的进程
    无效，所以这里用 os.execvpe 把自己换掉重跑一遍（对 shell 透明）。

用法（在任何 import torch 之前）：

    from tracking_bench.sim import gpu_lock
    gpu_lock.confine_to_one_gpu()

    import isaacgym      # IsaacGym 要在 torch 之前
    import torch
    ...

传 `--device cuda:N` 就会锁到物理 N 号卡。已经设过 CUDA_VISIBLE_DEVICES 时
尊重现状不动作（此时 --device 是逻辑卡号）。

⚠️ 不能用于 IsaacGym 相机（RGB）路径：CUDA 索引被重映射后和 Vulkan 的物理编号
对不上，IsaacGym 直接 segfault。那条路只能占 2 张卡，见 AGENTS.md。
"""

import argparse
import os
import sys


def _rewrite_device(argv, new_value):
    """把 argv 里的 --device 改成 new_value（`--device X` 和 `--device=X` 都认）。"""
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--device":
            out += ["--device", new_value]
            i += 2
            continue
        if a.startswith("--device="):
            out.append("--device=" + new_value)
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def confine_to_one_gpu(allow_rgb=False):
    """如果需要，重新 exec 自己到单张物理卡上。

    allow_rgb: True 时允许 --use-warp False（IsaacGym 相机）路径，那条路不重映射，
               只在 stderr 提醒它占 2 张卡。False 时直接拒绝，避免误用。
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--use-warp", default=True, type=lambda s: s.lower() != "false")
    early, _ = pre.parse_known_args()

    if not early.device.startswith("cuda:"):
        return  # 跑 CPU 之类，不用管

    if not early.use_warp:
        if not allow_rgb:
            sys.exit(
                "这个脚本只支持 warp 传感器（depth/seg，单卡）。\n"
                "IsaacGym 相机（--use-warp False）需要 CUDA_VISIBLE_DEVICES 不被重映射，\n"
                "那样一定占 2 张卡，而且换到别的脚本上容易 segfault。\n"
                "要 RGB 请用 record_sim.py --use-warp False --view rgb --device cuda:N"
            )
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            sys.exit(
                f"检测到 CUDA_VISIBLE_DEVICES={cvd}，同时又用 IsaacGym 相机（--use-warp False）。\n"
                "这会让 CUDA 和 Vulkan 指向不同的 GPU，IsaacGym 会 segfault。\n"
                "请去掉 CUDA_VISIBLE_DEVICES，用 --device 指定物理卡号。"
            )
        if early.device != "cuda:0":
            print(
                f"[gpu_lock] 注意：IsaacGym 相机会占 2 张卡 —— 仿真用 {early.device}，"
                "另外物理 0 号卡上还有一个约 450MB 的 CUDA 上下文（aerial_gym import 留下的）。",
                file=sys.stderr,
                flush=True,
            )
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        # 用户自己设了，尊重它（此时 --device 是逻辑卡号，通常就是 cuda:0）
        return

    idx = int(early.device.split(":")[1])
    if idx == 0:
        return  # 本来就在物理 0 号卡，上下文和它是同一张，已经是单卡

    # flush 是必须的：容器是 Python 3.8，stderr 被重定向（非 tty）时是**块缓冲**的，
    # 而 os.execvpe 不会 flush —— 不写 flush=True 这行提示会被直接丢掉。
    print(
        f"[gpu_lock] 单卡模式：锁到物理 {idx} 号卡"
        f"（自动设 CUDA_VISIBLE_DEVICES={idx}，然后重新 exec 自己）",
        file=sys.stderr,
        flush=True,
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
    argv = [sys.executable, os.path.abspath(sys.argv[0])]
    argv += _rewrite_device(sys.argv[1:], "cuda:0")  # 重映射后逻辑 0 号就是物理 idx
    try:
        os.execvpe(sys.executable, argv, env)
    except OSError as e:
        sys.exit(
            f"重新 exec 失败（{e}），无法保证单卡。请手动：\n"
            f"    CUDA_VISIBLE_DEVICES={idx} python {sys.argv[0]} --device cuda:0"
        )
