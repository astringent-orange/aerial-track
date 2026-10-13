# 无人机单目标跟踪 · 测试环境交接包

**这是什么**：一个标准化的无人机单目标跟踪测试环境。一架 chaser 无人机（挂着相机）
在 20×20×5 m 的障碍场地里跟踪一架自由飞行的目标无人机；你只需要写一个 `Tracker`
类，就能跑标准化测试、拿到统一口径的指标（距离误差 / 视线误差 / 在视场率 / 碰撞…）。

**动手前的硬规则**（GPU 选卡、容器红线、结论落盘）→ [`AGENTS.md`](AGENTS.md)。

**怎么用它**（完整动线）：

```
① 构建 Docker 镜像 ──> ② 获取源码+打补丁 ──> ③ 跑通 5 分钟 demo
                                                │
        ┌───────────────┬───────────────────────┤
        ▼               ▼                       ▼
   写自己的 Tracker   改场景/调参数          拿指标做对比
   → INTERFACE.md     → tracking_bench/README §5   → /outputs/runs/*/metrics.json
```

**怎么拿到它**：

```bash
git clone git@github.com:astringent-orange/aerial-track.git   # 或 https://github.com/astringent-orange/aerial-track
# github.com 直连超时的网络（如国内服务器）：
#   git clone https://ghfast.top/https://github.com/astringent-orange/aerial-track.git
# 也可以在 GitHub 网页 Download ZIP / 下 release 附件的 aerial-track.tar.gz
```

三条命令就能跑通（细节在 §2）：

```bash
./docker/build_image.sh /path/to/IsaacGym_Preview_4_Package.tar   # ① 镜像（一次性）
bash scripts/setup_workspace.sh                                   # ② 源码 + 补丁
bash scripts/start_container.sh                                   # ③ 进容器，然后见下
```

---

## 1. 目录地图 —— 每部分干什么、下一步读哪个文件

| 目录 / 文件 | 是什么 | 下一步读哪个 |
|---|---|---|
| **本 README** | 总入口：装机步骤、目录地图、使用规则与工作流 | —— |
| **`AGENTS.md`** | 硬规则 + 目录入口（动手前必读，≤50 行） | 先扫一遍 |
| `docker/` | 镜像构建：`Dockerfile` + `build_image.sh` + `requirements.txt` | 先跑 `build_image.sh` |
| `scripts/` | 一次性环境脚本：`setup_workspace.sh`（源码+补丁）、`start_container.sh`（容器） | 按 §2 顺序跑 |
| `patches/` | 对上游 aerial_gym 的**全部本地修改**（一个 git 补丁，8 个文件） | §4 有清单；想深究 → `docs/SETUP_NOTES.md` |
| `code/tracking_bench/` | **测试环境本体**（Python 包 + 入口 + 测试 + 配置） | 它自己的 `README.md` |
| `outputs/` | 运行产物目录（挂载 = `/outputs`），带结构约定 | `outputs/README.md` |
| `docs/` | 背景资料（**可选读物**，跑起来不需要） | 见下表 |

**「你接下来该读哪个文件」对照**：

| 你想干什么 | 读哪里 |
|---|---|
| 只想先跑起来看效果 | 本文件 §2 走一遍 → 完事 |
| **写自己的跟踪算法** | `code/tracking_bench/README.md` §4 → **`code/tracking_bench/INTERFACE.md`**（接口规格，写 tracker 只看它） |
| 看懂现成代码怎么写 | `code/tracking_bench/examples/minimal_example.py` |
| 改场地 / 加障碍 / 调目标运动 | `code/tracking_bench/README.md` §5 |
| 指标口径（多准算好、怎么判失败） | `code/tracking_bench/INTERFACE.md` §6 |
| 用 coding agent 长期迭代 | `AGENTS.md`（红线）→ 本文件 §3（工作流与条目模板） |
| 产物写哪、目录怎么组织 | `outputs/README.md` |
| 出问题排查 | `code/tracking_bench/README.md` §6（速查表）→ 还不行 → `docs/SETUP_NOTES.md` |
| 了解设计取舍与验收数字 | `docs/BENCH_PLAN.md` |
| 想改仿真内核 / 加传感器 | `docs/SETUP_NOTES.md` + `patches/aerial_gym_local.patch`（注释很全） |

### 各文件一句话

```
aerial-track/（clone / 解压后就是它）
├── README.md                ← 你在这
├── AGENTS.md                硬规则 + 目录入口（coding agent 动手前必读）
├── docker/
│   ├── Dockerfile           镜像定义：IsaacGym + gymutil 补丁 + 全部 pip 依赖
│   ├── build_image.sh       从 IsaacGym 官方压缩包构建镜像
│   ├── requirements.txt     pip 依赖清单（Dockerfile 自动装，留档核对用）
│   └── nvidia_icd.json / 10_nvidia.json   Vulkan/EGL 配置（构建时拷进镜像）
├── scripts/
│   ├── setup_workspace.sh   clone aerial_gym + urdfpy → 打 patches/ 里的补丁
│   └── start_container.sh   创建/进入容器（挂 code/ → /workspace 与 outputs/ → /outputs）
├── patches/
│   └── aerial_gym_local.patch   对上游 aerial_gym 的全部本地修改（含新文件）
├── code/
│   └── tracking_bench/      测试环境本体（另有 README.md → INTERFACE.md 两级文档）
├── outputs/                 运行产物（视频/指标/图），结构约定见其 README.md
└── docs/
    ├── BENCH_PLAN.md        设计方案 + 验收实测数字（可选）
    └── SETUP_NOTES.md       环境原理 + 踩坑全记录（可选）
```

> `code/` 下的 `aerial_gym_simulator/` 和 `urdfpy/` **现在没有**，是步骤 ② 生成的。

---

## 2. 安装（三步，一次性）

**硬件**：一台带 NVIDIA GPU 的 Linux 机器 + Docker + NVIDIA Container Toolkit。
显存 ≥ 8 GB 足够（默认单环境 480×270 深度渲染）。

### 步骤 ① 构建 Docker 镜像

IsaacGym Preview 4 是 **NVIDIA 许可软件，不随本包分发**——自行到 NVIDIA
开发者渠道申请 `IsaacGym_Preview_4_Package.tar`，然后：

```bash
./docker/build_image.sh /path/to/IsaacGym_Preview_4_Package.tar
```

脚本会解包到临时目录、覆盖成本包的 Dockerfile（IsaacGym 官方那份没有
gymutil 补丁和 pip 依赖）、`docker build` 出镜像 `isaacgym`。
构建会传入你宿主机的 uid/gid，保证挂载目录里的文件归你所有。

### 步骤 ② 获取源码 + 打补丁

```bash
bash scripts/setup_workspace.sh
# 直连 github.com 超时的网络（如国内服务器）：
#   GITHUB_BASE=https://ghfast.top/https://github.com bash scripts/setup_workspace.sh
```

这一步 clone 两个上游开源仓库并钉死版本，然后打上本包的补丁：

| 仓库 | 版本 | 交接包里带的 |
|---|---|---|
| `ntnu-arl/aerial_gym_simulator` | 2.0.1（f0d0f05） | 只带 `patches/aerial_gym_local.patch` |
| `mmatl/urdfpy` | 5466842 | 什么都不带（PyPI 版有 bug，必须用 GitHub master） |

跑完应该看到 8 个文件的本地修改（3 改 + 5 增，§4 有清单）。

### 步骤 ③ 生成 URDF、跑回归、跑 demo（容器内）

```bash
bash scripts/start_container.sh        # 进入容器，已在 /workspace
cd /workspace/tracking_bench

# 生成场景 URDF（墙 / 圆柱 / 目标，每个生成器跑一次）
python assets/make_wall_assets.py
python assets/make_cylinder_assets.py --count 12
python assets/make_target_assets.py

# 离线回归（不用 GPU，几十秒，应 5/5 全绿）
python tests/run_all.py

# 5 分钟 demo（60 步 ≈ 6 s 仿真；cuda:N 换成空闲卡号）
python scripts/run_bench.py --config configs/bench.yaml \
    --steps 60 --device cuda:0 --out /outputs/runs/demo
cat /outputs/runs/demo/summary.md
```

> **输出约定**：产物一律写挂载的 `/outputs/`（= 宿主 `outputs/`），别写回代码目录。
> 结构（`runs/` `videos/` `figures/` `scratch/`，带日期命名、**不要平铺**）见
> [`outputs/README.md`](outputs/README.md)。

到这里环境就通了。**接下来**：`code/tracking_bench/README.md`（怎么写 tracker、
怎么改场景）→ `code/tracking_bench/INTERFACE.md`（接口规格）。

---

## 3. 使用规则与工作流（coding agent 必读）

**总原则（也是 `AGENTS.md` 的硬规则）：结论落盘，对话即弃 —— 只存在于对话里的
信息，下一个 agent 看不见。**

**开工**：`AGENTS.md`（红线）→ 本文件（地图）→ 任务相关文档章节 →
自己实验日志的**最后一条**（当前状态只认这里，别翻聊天记录）。

**收工**（三件事，缺一不可）：

1. **进度条目**写进实验日志（接着 `docs/BENCH_PLAN.md` §6 往后写，或自建
   `docs/EXPERIMENT_LOG.md`，格式一样）：

   ```
   **YYYY-MM-DD　<阶段/任务名>　✅/❌ <一句话结果>**
   - 做了：<改动清单，指到文件>
   - 验收数字：<命令 + 实测值，没数字的结论不许写>
   - 踩的坑：<编号指向 docs/SETUP_NOTES.md 对应条目，没有则写"无">
   - 下一步：<具体到可执行>
   ```

2. **踩坑条目**写进 `docs/SETUP_NOTES.md`（追加，每条四行）：

   ```
   ### ⚠️ <一句话现象>
   **现象**：…　**根因**：…　**修法**：…　**怎么验证**：…
   ```

3. 刷新相关文档头部的「最近更新」行；新约定（目录结构、接口、脚本用法）
   同步进相应 `README.md`。

**信息写哪**：

| 文件 | 放什么 |
|---|---|
| `AGENTS.md` | 硬规则 + 目录入口（≤50 行，只写规则） |
| 本 `README.md` | 目录地图、装机、使用规则与工作流 |
| `docs/BENCH_PLAN.md` | 设计 + 接口规格 + 进度记录（§6） |
| `docs/SETUP_NOTES.md` | 原理 + 踩坑（现象/根因/修法/验证） |
| `code/tracking_bench/README.md` / `INTERFACE.md` | 测试环境怎么用 / 接口规格 |

---

## 4. 本包对上游 aerial_gym 改了什么（`patches/aerial_gym_local.patch`）

一个补丁 8 个文件。**丢了会静默出错**的只有第一行那条：

| 文件 | 改动 | 丢了会怎样 |
|---|---|---|
| `env_manager/warp_env_manager.py` | **改**：`post_physics_step()` 里补调 `update_meshes()`（warp 网格跟动物体） | 深度图里移动物体**冻在出生点**，不报错 |
| `config/sensor_config/.../base_depth_camera_config.py` | **改**：`randomize_placement=False`（外参固定） | 相机位姿每次 reset 抖 ±2.5cm/±5°，算不准 |
| `env_manager/__init__.py` | **改**：注册 `tracking_env`、`empty_with_cylinders` 两个环境名 | `--env tracking_env` 找不到 |
| `config/env_config/tracking_env.py` | **增**：跟踪实验环境（20×20 场地 + 墙 + 圆柱 + 目标，`reset_on_collision=False`） | 没有测试环境 |
| `config/env_config/empty_with_cylinders.py` | **增**：10×10 调试环境 | 少个调试场地（不影响主流程） |
| `config/asset_config/{cylinder,target,wall}_asset_config.py` | **增**：三种环境资产的配置 | 没有目标/墙/圆柱 |

场景 URDF（墙/圆柱/目标的 .urdf 文件）**不在补丁里**——由 `code/tracking_bench/assets/`
下的生成器重建（步骤 ③ 已含）。场地布局的唯一来源是
`code/tracking_bench/configs/arena_20x20.yaml`，改布局 → 重跑生成器 → `tests/run_all.py`
的槽位回归会替你把关。

---

## 5. 有意不打包的东西（为什么、怎么拿）

| 不带 | 为什么 | 怎么拿 |
|---|---|---|
| Docker 镜像（~27 GB） | Dockerfile 可完整复现 | 步骤 ① 一条命令 |
| IsaacGym 压缩包 | NVIDIA 许可，不允许再分发 | 自行申请，见步骤 ① |
| aerial_gym / urdfpy 源码 | 上游开源仓库，带补丁比重传源码小两个数量级 | 步骤 ② 一条命令 |
| pip 安装包（~600 MB） | 全部烧进镜像，重建容器不丢 | 步骤 ① 自动 |
| 场景 URDF | 纯生成物，改布局本来就要重生成 | 步骤 ③ 三条命令 |
| 开发过程脚本（录视频、诊断、阶段 0/1 脚本） | 与测试环境无关 | 无（需要再说） |

---

## 6. GPU 使用注意（共享机器）

- **任何程序最多同时用 2 张 GPU**。默认路径（warp 渲染，深度+分割）**只占 1 张**；
  要 RGB（`use_warp: false`）才占 2 张。
- 挑空闲卡看**显存**不看利用率（`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`），
  有人的闲置进程长期占着显存、利用率 0%。
- 跑测试显式给 `--device cuda:N`；**不要自己设 `CUDA_VISIBLE_DEVICES`**
  （脚本自己会锁到单卡并重 exec，你设了反而会多占卡/崩）。

---

## 7. 出问题先查这里

| 现象 | 原因 → 修法 |
|---|---|
| `useradd: UID already in use` 构建失败 | 你的 uid 和基础镜像用户撞了 → 改 `Dockerfile` 里 `ARG UID` 的默认值重 build |
| `setup_workspace.sh` clone 超时 | github.com 被墙 → `GITHUB_BASE=https://ghfast.top/https://github.com` 重跑 |
| 补丁打不上 | aerial_gym 版本不对 → 删掉 `code/aerial_gym_simulator` 重跑步骤 ② |
| `from urdfpy import URDF` 报 ImportError | 步骤 ② 没跑完，或没从 `start_container.sh` 进容器（PYTHONPATH 在镜像里指向 `/workspace/urdfpy`） |
| 深度图全是 `+1.0` | 超 10 m 或没打中（分不开）；或建完环境没 reset（相机位姿是 NaN） |
| 深度有负数 | **不是 bug**：`-1.0` = 比 0.2 m 还近，值域 `[-1]∪[0,1]` |
| 目标位置/深度冻在出生点 | warp 补丁丢了 → `tests/regression/test_warp_patch.py` 会红，重打补丁 |
| `import` YAML 报 `UnicodeDecodeError` | isaacgym 把默认编码变 ASCII 了，`open()` 显式 `encoding="utf-8"` |
| 一张卡变两张 | 自己设了 `CUDA_VISIBLE_DEVICES`，或开了 RGB |

再不行 → `docs/SETUP_NOTES.md`（踩坑全记录）→ `code/tracking_bench/README.md` §7（完整验收命令）。
