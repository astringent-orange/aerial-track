# Aerial Gym 仿真环境 — 安装记录

## 目录结构

> 完整地图（含挂载、产物结构、工作流）→ 根目录 `README.md`。这里只列
> 与安装相关的：

```
aerialgym_ws/
├── IsaacGym_Preview_4_Package.tar   # IsaacGym 安装包
├── isaacgym/                        # IsaacGym 源码 + Dockerfile（docker build 的 context）
├── code/                            # -> 容器内 /workspace（bind mount，代码在这里）
│   ├── aerial_gym_simulator/        # Aerial Gym v2.0.1（已打本地补丁）
│   ├── urdfpy/                      # 从源码安装（pip 版本有 cylinder mesh 问题）
│   └── tracking_bench/              # 测试环境（见其 README.md）
├── outputs/                         # -> 容器内 /outputs（运行产物，结构见 outputs/README.md）
├── container_home/.local/           # -> 容器内 ~/.local（bind mount，pip 装的包装在这）
├── docs/                            # 文档（本文件在这里）
├── aerial-track/ + make_aerial_track.sh       # 交接包及其打包脚本
└── start_container.sh               # 启动/进入容器
```

## 容器

- 镜像 `isaacgym:latest`，基于 `nvcr.io/nvidia/pytorch:22.09-py3`
- 容器 `aerialgym_dev`，Python 3.8.13
- **以 uid/gid 1019 运行**（与宿主 `zhangmian` 一致），所以 `/workspace` 下
  新建的文件在宿主上直接归你所有，无需 chmod 777
- 进入容器：`./start_container.sh` 或 `docker exec -it aerialgym_dev bash`

两个 bind mount：

| 宿主 | 容器内 | 为什么 |
|---|---|---|
| `code/` | `/workspace` | 代码 |
| `container_home/.local/` | `/home/gymuser/.local` | **pip 装的包** |

> ⚠️ 第二个 mount 很关键。pip 是以 `gymuser` 身份装的，包都落在 `~/.local`
> （conda 的 site-packages 是 root 的，写不进去）。如果不把 `~/.local` 挂出来，
> 它就在容器的可写层里，**`docker rm` 一次就把整个环境（pytorch3d、warp、
> numpy……606MB）删干净了**，然后要重新下 72MB 的 pytorch3d wheel 重装。
> 挂出来之后随便重建容器都不丢。

Dockerfile 里额外做了两件事（都已烘焙进镜像）：

1. `useradd --uid 1019 --gid 1019 gymuser`
2. 给 IsaacGym 打补丁 `gymutil.py`：`args = parser.parse_args()` →
   `args, _ = parser.parse_known_args()`，否则 IsaacGym 的 argparser 会吃掉
   RL 框架的 CLI 参数。

## GPU 怎么用

容器用 `--gpus all` 创建 —— **8 张卡全都可见**，但「可见」不等于「占用」，
不影响别人。要选卡就在**每次跑的时候**加环境变量：

```bash
nvidia-smi                                    # 先看哪张空
CUDA_VISIBLE_DEVICES=5 python position_control_example.py --headless True
```

- `cuda:0`（代码里写死的）会自动映射到你选的那张物理卡，**不用改代码**
- 不加这个变量时，默认用物理 0 号卡
- 换卡不需要重建容器

> 之前用的是 `--gpus "device=3"`，那是**创建容器时焊死**的，运行中改不了，
> 想换卡只能删掉容器重建 —— 所以改成了 `all` + `CUDA_VISIBLE_DEVICES`。

（`~/.cache/torch_extensions` 没挂出来，每次新容器会自动重新编译 gymtorch，
几秒钟的事，不用管。）

### ⚠️ 实际上每个进程会占 2 张卡，不是 1 张

实测（2026-09-23）发现：**只要 import 了 `aerial_gym`，进程就会在物理 0 号卡上
额外建一个约 450MB 的 CUDA 上下文**，跟你把仿真放在哪张卡无关。

```
--device cuda:5 的实际占用：
  GPU 5 : 约 5.2 GB   <- 仿真 + 渲染
  GPU 0 : 约 450 MB   <- 甩不掉的上下文
```

根因是 `aerial_gym/utils/math.py` 里大量使用 `@torch.jit.script` 装饰器 ——
TorchScript 在 import 期就要编译，编译时初始化 CUDA，而且**固定落在 0 号卡**。
验证过程：`import aerial_gym.utils.math` 前后用 nvidia-smi 对比，稳定 +452 MiB，
重复两次都一致。

试过的无效办法：
- `torch.cuda.set_device(4)` 之后再 import —— 没用，照样落在 0 号卡
- 把 import 顺序提前 —— 没用

**唯一有效的办法是启动时设 `CUDA_VISIBLE_DEVICES`**（实测有效：设了之后
`--device cuda:0` 只会用到那一张物理卡，0 号卡纹丝不动）。但它只适用于
warp 传感器路径；IsaacGym 相机路径设了会 segfault（见下文）。

`record_sim.py` 已经**自动**做了这件事：非 RGB 路径下如果你传了非 0 的物理卡号
（`--device cuda:5`），脚本会在 import torch / aerial_gym **之前**用 `os.execvpe`
把自己重新跑一遍，带上 `CUDA_VISIBLE_DEVICES=5` 并把 `--device` 改写成 `cuda:0`。
换进程对 shell 透明（pid、stdout/stderr 重定向都不变）。

```bash
$ python record_sim.py --steps 150 --device cuda:5
[record_sim] 单卡模式：把进程锁到物理 5 号卡（自动设 CUDA_VISIBLE_DEVICES=5，然后重新 exec 自己）
# 实测：运行中我们的进程只出现在物理 GPU 5（4914 MiB），0 号卡上没有任何我们的进程
```

所以现在：
- 跑 depth/seg（**默认**）：`python record_sim.py --device cuda:5 ...` → **1 张卡** ✅
- 跑 RGB：`python record_sim.py --use-warp False --view rgb --device cuda:5 ...`
  → **2 张卡**，躲不掉，别在同一时间再起别的任务

RGB 必须**显式**写 `--view rgb` 或 `--view all` 才会出；`--view auto` 的偏好顺序是
depth → seg → rgb，故意把 rgb 放最后。

> 踩坑记录：`os.execvpe` **不会 flush Python 的缓冲区**。容器是 Python 3.8，
> stderr 被重定向（非 tty）时是**块缓冲**的，所以 exec 前打印的提示会被整条丢掉。
> 那段 print 必须写 `flush=True`。（Python 3.9+ 的 stderr 恒为行缓冲，所以在
> 宿主机上测同样的代码是正常的，只有容器里会丢 —— 这个差异一开始把排查带偏了。）

### 选卡时：看显存，不要看利用率

`nvidia-smi` 的 `utilization` **不能**用来判断卡是不是空的。这台机器上有别人
一堆 idle 的容器长期占着显存（实测：8 个 `aerial-gym:isaacgym-p4` 的容器
各占 4.5~5.2GB，利用率 0%，已经挂了 6 小时）。利用率 0% 但显存 5GB 占着的卡，
照样不能算空。

判断顺序：
1. `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
2. **先看 memory.used**，5GB 以上的基本都是有东西的
3. 想知道具体是谁占的，见下面「怎么查是谁在用卡」

### 怎么查是谁在用卡 / 有没有重复开容器

docker **没有按用户隔离**，`docker ps -a` 看到的是**整台机器所有人的容器**
（实测 1976 个）。所以里面绝大多数跟我们无关，别被吓到，也别去动。

查「某个 pid 占的是哪张卡、属于哪个容器、哪个用户」：

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader          # 卡号 ↔ UUID
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -o user=,cmd= -p <pid>                                        # 谁在跑什么
docker top <容器名> -eo pid | grep -w <pid>                       # 属于哪个容器
```

> 两个坑：
> - `docker top` 给的是**宿主机 pid**，能直接和 `nvidia-smi` 对上；
>   `docker exec` 进去看到的 pid 是容器命名空间里的，对不上。
> - `docker top <容器> | grep <脚本名>` 会先匹配上 bash 包装进程，
>   要按 pid 列精确匹配。

查「我们的镜像有没有被重复开容器」：

```bash
docker ps -a --filter ancestor=isaacgym --format 'table {{.Names}}\t{{.Status}}'
```

2026-09-23 查的结果：**只有 `aerialgym_dev` 一个容器**，没有重复。
另外 `isaacgym:latest` 和 `isaacgym:pre-uid1019` 两个 tag 共享 57/60 层，
只有改 uid 那几层不同，第二个 tag 的实际磁盘增量是几百 MB 级，
不是 `docker images` 显示的 27.1GB（那个数字含共享层，两个 tag 会重复计）。

## 网络：GitHub 被墙

这台机器 **直连 github.com 会超时**（pypi / 清华源 / gitee 正常）。
容器内已经配好 git 重写，直接用原版 github URL 即可：

```bash
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
```

备选镜像（当时测试可用）：`ghfast.top`、`ghproxy.net`、`gh-proxy.com`。

两个排查时容易误导的点：

- **`git ls-remote` 偶尔能通不代表 github 可达** —— 它可能走通了瞬时路径，
  `git clone` 照样超时。判断通不通要试真 clone。
- **git 重写配在容器的 git config 里，重建容器会丢** —— 每次
  `docker rm`/重建 `aerialgym_dev` 后要重新执行上面那条 `git config`。

**推送/拉自己的仓库走 SSH 443，不用镜像**（2026-10-13 实测）：`ssh.github.com:443`
可达且本机密钥已注册（GitHub 账号 `astringent-orange`），宿主 `~/.ssh/config` 配了

```
Host github.com
    HostName ssh.github.com
    Port 443
```

之后 `git@github.com:OWNER/REPO.git` 照常 push/pull。该配置在**宿主 home**，
不在容器里，重建容器不受影响。`api.github.com` 也可直连（管理 API 能用）。
用途：`aerial-track` 仓库（github.com/astringent-orange/aerial-track）就靠它推送。

pip 走 pypi.org（可通，约 5s 延迟）。若想加速可加环境变量：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pip install ...
```

> 注：`/opt/conda/pip.conf` 是 site 级配置，优先级高于 `~/.config/pip/pip.conf`，
> 所以用户级 pip.conf 不生效，需要用环境变量覆盖。

## 已安装的关键版本

| 包 | 版本 | 说明 |
|---|---|---|
| isaacgym | 1.0rc4 | editable，`/opt/isaacgym/python` |
| aerial-gym | 2.0.1 | editable，`--no-deps` 装的 |
| torch | 1.13.0a0 | 镜像自带 NVIDIA 构建 |
| pytorch3d | 0.7.3 | 预编译 wheel（见下） |
| warp-lang | 1.0.0 | 传感器/光线投射渲染 |
| numpy | 1.23.5 | 见下方说明 |
| scipy | 1.10.1 | 为满足 numpy 1.23 而升级 |
| trimesh | 4.12.2 | |
| urdfpy | 0.0.22 | 源码安装，pin 了 `networkx==2.2` |
| gymnasium | 1.1.1 | |
| gym | 0.26.2 | `position_setpoint_task` 里 `from gym.spaces import ...` |

### 依赖上的几个坑

- **pytorch3d 是必需的**，不是只有训练才用：`position_control.py`、
  `base_lee_controller.py`、`robot_manager.py` 这些核心控制器都 import 它。
  预编译 wheel（torch 1.13 + cu117 + py3.8）：
  ```
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/pytorch3d-0.7.3-cp38-cp38-linux_x86_64.whl
  ```
  必须 `pip install --no-deps`，否则 pip 会把镜像自带的 torch 换成官方版。
  另需 `pip install fvcore iopath`。

- **numpy 1.23 vs scipy 1.6.3**：镜像自带 scipy 1.6.3 要求 `numpy<1.23`，
  会让 pip 把 numpy 降到 1.22（和 aerial_gym 的 `numpy==1.23` 冲突）。
  解法是一起升级：`pip install "numpy==1.23.5" "scipy==1.10.1"`。

- **urdfpy 必须源码装**，且它 pin 了 `networkx==2.2`、`pycollada==0.6`。
  aerial_gym 本身不 import networkx（只在 requirements.txt 里提了一句
  `networkx==3.1`），所以 networkx 2.2 可以保留。

- **显式 `import isaacgym` 要写在 `import torch` 之前**，否则报
  "PyTorch was imported before isaacgym modules"。

## 验证状态

| 测试 | 命令 | 结果 |
|---|---|---|
| 位置控制（无渲染） | `python position_control_example.py --headless True --num_envs 64` | ✅ 10000 步，约 62s，exit 0 |
| warp 传感器渲染 | `python /workspace/aerial_gym_smoke_test.py` | ✅ depth (4,1,270,480) + segmentation，已存图 |

渲染出图：`code/smoke_depth.png`（能看到障碍物盒子和墙面）。

注意 `--headless` 需要**带值**（`--headless True`），不是 flag。

运行日志里这几条是无害的：
- `allocation matrix is not full rank. Rank: 4` / `Can't create empty tensor` — 退出时的提示
- `[isaacgym:gymutil.py] Unknown args: []` — 正是补丁生效的标志
- `motor_model.py` 的 `torch.tensor(sourceTensor)` UserWarning — 上游代码的写法问题

## 还没装（下一步）

按你的选择，这一步只装了核心仿真，**RL 训练依赖没装**：

- `rl-games`
- `sample-factory`

需要训练时再装（参考 `aerial_gym_simulator/requirements.txt` 和官方文档
`docs/6_rl_training.md`）。

## 怎么看画面（无头服务器）

**先说一条会白费功夫的：`ssh -X` / X11 转发对 IsaacGym 没用。**
IsaacGym 用 Vulkan 渲染，而 Vulkan 从设计上就没有网络透明性 ——
画面没法通过 X11 转发到你本地。别在这上面花时间。

### 方案一：录成视频 → 在 VSCode 里看（已跑通，推荐）

```bash
docker exec -it aerialgym_dev bash
cd /workspace
python record_sim.py --steps 300 --out my_run.mp4     # 深度图
python record_sim.py --view seg --out my_seg.mp4      # 分割图（彩色）
python record_sim.py --hover --out hover.mp4          # 原地悬停
# 三路同录（RGB|深度|分割，跟踪实验用这个）：
python record_sim.py --view all --use-warp False --device cuda:5 --out tri.mp4
```

视频写进 `/workspace`，也就是宿主的 `code/`，VSCode 里直接双击就能播。
现成的例子：`depth_demo.mp4`、`seg_demo.mp4`、`rgb_demo.mp4`、`tri_demo.mp4`。

`--reset-every` 控制多久换一批随机目标点（默认 100 步）。不换的话无人机会
一头撞进障碍物，然后画面就一直贴着障碍物不动了。

### 方案二：实时看（VNC）—— 还没做

要装 TurboVNC + VirtualGL，而且 Vulkan 能不能在虚拟显示上正常工作有
不确定性，需要实验。想要的话再说。

### 踩到的坑：CUDA_VISIBLE_DEVICES 和 IsaacGym 相机会打架

两条渲染路径对选卡的要求**不一样**：

| 路径 | 怎么选卡 |
|---|---|
| warp 传感器（depth/seg） | `CUDA_VISIBLE_DEVICES=5 python ...` ✅ |
| IsaacGym 相机 / viewer | ❌ 不能设该变量，要 `--device cuda:5` |

原因：`aerial_gym` 直接把 `graphics_device_id` 设成 `sim_device_id`
（`IGE_env_manager.py:118`），而 IsaacGym 的 `create_sim` 用的是**物理**编号；
但 `CUDA_VISIBLE_DEVICES` 只重映射 CUDA 的编号，**Vulkan 不受影响**。
两者混用 → CUDA 和 Vulkan 指向不同的卡 → **直接 segfault**（不是报错，是崩）。

`record_sim.py` 里加了拦截，会提前报错而不是崩掉。

### 关于 RGB 真彩画面：可用 ✅

**结论：RGB 是能用的**，之前判断"不通"是我自己的 bug，不是环境问题。

三个要点：

1. **RGB 必须走 IsaacGym 相机路径**：`--use-warp False`。warp 传感器只出深度和分割，
   上游代码里明确写了 warp 的 RGB 还没做（"RGB cam support for Warp would be added
   on our end. Please use Isaac Gym's native RGB Camera till then."）。

2. **`rgb_pixels` 是 RGBA 四通道，不是三通道**，形状 `(num_envs, num_cams, H, W, 4)`。
   直接喂给视频编码器会糊成一片纯色 —— 这就是我一开始误判"RGB 不通"的原因。
   用之前要裁掉 alpha：`frame[..., :3]`。

3. **这条路径退出时会 segfault**（视频已经写完了，纯粹是 IsaacGym 析构的问题）。
   `record_sim.py` 里用 `os._exit(0)` 绕开了，退出码是干净的 0。

```bash
python record_sim.py --view rgb --use-warp False --device cuda:5 --out rgb.mp4
```

再提醒一次：RGB 路径**不能**设 `CUDA_VISIBLE_DEVICES`，要用 `--device cuda:<物理卡号>`。

### RGB + 深度 + 分割 同帧拿到（做跟踪实验用这个）

IsaacGym 原生相机**渲染一次就同时给出三路图**，而且是同一个相机、同一帧、
逐像素对齐的。实测 `global_tensor_dict` 里同时有：

| 张量 | 形状 | 含义 |
|---|---|---|
| `rgb_pixels` | `(N, 1, 270, 480, 4)` | 彩色，RGBA |
| `depth_range_pixels` | `(N, 1, 270, 480)` | 深度（已翻正） |
| `segmentation_pixels` | `(N, 1, 270, 480)` | 逐物体实例分割 |

对齐关系可以这样验证：RGB 里那块红色板子，在分割图里是同一位置的一块橙色，
在深度图里是同一位置的深色矩形。**三路是同一帧，不用做时间同步。**

```bash
python record_sim.py --view all --use-warp False --device cuda:5 --out tri.mp4
```

输出是三个画面横向拼接（RGB | 深度 | 分割），每路 480 宽，总共 1440 宽。
样例见 `tri_demo.mp4`。

### 相机本身的关键参数（跟踪实验要调的）

相机挂在机体上，**朝机体正前方**（`base_depth_camera_config.py`）：

| 参数 | 值 | 说明 |
|---|---|---|
| `nominal_position` | `[0.10, 0.0, 0.03]` | 机头前方 10cm、上方 3cm |
| `nominal_orientation_euler_deg` | `[0.0, 0.0, 0.0]` | 不旋转 = 正前方 |
| `horizontal_fov_deg` | `87.0` | 水平视场角 |
| `max_range` | `10.0` | **最远只能看 10 米** |
| `min_range` | `0.2` | 20cm 以内看不见 |

`stereo_camera_config.py` 只在此基础上加了 `baseline = -0.095`（双目基线）。
注意 `num_sensors = 1` —— 名字叫 stereo，但默认只出**一路**图，
不是左右两张。

> ⚠️ **跟踪实验要先想清楚"看谁"**：这个相机在无人机自己身上、朝前看，
> 所以它看到的是**前方场景**，看不到无人机自己。如果要跟踪一个目标，
> 得先有那个目标（默认环境里只有静止的箱子和板子，没有移动物体）；
> 如果是要从外部拍无人机，那需要另加一个不挂在机体上的相机。

### ⚠️ 深度图的数值含义：值域是 [-1] ∪ [0, 1]，不是 [0, 1]

（2026-09-24 查清。这条很关键，我在 `record_sim.py` 里一开始**理解反了**。）

`depth_range_pixels` 归一化后的取值（见 `base_depth_camera_config.py` 的
`far/near_out_of_range_value`，以及 `warp_sensor.py:218-225`）：

| 值 | 含义 | 该画成 |
|---|---|---|
| `+1.0` | 超出 `max_range`（10m）**或**射线没打中 —— **两者同值，分不开** | 白（最远） |
| `-1.0` | 比 `min_range`（0.2m）还近 | 黑（最近） |
| `[0, 1]` | 真实深度 / `max_range` | 按灰度 |

两个容易踩的点：

1. **负数不是"没打中"，是"太近"**。射线打空时 kernel 写的是
   `NO_HIT_RAY_VAL = 1000.0`（**正数**），随后被 `pixels > max_range`
   那条规则一起换成 `+max_range`，归一化后就是 `+1.0`。
2. **"没打中"和"10m 处的真实命中"在数据里无法区分**。想要区分只能提高
   `max_range`（让 10m 内的东西都算命中），或者用分割图/点云另想办法。

### ⚠️ `create_ground_plane` 加的地面，warp 传感器**看不见**

（2026-09-24 实测。做 depth/seg 的都要知道。）

`create_ground_plane = True` 只调用 `gym.add_ground()`，加的是 IsaacGym 的
**解析平面** —— 它参与物理碰撞，但**没有三角面片**。而 warp 传感器的射线
只打 `CONST_WARP_MESH_ID_LIST`，这个列表在
`warp_env_manager.py:124` 完全由 **asset 的三角 mesh** 拼出来。

后果：**地面不出现在 depth / seg 图里**。录出来是一堆悬在虚空里的柱子。

实测证据（机位 -4,0,1.5、相机水平、垂直视场约 56°，画面下方本该是地面）：
- 修正前：下半部分几乎全是 `+1.0`（没打中），反色看只有圆柱悬在黑底上
- 挂上地板 mesh 后：出现从近（暗）到地平线（白）的平滑梯度，圆柱立在上面

**解决办法：给环境挂一块地板 mesh**，用现成的
`env_object_config.bottom_wall`（`bottom_wall.urdf`，20×20×0.2 的 box，
顶面在 z=+0.1）。`env_with_obstacles` / `forest_env` 这些环境之所以没这问题，
就是因为它们资产表里有 `bottom_wall: True`。

```python
from aerial_gym.config.asset_config.env_object_config import bottom_wall
# ...
include_asset_type = {"cylinders": True, "bottom_wall": True}
asset_type_to_dict_map = {"cylinders": cylinder_asset_params, "bottom_wall": bottom_wall}
```

> `empty_with_cylinders` 已经加好了（2026-09-24）。`create_ground_plane`
> 保留为 `True` —— 解析地面管物理（无人机不会掉下去），地板 mesh 管传感器，
> 两者不冲突。

## 用代码搭建环境（加物体、改场地）

**结论：可以，而且整套环境就是纯 Python 配置，没有任何 GUI / 编辑器依赖。**

### 机制：两层配置 + 一堆 URDF

aerial_gym 的环境由三样东西组成：

```
① 环境配置   aerial_gym/config/env_config/*.py      —— 场地多大、放哪几类东西
② 资产配置   aerial_gym/config/asset_config/*.py    —— 每类东西几个、多大范围、怎么摆
③ 资产本体   resources/models/environment_assets/<类名>/*.urdf   —— 实际的几何形状
```

`AssetLoader.randomly_pick_assets_from_folder()` 只认 `.urdf` 文件，
所以**任何新物体都得先有一个 URDF**。然后：

- 环境配置里 `include_asset_type = {"类名": True}`
- `asset_type_to_dict_map = {"类名": <资产配置类>}`
- 在 `aerial_gym/env_manager/__init__.py` 里注册一个名字，才能在 `--env` 里用

### 位置是怎么决定的

资产配置里的 `min_state_ratio` / `max_state_ratio` 是 13 个元素：

```
[ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]
```

⚠️ **名字有歧义**：前 3 个是**环境边界的比例**（0 = 下界，1 = 上界），
但 roll/pitch/yaw 那 3 个是**绝对弧度**，不是比例。踩过一次坑。

每次 `reset()` 都会用这两个范围重新随机采样位置 → 所以每个环境的布局都不同，
而且反复 reset 会一直换。

### 已经做好的例子：空环境 + 随机圆柱

| 文件 | 作用 |
|---|---|
| `code/tracking_bench/assets/make_cylinder_assets.py` | 生成圆柱 URDF（可调数量/半径/高度/颜色） |
| `.../asset_config/cylinder_asset_config.py` | 每个环境 15 个圆柱，立在地面上 |
| `.../env_config/empty_with_cylinders.py` | 10×10×5m 空地 + 地面 + 圆柱 |
| `record_sim.py --env empty_with_cylinders` | 跑起来 |

```bash
# 重新生成一批圆柱（想改尺寸就改参数重跑）
python tracking_bench/assets/make_cylinder_assets.py --count 12 --radius-max 0.35 --length-max 2.5

# 看效果
python record_sim.py --env empty_with_cylinders --view all \
    --use-warp False --device cuda:5 --pan --pan-pos -4 0 1.5 --out cyl.mp4
```

样例视频：`cylinders_demo.mp4`（RGB | 深度 | 分割 三路并排）。

### 圆柱为什么把原点放在底面

URDF 里圆柱默认原点在几何中心，那样不同高度的圆柱要配不同的 z 才能立在地面上。
生成脚本把 visual/collision/inertial 的 origin 整体上移 `length/2`，
**link 原点就落在底面** —— 于是所有圆柱 `z = 0` 就都正好站在地面上，
一个值适配所有尺寸。

### 几个要注意的地方

- **`fix_base_link = True`**（当前设置）圆柱是静态的，不会被撞倒。
  想让它们倒下就把这项改成 `False`，同时要保证环境开了 `create_ground_plane`。
- **`replace_cylinder_with_capsule`** 默认是 `True`，会把碰撞体偷偷换成两端圆头的胶囊，
  和视觉上的圆柱对不上。已经在 cylinder 配置里关掉了。
- **`semantic_id = -1`** 表示每个实例分配独立的分割 id（从 100 递增）。
  跟踪实验要区分"是哪一个圆柱"，所以保留 -1。设成固定正数则所有圆柱共用一个 id。
- **四旋翼出生点在 x 比例 0.1~0.2 处**，圆柱的 x 从 0.3 起，避开它，
  免得一开机就卡在圆柱里。
- **aerial_gym 会在 reset 时随机减半障碍物数量**（`env_manager.py:285-295`，
  对约 15% 的环境生效）。所以有时你会看到圆柱变少了 —— 这是框架自带的
  curriculum 机制，不是 bug。

### 分割图的显示坑

分割图的值是**实例 id**（100, 101, 102…连续递增），不是连续量。
一开始我用 min-max 归一化再套 `plasma` 色图，结果相邻 id 被压到一起，
整张图糊成一种黄色 —— 恰好丢掉了"区分每个实例"这个最要紧的信息。
现在改成按黄金比例散列色相，相邻 id 颜色差别很大，见 `record_sim.py` 的 `to_uint8`。

### ⚠️ `record_sim.py --env tracking_env` 录出来圆柱是**随机撒的** —— 不是 arena_20x20.yaml 的场地

**现象**：录 tracking_env 场景视频，6 根圆柱每次都撒在不同位置（可能撒到场地中央、
甚至相机机位上），和 `configs/arena_20x20.yaml` 的固定布局（半径 8.5 外圈）对不上。
**根因**：6 根圆柱共用一个资产类，配置做不到 6 个固定位置（见 `tracking_lib.
pin_static_layout` 的说明），必须在**每次 reset 之后**覆写状态张量。`run_tracking.py`
一直有这一步，`record_sim.py` 没有 —— 布局静默错，不报任何错。
**修法**：`record_sim.py` 在 `reset()` / `reset_terminated_and_truncated_envs()`
之后自动调 `pin_layout_if_needed()`（仅 `--env tracking_env` 时生效，别的环境槽位
含义不同，不能乱写）。
**怎么验证**：`python record_sim.py --env tracking_env --view all --use-warp False
--pan --pan-pos 0 0 2 --steps 300 --device cuda:4 --out /outputs/videos/xxx.mp4`
抽 4 帧核对：墙环带缺口、目标钉在 (2,0,2)、圆柱在外圈（2026-10-13 实测通过）。

## 跟踪闭环（阶段 1）踩过的坑

阶段 1（目标绕圈飞 + 追踪机跟随 + 录深度视频）已经跑通，产物在
`code/depth.mp4`、`code/traj_topdown.png`。下面每一条都是**不报错、但结果是错的**，
按杀伤力排序。改渲染/驱动环境物体之前先看这一节。

### ⚠️⚠️ warp 网格只在 reset 时重建 —— 移动物体在深度图里是**静止**的

**最狠的一个，沉默、不警告，症状还特别像别的问题。**

warp 相机的射线打的是 `wp.Mesh` 里**烘焙好的顶点**（`warp_camera_kernels` 直接
`mesh_query_ray`），它不跟着 PhysX 走。更新网格有两步：

```python
# aerial_gym/env_manager/warp_env_manager.py
vertex_maps_per_env_updated[:] = tf_apply(位姿, 原始顶点)   # ① 挪顶点
warp_mesh_per_env[i].refit()                               # ② 重建 BVH
```

**原代码只有 `reset_idx()` 调这两步，`pre_physics_step` / `post_physics_step` /
`step` 全是 `pass`。** 于是：物体在物理里正常飞，状态张量里的位置也对，
但**渲染出来的那个物体冻在上次 reset 的出生点**。

排查时踩过的五个假方向（都因为"状态张量是对的"而排除不掉）：

| 误判 | 当时的依据 | 实际 |
|---|---|---|
| 机体俯仰在晃 | 目标忽上忽下 | roll/pitch 是常数 4.4° |
| 目标飞出画面边缘 | 丢帧时画面底部有暗斑 | 那暗斑是圆柱，不是目标 |
| 位置驱动 vs 速度驱动 | 以为 warp 不跟速度积分 | 两种驱动都一样丢 |
| 多环境资产重合 | 4 个 env 共用一套世界坐标会打架 | `--num-envs 1` 一样丢 |
| 场地 180° 不对称 | 两个半圈表现不同 | 静止场景本来就不对称 |

**怎么一眼认出来**：拿目标的出生点反算像素，和实际分割图对照。当时渲染里的目标
始终停在配置的 reset 位置 `(2, 0, 2)`，11 个采样点全部吻合；而状态张量早跑到
别处了。**张量和画面对不上，就是这个 bug。**

**修法**：把 `reset_idx` 里的重算抽成 `update_meshes()`，在 `post_physics_step()` 里
也调一次（`env_manager.py:355` 每个物理步都会走到这里）。修完可见率 **54% → 100%**
（600 步里只有第 0 步出生瞬间没拍到）。

> `env_manager.py:338` 有一句注释 `# If you change the mesh, refit() needs to be
> called (expensive).` —— 作者自己知道这件事，只是钩子没接上。
> 如果嫌每个物理步 refit 太慢（`step()` 里 `simulate()` 跑 10 次，只有最后一次
> 会被渲染用到），可以把 `update_meshes()` 挪到 `render_sensors()` 里、
> `capture_sensors()` 之前，正好每次渲染一次。目前没测出性能问题，先不动。

### ⚠️ `env_actions` 要生效，必须开 `write_to_sim_at_every_timestep`

`IGE_env_manager.py:442` 用这个开关控制每步是否把状态写回 PhysX。
**默认是 `False`**（`empty_env.py:24`、`empty_with_cylinders.py:56` 等都是），
开着 `env_actions` 也不会有任何效果，而且不报错。
`tracking_env.py:122` 已经设成 `True`，动态环境的官方配置
`dynamic_environment.py:27` 也是 `True`。

### ⚠️ 所有实例共用一对 `min/max_state_ratio` —— 配置做不到"6 个固定不同位置"

`asset_loader.py:87` 把这对范围存在**资产类**上，不是每个实例一份。
所以 `min == max` 会让 6 根圆柱叠在同一个点；`min != max` 又会让它们每次 reset 乱跳。

只能在 reset 之后**覆写状态张量**（`tracking_lib.py:pin_cylinder_layout`），
而且**每次 reset 之后都要再调**：`asset_manager.reset_idx()` 会重新随机采样，
覆写会被冲掉。覆写完必须 `IGE_env.write_to_sim()`，否则物理里还是旧位置。

### ⚠️ 槽位顺序由 `asset_type_to_dict_map` 决定，`keep_in_env` 的排最前

`asset_loader.py:174` 用 `appendleft`，**后写的插更前**。当前顺序
`cylinders → bottom_wall → target` 得到：

```
槽 0 = target   槽 1 = bottom_wall   槽 2..7 = cylinders
```

跑错的后果是"驱动了圆柱/地板"，同样不报错。启动时打印
`num_obstacles_in_env` 和 `state.shape` 核对一下（`run_tracking.py` 开头会打）。

分割 id 跟着槽位走：`segmentation_ctr` 从 100 起，**target = 100**，
`bottom_wall` 是固定 **13**（不是 101），圆柱 102..107，`-2` = 射线没打中。

### ⚠️ 相机是 OpenCV 约定，v **向下**增大

cam-x = 机体右、cam-y ≈ 世界 −z（朝下）、cam-z = 光轴 ≈ 机体 +x。
深度值 = `cam_z / max_range`（10m）。成像 480×270，HFOV 87°（半角 43.5°），
VFOV 半角 28.1°。

```
u = 240 + degrees(atan2(cx, cz)) / 43.5 * 240
v = 135 + degrees(atan2(cy, cz)) / 28.1 * 135     # v 向下
```

想成"y 朝上"就会把上下判反。标定方法见 `code/diag_cam.py`：读
`warp_sensor.sensor_position/sensor_orientation` 的真实四元数，别用 euler 猜。
实测目标的分割质心比理论投影点系统性偏 **+16px (u)、+2px (v)**，对得上。

### ⚠️ `randomize_placement = False` 时相机在**机体原点**，不在 `nominal_position`

`warp_sensor.py:169-171`：不随机化就直接 "Do nothing"，
`sensor_local_position` 保持 0 —— `nominal_position = [0.10, 0, 0.03]` 是给
IsaacGym 相机用的，warp 路径根本不看。两个渲染路径的外参**不一致**。

### 跟随几何：追踪机停在目标的**径向外侧**

不能用 `normalize(追踪机 - 目标)` 定义"外侧"。第一版这么写，结果追踪机一跑到
原点另一侧就锁死在内圈绕（轨迹半径 0.5~0.9m），相机角度极别扭，可见率掉到 44%。
要取**从场地原点指向目标的径向**（`tracking_lib.py:chaser_follow`）。

### 速度限幅不能照搬"别超过目标的速度"

目标在半径 2 上 30°/s，线速度 1.05 m/s；追踪机在半径 4 上要跟同一角速度，
线速度是 **2.09 m/s** —— 是目标的两倍。限成 2.0 会让它永远追不上（角向滞后 43°）。
现在 3.0 m/s，留 43% 余量给暂态。

### ⚠️ 前馈超前量：偏航那一路是**减号**，和位置那一路相反

> **⚠️ 这一节只对阶段 0/1 的旧几何（圆轨迹 + 追踪机在径向外侧）成立。**
> 阶段 2 把跟随方位换成"沿目标速度反方向退"之后这个减号**变得有害**，
> 定稿形式是预测视线。重推过程 + 三轮实测 → 阶段 2 的
> 「跟随方位换了以后，两个前馈超前量全部重推」。

Lee 位置控制器是 PD 无积分项，跟匀速斜坡设定点留常值误差 `e = (Kd/Kp)·v`，
用"设定点往前推 τ"能抵掉。但**偏航的瞄准点要往目标身后放**（`target - v·τ`）：

> 追踪机在目标的**径向外侧**时，"机头→目标"的视线矢量大致指向原点，
> 而目标速度是切向的。把瞄准点沿 `+v` 推，视线矢量是**顺时针**转的，越推越落后。

实测（周期 12s = 30°/s）：`lead = 0.45s → 误差 19.5°`，`1.10s → 37.9°`，
斜率 +28.3°/s ≈ 角速度。换成减号后 **19.5° → 6.1°**。
位置那一路符号是正的（把设定点推到轨道更前方，减小角向落后 25° → 4.5°）。
周期改了要重测，误差 ∝ 角速度。见 `tracking_lib.py` 的 `POS_LEAD_TIME` / `YAW_LEAD_TIME`。

### 两机不能等高

等高时相机平视，目标贴在画面上沿（地平线上方约 20°，而垂直半视场才 28°），
轨道稍有起伏就从**上边缘**掉出去，可见率只有 64%。追踪机比目标高 0.8m 就稳了
（其中约 0.5m 被位置环的稳态误差吃掉，净高差 0.3m，正合适）。

---

## 阶段 2（测试环境）踩过的坑

### ⚠️ `env_actions` 的宽度 = `env_asset_state_tensor.shape[1]`，**不要再减 1**

`obstacle_manager.pre_physics_step()` 是把整块 `env_actions` 写进
`obstacle_linvel`，而后者就是 `env_asset_state_tensor[:, :, 7:10]` ——
**已经不含机器人**（`IGE_env_manager.py` 里 `env_asset_state_tensor = vec_root_tensor[:, 1:, :]`）。
所以宽度直接取 `state.shape[1]`。

减了 1 的症状是硬报错，好认：

```
obstacle_manager.py:43  self.obstacle_linvel[:] = actions[:, :, 0:3]
RuntimeError: The expanded size of the tensor (16) must match the existing size (15)
  at non-singleton dimension 1.  Target sizes: [1, 16, 3].  Tensor sizes: [15, 3]
```

容易踩是因为 `run_tracking.py` 里写的是 `IGE_env.num_assets_per_env - 1`，
那个 `-1` 减的是**机器人**（`num_assets_per_env` 数的是 `asset_handles`，含机器人）。
两个 `"-1"` 减的不是同一个东西，抄过来就多减了一次。
（`check_p1.py:187` 踩过，2026-10-12。）

### ⚠️⚠️ 建完环境**必须先 `env_manager.reset()`**，否则 warp 相机位姿是 NaN

现象：分割图整帧 `-2`（没打中），深度图全白，`warp_sensor.sensor_position` 是
**NaN**，**不报错、不告警**，一切看起来都"跑起来了"。

根因：`warp_sensor.update()` 里

```python
self.sensor_position[:] = tf_apply(self.robot_orientation, self.robot_position, ...)
```

`robot_orientation` 来自 IsaacGym 的 root state 张量。`build_env()` 返回后张量还没
被初始化，`tf_apply` 吃到零四元数就吐 NaN，后面 ray cast 的原点是 NaN，
全部 miss。

修法：`build_env()` 之后**立刻** `env_manager.reset()`，再 pin 布局，
然后才进主循环 —— 和 `run_tracking.py` 一样：

```python
env_manager.reset()
tl.pin_static_layout(env_manager)
```

（`check_p1.py:2` 踩过，2026-10-12。症状看着像"warp 看不见墙"，其实墙根本没被渲染。）

### ⚠️ `import isaacgym` 之后默认文本编码变 ASCII —— **打开文件必须显式 `encoding`**

现象：`yaml.safe_load` 读 `configs/arena_20x20.yaml` 直接抛

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 4
```

而同一份 YAML 在 `import isaacgym` **之前**读得好好的。

根因：isaacgym 初始化时会动 locale，`locale.getpreferredencoding()` 变成
`ASCII`。Python 的 `open(path)` 不写 `encoding` 就跟这个默认值走，于是
YAML 里第 4 字节的 `×`（`20×20`，U+00D7 = `0xc3 0x97`）和后面的中文注释全炸。
文件本身是合法 UTF-8，`file` 也报 `UTF-8 text`，**锅不在 YAML**。

修法：**所有**读写文本文件的地方都钉死 `encoding="utf-8"`，不管它现在是不是
只碰 ASCII —— 少一个就留一个雷。这次一起钉掉的：

| 文件 | 行 |
|---|---|
| `waypoint_policy.py` | `open(arena_yaml, encoding="utf-8")` |
| `tracking_lib.py` | `open(_CFG_PATH, encoding="utf-8")` |
| `aerial_gym/.../wall_asset_config.py` | `open(CFG_PATH, encoding="utf-8")` |
| `tracking_bench/assets/make_wall_assets.py` / `make_cylinder_assets.py` / `make_target_asset.py` | 读配置 + 写 URDF |

（2026-10-12 踩过。触发点是 `check_p2.py --with-sim`：数值部分不 import isaacgym
所以一直没事，一进仿真段就炸。）

### ⚠️⚠️ 轨迹一旦定稿就**不能再改** —— 否则播放中目标会瞬移

现象：`check_p2.py --with-sim` 里，目标位置**精确**写在 `state_at(t)` 给的值上
（读回偏差 6e-7 m），却会突然跳 **0.50 m**（4.95 m/s，是 `v_max` 的 4 倍），
还会跳进墙里。事后拿 `state_at(t)` 重算，和当时飞的那条对不上。

根因：`WaypointPolicy` 每加一个路点就 `_rebuild()` **重算全部段**。Catmull-Rom
切向 `vi = v_cruise·normalize(P_{i+1} − P_{i-1})` 看前后两个邻点，所以新路点
会改上一段的**末端切向**，连段时长 `_cum_t` 一起挪 —— 于是同一个 `t` 在
"续段之前"和"续段之后"落到不同的位置上。播放时每一步都可能踩在轨迹的
某个旧版本上，看着就是目标抽风。

这不是"调参没调好"，是**生成器的正确性**问题：播放过的时刻必须永远可复现。

修法：段改成 **append-only** —— 定稿了就不改。

- 段的**起始**切向 = 上一段的末端切向 ⇒ `v` 天然连续（这条验收要）
- 段的**末端**切向 = Catmull-Rom，而它要等 `pts[i+2]` 出现才算得出
- 所以段表比路点表**滞后一段**：`len(segments) == len(pts) − 2`
- 只有 `set_waypoints`（用户把路点给全了）才用后向差分收尾（`_flush_tail`）

验证：`check_p2.py` 的 **1e** —— 轨迹续到 200 s 之后，回头重算前 12 s 的
120 个采样点，位移必须是 `0.00e+00`。回归时这条红了就是又改回去了。
（2026-10-12 踩过，`check_p2.py` 的 5a'/5c/5d 三项一度全红。）

### ⚠️ 验收脚本里别把追踪机指到 `(0,0,0)` —— 撞地会触发 reset，资产被重采样

现象：`check_p2.py --with-sim` 的 `actions` 初始化成全 0，等于命令 Lee 控制器
飞到 `(0,0,0)`。地面在 `z=0`，追踪机撞地 → `collision_tensor` 置位 →
`reset_terminated_and_truncated_envs()` → `asset_manager.reset_idx()` 把**所有**
资产位置重新采样。目标的 `min_state_ratio == max_state_ratio`，所以是**瞬移到
固定点**，不是随机漂。600 步里能触发好几次，统计全毁。

根因：`reset_terminated_and_truncated_envs()` 只要有 collision 就 reset 整个 env，
`reset_idx` 会冲掉之前写进状态张量的一切。`check_p1.py` 是对的（`actions[:, 0:3]
= cpos0` 钉住不动），`check_p2` 第一版抄漏了。

修法：只验目标运动模型时，追踪机**钉在初始位置不动**；循环里统计 reset 次数并
重新 pin 布局；把"全程无 reset"做成一条独立验收项（`check_p2.py` 的 5e）。
（2026-10-12 踩过，现象是 5a 偏差 0.39 m、5c 最大步长 0.50 m，看着像目标抽风。）

### ⚠️ 两种净距别混用：路点 0.5 m / 曲线 0.05 m

五次曲线在弯道上会往弦外鼓（有上界，约 0.2·|Δp|，见 `waypoint_policy.py` 模块头）。
所以判据分两层：

| 查什么 | 净距 | 函数 |
|---|---|---|
| 路点 + 折线 | `wall_clearance` = **0.5 m** | `point_clear(..., wall_clearance)` |
| 飞出来的曲线 | `curve_clearance` = **0.05 m**（不进墙即可） | `_curve_point_ok` |

拿 0.5 去卡曲线会**误报**穿墙 —— 鼓肚本来就允许顶到 0.05 m 附近。
`check_p2.py` 的 2a/2b/5d 必须用同一个 `_curve_point_ok`；
第一版 5d 拿 `wall_clearance` 去卡，544/600 误报"出墙"，其实最近离墙还有 0.78 m。
（另外曲线的**场地边界**用 `arena.bounds_*`（±10）不是路点的 ±9，鼓肚鼓到 ±9
之外仍算在场内。）

### ⚠️ 验收脚本的判据要和**机器人初始位姿无关**，否则会假失败

**现象**：`check_p1.py` 的 3a「至少 3 面墙在 warp 分割图里找得到」在 P1 首次验收时
通过，后来重跑**变成 2/8 失败**，而这一轮我只动过 `tracking_lib.py` 的编码和
`chaser_follow_behind`、`wall_asset_config.py` 的编码、`waypoint_policy.py` 的
append-only 重构 —— **没有一个碰渲染**。

**根因**：3a 的判据是"把每面墙的几何中心投影到像素，在 50×50 窗口里取众数 id，
看是不是非目标/非地板"。但**机器人的初始位姿是 `env_manager.reset()` 随机的**，
相机朝向每次都不同：

- 8 面墙里有 5 面直接投影到画外（u/v 是负的或超 480×270）；
- 有 2 面虽在画内但超出 10 m 量程 —— 深度图打成 `+1.0`，而 `+1.0` **分不开**
  "超程"和"没打中"，分割图里就是 `-2`；
- 剩下的窗口还可能被地板/更近的面盖住。

于是窗口法只认出 1 个 id 就判失败。**可是同一帧的全局直方图里明明有 3 个墙实例 id**
（`{103: 723, 104: 1637, 107: 13530}`，共 15890 px）—— 墙一直在被渲染，
是**判据量错了东西**：它量的是"这次相机朝向"，不是"warp 看不看得见墙"。

**修法**：把 3a 的验收判据换成**整帧里有几个墙实例 id**（非 `-2` / 非目标 100 /
非地板 13，且像素数 > 30）。这跟位姿无关，就是要证的那件事本身。
原来那张逐墙窗口表**保留成纯诊断输出**，并且给每一行打上「画外 / 超程 / 太近 /
窗口被别的面盖住 / 命中 id=N」的标签，外加一句"这张表跟 reset 位姿走，不作验收"。

**怎么验证的**：修完重跑 `check_p1.py --device cuda:2 --steps 120`，12/12 全绿。
诊断表显示这次相机朝向 8 面墙里 5 面画外、2 面超程、1 面被地板盖住，
**窗口法 0 命中**，而整帧里有 4 个墙实例 id（15072 px）—— 旧判据在这次位姿下
必然假失败，新判据正确通过。

> 通用教训：**别在验收脚本里断言一个依赖随机初始位姿的量**。要么先把位姿钉死再断言，
> 要么把断言改成位姿无关的不变量（这里是"渲染看得见墙"）。

---

### 跟随方位换了以后，两个前馈超前量全部重推（2026-10-12）

**为什么要重推**：`POS_LEAD_TIME = 0.83` / `YAW_LEAD_TIME` 的减号都是在
**旧几何**（圆轨迹 + 追踪机在目标径向外侧）上标的。BENCH_PLAN §3 P4 把跟随方位
换成"沿目标速度反方向退 `standoff`"，旧几何的前提没了。`check_follow.py`
在**同一个仿真里**换超前量跑对照，400 步取后半段稳态，seed 20261012。

**位置前馈：几何无关，但数值变了。** 理论是 `e = (Kd/Kp)·v`，xy 面
`K_pos/K_vel = 2.0/2.0 = 1.0 s`（`lee_controller_config`，`randomize_params=False`），
所以 τ 的理论值是 **1.0**。实测 `沿 v̂ = (追踪机 − 目标)·v̂` 的均值（理想 = −2.00）：

| pos_lead | 0.00 | 0.45 | 0.70 | 0.83 | **0.87** | 0.95 | 1.10 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| 沿 v̂ | −2.875 | −2.422 | −2.169 | −2.037 | **−1.997** | −1.917 | −1.767 | −1.667 |

对 `pos_lead` 是干净的直线，斜率 **1.008 ≈ 轨迹平均 \|v\|**，零点拟合 **0.868**。
拿 0.87 复测，实测 `沿 v̂ = −1.997`，离理想只有 **3 mm** —— 拟合和实测对上了。
**定稿 `POS_LEAD_TIME = 0.87`**（不是 0.83，差 4 cm；也不是理论值 1.0，
五次轨迹不是纯斜坡、拐弯时 v̂ 在转、`limit_speed` 还额外吃掉一点）。

> ⚠️ 判据是 `沿 v̂`，**不是 `|‖d‖ − standoff|`**。距离对了但方位歪了，
> 后者一样小，而方位歪是偏航环的事。

**偏航前馈：旧几何的减号在新几何下有害，正确形式是预测视线。**

| 方案 | 偏航误差 |
|---|---|
| 零超前（基线） | 4.80° |
| 旧减号 `target − v·0.5`（挪瞄准点到身后） | **8.53°** ← 比不加还差 |
| 挪瞄准点 `target + v·0.5` | 2.44° |
| **预测视线 τ = 0.30** | **0.22°** |

**这里要更正我先前的一个说法。** 我一开始推导"新几何里跟踪到位时视线正好顺着
`v̂`，沿 ±v 挪瞄准点等于沿视线方向推，一阶上转不动视线"，据此断言挪瞄准点没用。
**这是错的。** 沿 ±v 挪瞄准点的转动效果**不为零**，只是二阶小量，它透过跟踪误差
`e` 起作用：

```
δθ = −|v| · yaw_lead · (v̂ × e) / |LOS|²
```

`e` 是追踪机的位置误差，永远不为零（尤其拐弯时），所以 +0.50 能把 4.80° 压到
2.44°、−0.50 却涨到 8.53° —— 方向性跟旧几何相反，但**不是转不动**。
只是这条路不干净（效果依赖 `e` 的大小和方向），不如直接用**预测视线**：

```
yaw_cmd = atan2( (target + v·τ) − (chaser + v_c·τ) )
```

视界 τ 的扫描是 U 形，谷底在 **0.30 s**：

| τ | 0.10 | 0.15 | 0.25 | **0.30** | 0.35 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|
| 偏航° | 3.18 | 2.43 | 0.92 | **0.22** | 0.60 | 1.33 | 2.79 |

**定稿 `YAW_LEAD_TIME = 0.30`，形式用预测视线（必须传 `chaser_vel`）。**
0.30 s 的 0.22° 比零超前好 22 倍。τ 太小补不够转弯、太大超前过头，两侧都涨。

**怎么验证的**：三轮 `check_follow.py`（`--preset base` / `refine` / `confirm`，
各 400 步，只统计后半段稳态，全程 0 次 env reset）。`refine` 与 `confirm` 在
τ = 0.25 / 0.35 两点完全复现（0.93/0.92、0.59/0.60），说明噪声可忽略。
`confirm` 把 pos_lead = 0.83 与 0.87 并排跑，0.87 的 `沿 v̂` 从 −2.037 收到 −1.997。

> ⚠️ 这两个数只对 **dt = 0.1 s、v_cruise = 1.2 m/s、Lee 增益 (2,2)/(2,2)**
> 这套配置有效。换控制周期 / 换增益 / 换轨迹速度都要重测。
>
> ⚠️ `check_follow.py` 的"自动挑最优"只在 `kind == "pos"` 的行里比。
> `--preset refine` 把 0.83 那几行标成了 `kind="yaw"`，于是漏掉 0.83、
> 误报 0.95 最优。跨 preset 比数字要自己把表拼起来看。

---

### ⚠️⚠️ 投影公式是**针孔**，不是等距柱状 —— "掩码质心偏 +16px" 是误记

**现象**：`check_p3.py` 的 1d 要"确认掩码质心相对几何投影点的 +16px u / +2px v
偏置仍在"。实测三个静止场景：

| 场景 | 旧公式 u | 掩码质心 u | "偏置" |
|---|---|---|---|
| 正前方 3.0 m | 240.0 | 240.0 | **0.0** |
| 正前方 5.0 m | 240.0 | 240.0 | **0.0** |
| 偏左 1.2 m @4.0 m | 147.9 | 163.4 | **+15.5** |

偏置在光轴上是 0、离轴 15° 才涨到 15.5 —— **常数偏置不会长这样**。

**根因**：投影公式用错了模型。计划里那条

```
u = 240 + degrees(atan2(cx, cz)) / 43.5 * 240
```

是**等距柱状**（像素 ∝ 角度）。而 warp 渲染器是**针孔**（像素 ∝ tan(角度)），
`warp_cam.py:36-40` 写得很明白：

```python
f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)   # = 240/tan(43.5°) = 252.908
vertical_fov = 2 * math.atan(H / (2 * f))           # = 56.19°，半角 28.10°
alpha_u = u_0 / math.tan(self.horizontal_fov / 2)   # = f
alpha_v = v_0 / math.tan(vertical_fov / 2)          # = f   → 方像素
```

两套公式只在**图心**（u = 240）和**图边**重合，中间最大差 16 px：

```
角度 15°  ->  等距柱状  240 - 240·15/43.5 = 157.2
            针孔       240 - f·tan(15°)  = 172.3      差 15.1 px
```

正好就是那个"+16px"。所以**根本不存在掩码质心偏置**，是参照系错了。
（`VFOV_HALF = 28.1` 本来就是从针孔的 `atan(H/(2f))` 推出来的，等于自己印证了 f 是对的。）

**修法**：`truth.py` 的 `project()` 改成针孔，内参暴露成 `K = (fx, fy, cx, cy)`：

```
cx, cy, cz = R_cam.T @ (target_pos - cam_pos)
u = 240 + f * cx / cz
v = 135 + f * cy / cz
f = 240 / tan(radians(43.5)) = 252.908
```

旧公式留成 `truth.project_equirect()`，**仅留档**，好让上面这段记录能被复算。
`check_p1.py` 的 `project()` 也改成委托 `truth.project`，别再各写一份。
`cz <= 0`（目标在相机背后）时给符号正确的 1e9，保证 `in_fov=False` 又不产生 nan。

**怎么验证的**：`check_p3.py` 1d，三个静止场景，针孔投影 vs 掩码质心：

| 场景 | 针孔 u | 等距柱状 u | 质心 u | 针孔残差 |
|---|---|---|---|---|
| 正前方 3.0 m | 240.0 | 240.0 | 240.0 | 0.00 |
| 正前方 5.0 m | 240.0 | 240.0 | 240.0 | −0.00 |
| 偏左 1.2 m @4.0 m | **164.1** | 147.9 | **163.4** | **−0.75** |

残差均值 **(−0.25, +0.20) px**，最差 **0.75 px**，std (0.35, 0.36) px —— 对到亚像素。
12/12 全绿。

> **这件事为什么重要**：`projected_pixel` 的用途是"定位是跟踪器的问题还是成像的问题"。
> 拿一个已知错 16 px 的公式当参照，会把跟踪器的误差误判成成像误差，或反过来。
> 顺带 §2.2 的 `K`（fx, fy, cx, cy）字段现在是真的能填了 —— 只有针孔模型才有 K。
>
> **教训**：写"某某有 +N 的系统性偏置"这种结论之前，先问一句"这个偏置是不是
> 我的参照系带来的"。判据是常数还是随位置变，一测就知道。

---

### ⚠️ 目标贴相机时 `target_depth_mean` 是**混合均值**，不能当距离用

**现象**：`check_p3.py` 场景 4（目标表面离相机 0.186 m < MIN_RANGE 0.2 m）
算出 `target_depth_mean = -0.543`。直接当归一化深度换算距离会得到 **−5.43 m**。

**根因**：深度图值域是 `[-1] ∪ [0,1]`，`-1.0` = 比 `MIN_RANGE` 还近。
目标离得极近时，**近端**那部分像素写 -1.0，**远端**还在量程内写正值。
实测掩码 106273 px 里 58777 个是 -1.0、其余是正值，均值就在中间。

**修法**：`truth.py` 保留"掩码内深度均值"的原义（`-1.0` **算进去**，否则
"整块都在近程里"这件事会被抹平），但**同时给出** `target_depth_neg1_ratio`
（掩码里 -1.0 的占比）。`summarize()` 在占比非 0 时会把它一起打出来：

```
depth=-0.543(近程占55%)
```

用的时候：`neg1_ratio` 高 → 整个目标都在 `MIN_RANGE` 里；低 → 只有近端超近程。

**顺带踩的一个轴向坑**：OpenCV 约定下光轴是 **cam-z**，不是 cam-x。
第一版把"沿光轴放 0.35 m"写成 `target_rel_to_cam = (0.35, 0, 0)`，
目标跑到相机**右边 90°** 直接出画（`proj` = (736, −299)，`in_fov=False`）。
正确的写法是 `(0, 0, d)`。

**怎么验证的**：`check_p3.py` 4a/4b/4c —— 表面距离 0.186 m < 0.2 m、
`target_depth_mean = -0.543`、掩码里 58777/106273 个 -1.0，
且 `in_fov` / `visible` 照常给、不崩。

---

## 阶段 3（接口层 + 自动统计）踩过的坑

这一段的背景：P4 把 `truth.py` / `waypoint_policy.py` / `runner.py` /
`bench_eval.py` 串成一个能跑 episode、能出指标的测试环境，`check_p4.py` 做验收。
**第一轮 `--quick` 就有 4 条红**，底下 5 个坑是逐条查出来的。

### ⚠️⚠️ `in_fov` 不能只看**中心投影** —— 目标是扩展物体

**现象**：`check_p4` 的 baseline 跑出 `visible_ratio = 0.925 > in_fov_ratio = 0.692`、
`occluded_steps = 0`。三者互相矛盾：掩码里有像素就必然投影在图内，
`visible ⊆ in_fov` 应该是不变量。同时 `pixel_err` 打出 `5e7` 这种数。

**根因**：`in_fov` 原来定义成 `in_image(中心投影)`. 目标是**直径 0.6 m 的圆柱**，
不是点 —— 中心刚滑出图下边缘时，上半截还占着半屏，于是
`visible=True ∧ in_fov=False` 这种帧**是真的会出现的**，不是投影算错。
`diag_truth.py` 逐帧对照 40 步，矛盾帧 0/40，投影和掩码质心对到亚像素；
第 41 步 `proj_v = 270.3`（H=270，中心刚出图）而掩码仍有 2055 px —— 正是这种情况。

危害不只是指标难看：默认判据 `lost_criterion = in_fov`，于是"目标明明还在画面里"
会被当成**丢失**，3 s 后直接判跟踪失败。

**修法**（`truth.py::compute_truth`）：

```python
in_fov   = in_image(中心投影) or visible          # 有一部分在视场里（不管挡没挡）
visible  = 掩码非空
occluded = in_fov and not visible                 # = in_image(中心) and not visible
丢失     = not in_fov                             # = 中心出图 **且** 一个像素都没有
```

三条不变量由此成立：`visible ⊆ in_fov`；`occluded ⇒ 中心在图内`；
丢失 = 真的什么都看不见（而不是"没对准中心"）。

**怎么验证的**：`diag_truth.py` 60 步矛盾帧 0/60，且 `proj_v=270.3` 那种
"中心出图、掩码还在"的帧被正确判成 `in_fov=True`；`check_p3.py` 的
1a/2a/2c/3a/3b/4c 六条断言在新语义下**全部仍成立**（`visible ⇒ in_fov`
只会让 1a/4c 更强），不用改；`check_p4` 1d 的 `in_fov_ratio` 从 0.692 → 0.933。

### ⚠️ `T_body_cam` 别去"实测" —— 传感器位姿在建环境后还没同步

**现象**：实测出的 `T_body_cam` 平移是 `(7.77, 3.25, -1.32)`、旋转块是一次
−17° 的偏航。但同一时刻 `cam_pos == body_pos`（相机就在机体原点），
平移应当是 0。差出来的那 8 m 就是当时机器人和传感器的位姿差。

**根因**：`warp_sensor.update()`（`warp_sensor.py:180-187`）才把
`sensor_position/sensor_orientation` 从机器人位姿算出来：

```
sensor_position    = tf_apply(body_q, body_p, sensor_local_position)
sensor_orientation = body_q ⊗ (sensor_local_orientation ⊗ data_frame_quat)
```

`env_manager.reset()` 之后、第一次 `update()` 之前，这两个张量还是初始化的值。
拿它们去和机器人位姿做差，量到的是废值 —— 而且**不报错**。

**修法**：安装变换在 `randomize_placement=False` 下是**常量**，直接读安装参数
构造，不去做减法（`runner.SimSession._mount_T_body_cam`）：

```
t_mount = sensor_local_position
R_mount = quat_to_mat(sensor_local_orientation ⊗ sensor_data_frame_quat)
T_body_cam[:3,:3] = R_mount.T                 # body -> cam
T_body_cam[:3, 3] = -R_mount.T @ t_mount
```

⚠️ 旋转块是 **body→cam**。相机三轴在**机体系**里的方向是 `R_mount` 的列
（= `T[:3,:3].T` 的列），别和旋转块本身搞混。接口文档里两条都写了。
⚠️ 要是哪天开了 `randomize_placement=True`，这个常量假设就破了，得改成每次 reset 重算。

**怎么验证的**：改完实测 `T_body_cam = [[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]`、
平移全 0，且 `cam_pos == body_pos`；手验三条映射：
机体 +x（前）→ cam +z（光轴）、机体 +y（左）→ cam −x（右的反方向）、
机体 +z（上）→ cam −y（下的反方向），与 OpenCV 相机约定一致。

### ⚠️ `pixel_err` 把 `±1e9` 哨兵平均进去了

**现象**：baseline 的 `pixel_err u = 5.0e7 ± 2.2e8`，do_nothing 的
`u = -1.00000024e9`。指标表里出现这种数，整组像素误差就废了。

**根因**：目标在相机**背后**时 `truth.project()` 返回 `±1e9` 的哨兵
（为了让 `in_fov=False` 且不产生 nan/inf）。`bench_eval.metrics()` 里
"没有掩码就退回 `projected_pixel`"这条兜底把哨兵一起平均了。

**修法**：只统计**真的在像平面前**的样本，`|u|,|v| < 1e6` 才要；
并在 `pixel_err` 里多给一个 `n_samples`，让读的人知道均值是几个帧算出来的。

**怎么验证的**：`check_p4` 2 组（do_nothing，30 步全在背后）的 `pixel_err`
现在是 `n_samples=0`、四个统计量 `null`，不再是 −1e9；1 组 baseline 的
`pixel_err` 是正常的几十像素量级。

### ⚠️⚠️ `DoNothingTracker` 只锁了位置，**没锁偏航**

**现象**：`scene_occluded` 换成 `do_nothing` 之后
`in_fov_ratio = 0.09`、`visible_ratio = 0.0`、第 33 步判 `lost_timeout` ——
目标根本不在画面里，遮挡几何整个作废。

**根因**：`DoNothingTracker.__init__(hold_yaw=0.0)`，而 `reset()` 是空的。
位置在第一次 `act()` 里从 `obs.chaser_pos` 锁（正确），**偏航却用写死的 0.0**。
`scene_lost` 的 `start_pose` 偏航恰好是 0，侥幸对上；`scene_occluded` 的是
π/2（朝 +y），于是追踪机站在 (0, 3, 2.5) 却**朝 +x 看** ——
目标在 69°~111° 方位上，远超 43.5° 半视场。

**修法**：位置和偏航**都**从 settle 结果里锁 —— 第一次 `act()` 一起定死，
`hold_yaw=None` 表示"跟 settle 走"，给浮点数才用给定的。

**怎么验证的**：`check_p4` 4a/4b 从红转绿 —— `occluded_steps = 62`、
`in_fov_ratio = 1.0`、`visible_ratio = 0.225`、80 步跑满不判失败；
4c 对照（`lost_criterion=visible`）第 30 步判 `lost_timeout`。

**顺带修的同源问题**：`scene_occluded.yaml` 原来写的是 `tracker.type: baseline`。
baseline 会朝目标飞，直接翻过墙跑到目标旁边（实测 `distance≈2.0`、
`occluded_steps=3`），遮挡几何同样作废。验收场景必须 `do_nothing` + 钉死
`start_pose`，几何才和 YAML 头注释里算的一致。

### ⚠️ `CHASER_Z_OFFSET` 的 0.8 是给**旧绕圈几何**标的，新几何下把目标顶出画面

**现象**：baseline 跑起来目标一路从 `proj_v=168` 滑到 270+（H=270），
贴着画面下缘，距离一近到 1.5 m 以内就掉出图。

**根因**：`CHASER_Z_OFFSET = 0.8` 的注释写的是"0.8 里约有 0.5 m 被位置控制器
的稳态误差吃掉，净效果只比目标高 0.3 m"。那个 0.5 m 下沉是**旧的绕圈几何**里
向心加速度造成的（命令 z=2.0 实际停 z≈1.46）。**新几何（分段五次）没有持续
向心加速度**，`diag_truth.py` 实测 `sp_z − target_z` 恒等于设定的 0.80、
追踪机跟设定点的偏差 ≤ 0.41 m（只在下降时滞后）。所以真实相对高度就是 0.8 m。

相机是**刚性前视、无云台**，俯角只由相对高度决定：2 m 站位下
`atan(0.8/2) = 21.8°`，把 28.1° 的垂直半视场吃掉 **78%**，目标被压在下缘。
等效地，相对高度只要大于 `range·tan(28.1°)` 目标就出图 —— range=1.5 m 时
这个上限是 0.80 m，正好卡在临界。

**修法**：`CHASER_Z_OFFSET` 0.8 → **0.30**。2 m 站位俯角 8.5°、`proj_v ≈ 173`；
相对高度 ±0.5 m 的扰动下 `proj_v ∈ [110, 236]`，稳在 `[0, 270]` 内。
目标 z 顶到 4.0 时 clamp 会把偏移压成 −0.3（俯角 −8.5°、`proj_v≈97`），照样在画内。

⚠️ **不影响水平通道**，`POS_LEAD_TIME` / `YAW_LEAD_TIME` 不用重标：
`chaser_follow_behind` 的 `desired[2] = target_z + offset` **不吃 pos_lead**，
`v_hat` 也只用 xy 速度。垂直和水平是解耦的。

### ⚠️ 验收脚本里"跑满多少步"要用**该组自己的** `max_steps`

**现象**：`check_p4 --quick` 的 3b 报 `跑满 120 步` 却红了，detail 显示
`steps_run=60 result=success`。

**根因**：3 组（`lost_timeout` 可配）跑的是 `scene_lost`，quick 模式下
`max_steps=60`；断言里写的却是主场景的 `n_main=120`。全量模式下两者都是 600
所以看不出来，`--quick` 一跑就露馅。

**修法**：抽一个 `n_lost = cfg_lost.episode.max_steps`，3b/3c 都拿它当基准。
教训：**同一份脚本里跑了多组不同 episode 长度的场景，满程步数别共用一个变量。**

---

### ⚠️⚠️ 自由空间检查只查墙、**不查圆柱** —— 路点能采到柱面上

**现象**：`check_p4` 全量（600 步）的 baseline 在**第 161 步**撞死
（`fail reason=collision@161`）。`--quick`（120 步）一直是绿的，所以一直没露出来。
跟得其实很好：`in_fov=0.944`、`dist_err mean=0.222`、`visible=0.944`。

**根因**：`waypoint_policy.point_clear()` 只吃 `arena["walls"]`，而 6 根圆柱的布局
当时硬编码在 `tracking_lib.CYLINDER_XY` 里，**从来没进过自由空间检查**。
路点直接采到 `(-8.47, -8.04)` —— 离柱心 `(-8.5, -8.5)` 只有 **0.46 m**，
柱半径 0.30 m，也就是**离柱面只剩 0.16 m**，远小于 `wall_clearance = 0.5`。
目标是运动学驱动的，贴着柱子飞过去一点事没有；追踪机隔 2 m 跟在后面，
设定点路径整段穿柱，物理体撞上就判失败。

**怎么定位的**：`diag_collide.py` —— 不开 IsaacGym、不占 GPU，纯离线重放
`chaser_follow_behind` + `limit_speed`，把每一步的设定点跟墙/圆柱做碰撞检查。
161 步里从第 147 步起连续 15 步落在圆柱 (−8.5,−8.5) 里，一眼看穿。

**修法**：圆柱布局搬进 `configs/arena_20x20.yaml` 的 `cylinders:` 一节
（那个文件头本来就写着"场地布局的唯一来源"，墙在 YAML、柱在 Python 里是自相矛盾），
`tracking_lib.CYLINDER_XY` / `CYLINDER_SIZE` / `N_CYLINDERS` 改成从 YAML 读；
`waypoint_policy` 把墙和圆柱合成一张障碍表 `self.obstacles` 一起查。

**怎么验证**：旧坏点 `(-8.47,-8.04,2.14)` 现在 `_point_ok()` 返回 `False`；
`check_p2` 数值部分 12/12 仍然绿（`2b` 报「600/600 步在自由空间」）。
同 seed 的路点会**变**（种子一样、可行域变了），这是预期的，
`3a` 的"同 seed 逐点一致"不受影响。

**顺带发现的第二层问题**（同一个根因的另一面）：**目标在自由空间里飞，
不代表 `target − standoff·v̂` 这个跟随点也在自由空间里。**
补完圆柱检查后离线扫 600 步，仍有 **3 步**期望点整个落进圆柱、
另有若干步贴着墙（0.15 m 边距下算命中）。所以 baseline 还得自己避障 ——
见下一条。

---

### ⚠️⚠️ 一局撞卡之后，**后面的 episode 全废** —— settle 飞不出来

**现象**：baseline 在第 161 步撞死之后，第 2~5 组每个 episode 都是
`fail (collision @ step 2)`、`dist_err≈12 m`（两机相距 14.3 m）。
但 settle 打出来的命令明明是 `pos=(0.00,0.00,2.50) yaw=0°`。

**根因**：撞进圆柱后无人机会**卡在几何体里**，Lee 位置控制器被接触力顶住，
光发位置设定点根本顶不出来。settle 跑 40 步，机身纹丝不动，
下一局从错误的位置开局、第 2 步就撞。共享的 `SimSession` 被一局撞废了。

**怎么定位的**：全量日志里 settle 报的命令位姿是对的、机身却没动 ——
"命令对、执行不动"这个组合直接指向卡死而不是控制律问题。

**修法**：`SimSession.teleport_robot(pos, yaw)` —— 直接写
`gtd["robot_state_tensor"]`（位置 + 偏航四元数 + 速度/角速度清零）再
`IGE_env.write_to_sim()`，把机身**搬**过去而不是飞过去。`settle()` 开头先传送。

> ⚠️ `robot_state_tensor` 是 `unfolded_vec_root_tensor[:, 0, :]` 的**视图**，
> 所以写完调 `IGE_env.write_to_sim()` 就生效（它写的是整张 root 张量，
> **含机器人**）。别动 `env_asset_state_tensor`，那是槽 1..N 的资产，不含机器人。

**为什么不能用 `env_manager.reset()` 来复位**：它会把所有资产位置重新随机采样，
得再 `pin_static_layout()` 才能摆回来，多一次机会摆错。传送机器人是最小改动。

**怎么验证**：全量 `check_p4` 里每组 settle 后 `dist_err` 回到 2 m 量级，
不再出现 `collision@2`。

---

### ⚠️ baseline 要当参考实现用，就得**自己避障**

**现象**：补完圆柱的自由空间检查后，离线扫 600 步仍有 3 步期望点落进圆柱、
若干步贴墙（margin 0.15）。目标穿障碍没事，追踪机穿就判失败。

**根因**：`chaser_follow_behind` 的期望点是 `target + v·pos_lead − standoff·v̂`。
这个偏移点的自由性跟目标本身无关 —— 目标贴着障碍飞，跟随点就在障碍里。

**修法**：`tracking_lib.avoid_setpoint(chaser, desired, obstacles, margin=0.30)`：
期望点本身自由且从追踪机直线可达就直接用；否则**绕追踪机在水平面转一个角**
（±15°…±180°，距离不变），取第一个自由且可达的；全挡死就原地不动。
`BaselineTracker` 在 `limit_speed` 之后调它，改写完**再限一次速**
（避障是跳过去的，一步最多跳 0.4 m，不二次限速会甩过头）。

两条设计约束，别改坏了：

- **偏航不受避障影响**，始终指向目标。位置躲障碍躲到跟丢就本末倒置了。
- 全挡死时**原地不动**（沿用上一步设定点），不要穿墙。停下来目标会飞走，
  但那会如实记成 `lost_timeout`，比静默撞墙诚实。

**怎么验证**：`python diag_collide.py --fresh --avoid --margin 0.05 --steps 600`
→ 0 次命中、19 步改写、1 步完全挡死。

---

## 阶段 4（P5 打包 + 交接）踩过的坑

### ⚠️⚠️ 打包搬家后，所有按 `__file__` 拼的配置路径**全部解析错**

**现象**：P5 把 flat 模块收进 `tracking_bench/` 包后，`import tracking_lib` 直接炸
`FileNotFoundError: '/workspace/tracking_bench/tracking_bench/configs/arena_20x20.yaml'`。

**根因**：`tracking_lib.py` / `bench_config.py` / `target/waypoint_policy.py` 都写
`os.path.join(os.path.dirname(abspath(__file__)), "configs", ...)` —— 模块在
`code/` 时深度是 1，搬进包里深度变 2，拼出来的路径就到了子包下面。
外层脚本同病：`tests/acceptance/check_p4.py` 按自己的目录拼 `configs/`，
`assets/make_*.py` 按 `__file__` 拼资产目录，全歪。

**修法**：收成**一处** `tracking_bench/paths.py`：
`find_config(名)` 往包根找 `configs/`（多试两层，找不到报清楚）；
`aerial_gym_asset_dir(子目录)` 用 `importlib.util.find_spec("aerial_gym")` 定位
安装根，再拼 `resources/models/environment_assets/...`（`AssetLoader` 只认这个位置）。
配置里的 `arena_yaml:` 也不再"相对当前目录"，`bench_config._resolve_arena_yaml`
按「这份 YAML 的目录 → 包根 → configs/ → cwd」的顺序找。
**教训：凡是按 `__file__` 拼路径的地方，搬家必断 —— 公共路径解析要只有一个家。**

**怎么验证**：从**任意目录**（实测 `/tmp`）跑 `tests/smoke_import.py` 全通；
三个 `make_*_assets.py --list` 打印出的路径都指回真的
`/workspace/aerial_gym_simulator/resources/models/environment_assets/...`。

### ⚠️ 拆 `sim/session.py` 时把 `Obstacle` 落下了 —— 运行期才炸

**现象**：`SimSession.__init__` 里 `NameError: name 'Obstacle' is not defined`，
import 冒烟测不出来（`smoke_import` 只 import 模块，不建会话）。

**根因**：P5 把 `SimSession` 从 `runner.py` 切到 `sim/session.py` 时，
`from tracking_bench.trackers.tracker_api import (...)` 的清单没同步，
`Obstacle` 还留在旧文件的 import 里。**跨文件搬类，import 清单要对着用到的名字逐个核。**

**修法**：import 清单补 `Obstacle`。
**怎么验证**：全新目录真跑一次 `scripts/run_bench.py --steps 60`（不是只 import）。

### ⚠️⚠️ 障碍表把圆柱尺寸写成目标的尺寸 —— 飞行高度**避障静默失效**

**现象**：`session.py` 障碍表里圆柱 `size=(0.60, 0.60, 0.35)` —— 那是**目标**
扁圆柱的尺寸，不是圆柱障碍的 `(0.6, 0.6, 3.0)`。避障的盒测试带 z 判定
（`-margin ≤ z ≤ sz+margin`），飞行高度 z≈2 时永远判不到柱子。

**根因**：手写常量没跟着 `tracking_lib.CYLINDER_SIZE` 走（YAML 是唯一来源，
代码里再抄一份必漂）。之前 check_p4 能过纯属侥幸 —— 路点侧的自由空间检查
（对的高度）把目标路径拦住了，跟随点没再撞上。

**修法**：`size=tuple(tl.CYLINDER_SIZE)`，并在注释里钉死"别手写"。
**怎么验证**：改完重跑**全量** `check_p4`（22/22 绿）；`diag_collide --avoid`
的避障逻辑和它同一套几何。

### ⚠️⚠️ 清理老 `configs/` 时把 aerial_gym 那边的读取弄断了 —— **建环境才炸**

**现象**：P5 清理把 `code/configs/` 删了（内容已并进 `tracking_bench/configs/`），
但 `aerial_gym/.../wall_asset_config.py` 在 **import 时**读
`AERIAL_GYM_DIRECTORY/../configs/arena_20x20.yaml` —— 那正是被删的路径。
离线回归全绿、`diag_collide` 照跑（它们不 import aerial_gym），
**下一次真建环境**才 `FileNotFoundError`。

**根因**：一个真相源被搬了家，**读者不止一个** —— 只改了包内的读者，
vendored aerial_gym 里的读者漏了。import 冒烟覆盖不到（它不建环境）。

**修法**：`wall_asset_config.py` 的 `CFG_PATH` 改成两级搜索
（`../tracking_bench/configs/` → `../configs/` 兜底），报错时列出找过的位置。
并加回归 `tests/regression/test_arena_single_source.py`：按同样顺序找 YAML、
断言**存在且和包内正源逐字节一致**（顺带钉死"单一真相源"）。
**怎么验证**：`tests/run_all.py` 5/5；真跑一次
`scripts/run_bench.py --steps 10` 能建起环境。

---

## 交接包（aerial-track/）—— 精简打包的组装记录

背景（2026-10-12 新要求）：尽量减少打包内容 —— 镜像只留 Dockerfile、
aerial_gym 核心只给获取方式 + 补丁、解压后顶层 README 当目录地图。
组装入口是工作区根目录的 `./make_aerial_track.sh`（同步生成物 + 打 `aerial-track.tar.gz`，
**改完代码要重跑一次再发**），包结构与装机三步见 `aerial-track/README.md`。

### 关键决策（以及为什么）

- **pip 依赖烧进镜像，不再交 `container_home/.local`**：开发机那套 bind-mount
  `.local` 的做法是怕 `docker rm` 丢环境（见「容器」节），但对新机器反而是
  坑（挂了个空目录就把镜像内的包影蔽了）。交接版 Dockerfile 把全部 pip
  依赖装进镜像，`start_container.sh` 只挂 `code/`，重建容器零成本。
- **aerial_gym / urdfpy 不打包，走 clone + 钉版本 + 一个 git 补丁**：
  本地改动只有 8 个文件（3 改 + 5 增），`git add -N <5 个新文件>` 后
  `git diff` 就能出**含新文件**的单一补丁（记得 `git reset` 收回 index）。
  URDF 生成物不进补丁（`make_*_assets.py` 可重建）。
- **urdfpy 必须用 GitHub master（5466842），不能 `pip install urdfpy`**：
  PyPI 版有 cylinder mesh 的 bug（早期踩过）；aerial_gym 在
  `warp_asset.py` 第一行就 `from urdfpy import URDF`，缺了直接起不来。
  交接版里它和 aerial_gym 一起用 `PYTHONPATH=/workspace/...` 导入，
  不做 pip install —— 换代码不用重装。
- **IsaacGym 压缩包不分发**（NVIDIA 许可），接收者自行申请；
  `docker/build_image.sh` 用交接包的 Dockerfile 覆盖官方那份再 build
  （官方的缺 gymutil 补丁和 pip 层）。
- 镜像的 uid/gid 用 build-arg 传宿主机的，避免挂载目录文件归属错乱；
  撞了已有 UID 会明确报错（交接包 README §7 有一行处置）。

### 怎么验证的（模拟接收者全流程，2026-10-12）

解压 `aerial-track.tar.gz` → `setup_workspace.sh`（走 ghfast 镜像，clone
aerial_gym@f0d0f05 + urdfpy@5466842，`git apply` 一次成功、正好 8 文件，
重跑幂等）→ 接收者的 `code/aerial_gym_simulator` 与本工作区 `diff -r`
逐字节一致（除 `__pycache__`/egg-info）→ 离线回归 5/5 →
`run_bench.py --steps 10 --device cuda:3` GPU 冒烟 `success`。
**唯一没验的是 Dockerfile 的完整 build**（27GB 镜像，共享盘不重建）——
它和现用镜像的 Dockerfile 同源，只多 pip 层与 PYTHONPATH 两段。

---

## 回滚

旧镜像保留为 tag `isaacgym:pre-uid1019`（uid 1000 版本）。
确认新环境没问题后可以删掉：`docker rmi isaacgym:pre-uid1019`。
