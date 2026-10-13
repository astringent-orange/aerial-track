# 无人机跟踪测试环境 —— 镜像构建文件
#
# 在**解包后的 IsaacGym Preview 4 目录**里构建（build_image.sh 会替你组装）：
#   ./build_image.sh /path/to/IsaacGym_Preview_4_Package.tar
#
# 与开发机上那份 isaacgym/docker/Dockerfile 的区别：
#   开发机把 pip 依赖装在 bind-mount 的 container_home/.local 里（docker rm 即丢），
#   这份交接版把 pip 依赖**直接烧进镜像**，容器只挂 code/，重建容器零成本。
#   另外把 aerial_gym / urdfpy 用 PYTHONPATH 指到 /workspace（它们不随镜像走，
#   由 scripts/setup_workspace.sh 从上游 clone + 打补丁）。
#
# IsaacGym 是 NVIDIA 许可软件，**不随交接包分发**，需自行从 NVIDIA 开发者渠道获取。

FROM nvcr.io/nvidia/pytorch:22.09-py3
ENV DEBIAN_FRONTEND=noninteractive

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym

# 用宿主机的 uid/gid 建用户，bind-mount 出来的文件才归你所有。
# build_image.sh 会传 --build-arg UID=$(id -u) GID=$(id -g)；
# 如果 useradd 报 "UID already in use"，说明和基础镜像里已有用户撞了，
# 换个 UID 重新 build（见交接包 README §7）。
ARG UID=1019
ARG GID=1019
RUN groupadd --gid ${GID} gymuser 2>/dev/null || true \
 && useradd --create-home --uid ${UID} --gid ${GID} gymuser
USER gymuser

# copy gym repo to docker
COPY --chown=gymuser . .

# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"
RUN cd python && pip install -q -e .

# Aerial Gym requirement: Isaac Gym's argparser otherwise swallows CLI flags
# that the learning frameworks need (see Aerial Gym docs, "Change argument
# parser in Isaac Gym's gymutil.py").
RUN sed -i 's/^    args = parser\.parse_args()$/    args, _ = parser.parse_known_args()/' \
        /opt/isaacgym/python/isaacgym/gymutil.py \
 && grep -q "args, _ = parser.parse_known_args()" /opt/isaacgym/python/isaacgym/gymutil.py

# ---- 本项目额外的 Python 依赖 ----
# 版本组合在开发机上验证过，别单独升某个包（numpy/scipy 必须一起动，见
# requirements.txt 里的注释）。装在用户目录 /home/gymuser/.local，
# 随镜像走，重建容器不丢。
COPY --chown=gymuser docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir --no-deps \
      https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/pytorch3d-0.7.3-cp38-cp38-linux_x86_64.whl \
 && python -c "import pytorch3d, warp, gymnasium, gym, trimesh, networkx, collada, imageio"

# aerial_gym 与 urdfpy 是纯 Python 包，由 scripts/setup_workspace.sh
# clone 到 /workspace 下（打完补丁的版本），这里只把它们加进模块搜索路径，
# 不做 pip install —— 这样换机器/换代码不用重装。
ENV PYTHONPATH=/workspace/aerial_gym_simulator:/workspace/urdfpy

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
