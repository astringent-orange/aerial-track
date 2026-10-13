#!/usr/bin/env bash
# 创建（或进入）跟踪测试环境的容器。
#
#   bash scripts/start_container.sh
#
# 前提：镜像 isaacgym 已经 build 好（docker/build_image.sh），
#       code/ 下已有 tracking_bench + 打过补丁的 aerial_gym_simulator + urdfpy。
#
# 挂载两个目录（按功能分离）：
#   code/    -> /workspace   代码
#   outputs/ -> /outputs     运行产物（视频/指标/图，结构约定见 outputs/README.md）
# pip 依赖都烧在镜像里，所以**不需要**挂 container_home/.local —— 重建容器零成本。
set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd -P)"
NAME="${CONTAINER_NAME:-aerialgym_dev}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "starting existing container '$NAME'..."
    docker start "$NAME" >/dev/null
else
    echo "creating container '$NAME' (all GPUs visible)..."
    mkdir -p "$WS/outputs"
    docker run -d \
        --name "$NAME" \
        --gpus all \
        --ipc=host \
        --shm-size=8g \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        -v "$WS/code:/workspace" \
        -v "$WS/outputs:/outputs" \
        isaacgym \
        sleep infinity >/dev/null
fi

# 非交互调用（脚本/CI）不进 shell，只保证容器在跑
if [[ -t 0 && -t 1 ]]; then
    docker exec -it "$NAME" bash
else
    echo "container '$NAME' is running (非交互调用，跳过进 shell)"
fi
