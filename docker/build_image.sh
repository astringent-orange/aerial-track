#!/usr/bin/env bash
# 构建 isaacgym 镜像（一次性，约 30 分钟，视网络而定）。
#
#   ./build_image.sh /path/to/IsaacGym_Preview_4_Package.tar [镜像名]
#
# 参数是 IsaacGym Preview 4 的官方压缩包（.tar 或 .tar.gz 都行）。
# 该包是 NVIDIA 许可软件，**不随交接包分发**，需自行申请下载。
#
# 做的事：解包到临时目录 → 覆盖成本交接包的 Dockerfile（含 gymutil 补丁和
# pip 依赖）→ docker build。产物镜像默认叫 isaacgym。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"

TAR="${1:?用法: ./build_image.sh /path/to/IsaacGym_Preview_4_Package.tar [镜像名]}"
TAG="${2:-isaacgym}"

if [[ ! -f "$TAR" ]]; then
    echo "找不到 IsaacGym 压缩包: $TAR" >&2
    exit 1
fi

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

echo "==> 解包 IsaacGym 到临时构建目录..."
tar -xf "$TAR" -C "$CTX"
ROOT="$(find "$CTX" -maxdepth 1 -mindepth 1 -type d | head -1)"
if [[ -z "$ROOT" || ! -d "$ROOT/python/isaacgym" ]]; then
    echo "压缩包结构不对：期望里面是 isaacgym/ 一层目录（含 python/isaacgym/）" >&2
    exit 1
fi

echo "==> 覆盖 Dockerfile / Vulkan-EGL 配置..."
mkdir -p "$ROOT/docker"
cp "$HERE/Dockerfile" "$ROOT/docker/Dockerfile"
cp "$HERE/nvidia_icd.json" "$HERE/10_nvidia.json" "$HERE/requirements.txt" "$ROOT/docker/"

echo "==> docker build -t $TAG（UID=$(id -u) GID=$(id -g)）..."
docker build --network host \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -t "$TAG" \
    -f "$ROOT/docker/Dockerfile" \
    "$ROOT"

echo "==> 完成。下一步: bash scripts/setup_workspace.sh"
