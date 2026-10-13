#!/usr/bin/env bash
# 获取 aerial_gym 与 urdfpy 源码（都不随交接包分发），打上本地补丁。
# 只依赖 git，在宿主机跑即可（不用进容器）：
#
#   bash scripts/setup_workspace.sh
#
# 两个仓库都是上游开源代码，交接包只带**本地修改**（patches/aerial_gym_local.patch）。
# 版本钉死：
#   aerial_gym_simulator  @ 2.0.1（f0d0f05）
#   urdfpy                @ 5466842（GitHub master，PyPI 版有 cylinder mesh 的 bug）
#
# 网络说明：直连 github.com 超时的机器用镜像前缀，例如：
#   GITHUB_BASE=https://ghfast.top/https://github.com bash scripts/setup_workspace.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(dirname "$HERE")"          # 交接包根目录
CODE="$ROOT/code"
PATCH="$ROOT/patches/aerial_gym_local.patch"

GITHUB_BASE="${GITHUB_BASE:-https://github.com}"
AERIAL_GYM_COMMIT="f0d0f05283f7897bab5a1bcc7b19b91cebbab218"
URDFPY_COMMIT="5466842"

clone_pinned() {  # <url> <目录> <commit>
    local url="$1" dir="$2" commit="$3"
    if [[ -d "$dir/.git" ]]; then
        echo "==> $dir 已存在，跳过 clone（要重来先删掉它）"
    else
        echo "==> clone $url"
        git clone "$url" "$dir"
    fi
    git -C "$dir" checkout -q "$commit"
    echo "    @ $(git -C "$dir" rev-parse --short HEAD)"
}

mkdir -p "$CODE"

clone_pinned "$GITHUB_BASE/ntnu-arl/aerial_gym_simulator.git" \
             "$CODE/aerial_gym_simulator" "$AERIAL_GYM_COMMIT"

clone_pinned "$GITHUB_BASE/mmatl/urdfpy.git" \
             "$CODE/urdfpy" "$URDFPY_COMMIT"

echo "==> 打本地补丁 patches/aerial_gym_local.patch"
if git -C "$CODE/aerial_gym_simulator" apply --check "$PATCH" 2>/dev/null; then
    git -C "$CODE/aerial_gym_simulator" apply "$PATCH"
elif git -C "$CODE/aerial_gym_simulator" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "    补丁已经打过，跳过"
else
    echo "补丁打不上（版本不对或有过改动）。请确认 aerial_gym_simulator 是" >&2
    echo "  $AERIAL_GYM_COMMIT" >&2
    exit 1
fi

echo "==> 核对：本地修改应该正好 8 个文件（3 改 + 5 增）"
git -C "$CODE/aerial_gym_simulator" status --porcelain

echo "==> 完成。下一步: bash scripts/start_container.sh"
