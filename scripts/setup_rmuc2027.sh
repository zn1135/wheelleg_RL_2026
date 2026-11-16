#!/usr/bin/env bash
# 只下载固定版本场地资产，不安装依赖或修改训练环境。
set -euo pipefail
CT_ARENA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CT_ARENA_REV=ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001
CT_ARENA_DEST="$CT_ARENA_ROOT/artifacts/rmuc2027/upstream-$CT_ARENA_REV"
if [[ -f "$CT_ARENA_DEST/rmuc2027_mjarena/arena_mjcf/model.xml" ]]; then
    echo "$CT_ARENA_DEST"
    exit 0
fi
CT_ARENA_TMP=$(mktemp -d)
trap 'rm -rf -- "$CT_ARENA_TMP"' EXIT
curl --fail --location --retry 3 \
    "https://codeload.github.com/CosmosMount/rmuc2027_mjarena/tar.gz/$CT_ARENA_REV" \
    --output "$CT_ARENA_TMP/arena.tar.gz"
mkdir -p -- "$CT_ARENA_TMP/source"
tar -xzf "$CT_ARENA_TMP/arena.tar.gz" --strip-components=1 -C "$CT_ARENA_TMP/source"
mkdir -p -- "$(dirname -- "$CT_ARENA_DEST")"
mv -- "$CT_ARENA_TMP/source" "$CT_ARENA_DEST"
echo "$CT_ARENA_DEST"
