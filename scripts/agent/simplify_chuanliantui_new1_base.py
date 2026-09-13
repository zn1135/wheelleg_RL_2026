#!/usr/bin/env python3
"""将 chuanliantui_new_1 的 CAD 底盘网格降到 MuJoCo STL 面数上限以下。"""

from __future__ import annotations

from pathlib import Path

import fast_simplification
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "resources/robots/chuanliantui_new_1/meshes/base_link.STL"
OUTPUT = REPO_ROOT / "resources/robots/chuanliantui_new_1/meshes/base_link_simple.STL"
TARGET_FACES = 146_993


def main():
    mesh = trimesh.load_mesh(SOURCE, process=False)
    print("source: faces={}, vertices={}".format(len(mesh.faces), len(mesh.vertices)))
    mesh.merge_vertices()
    print("merged: faces={}, vertices={}".format(len(mesh.faces), len(mesh.vertices)))
    vertices, faces = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_count=TARGET_FACES, agg=7
    )
    simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(simplified.faces) > 200_000:
        raise RuntimeError("降面失败：{} faces 仍超过 MuJoCo 上限".format(len(simplified.faces)))
    simplified.export(OUTPUT)
    print("output: {} faces -> {}".format(len(simplified.faces), OUTPUT))


if __name__ == "__main__":
    main()
