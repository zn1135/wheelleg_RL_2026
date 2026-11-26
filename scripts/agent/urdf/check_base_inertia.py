"""Read-only base_link mass/inertia scale comparison for a CAD URDF ZIP.

Valid inertia inequalities do not establish hardware accuracy. The mesh's
outer bounds describe occupied geometry, not its mass distribution.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import trimesh


def values(root):
    inertial = root.find("link[@name='base_link']/inertial")
    mass = float(inertial.find("mass").get("value"))
    inertia = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    matrix = np.array([[inertia["ixx"], inertia["ixy"], inertia["ixz"]],
                       [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
                       [inertia["ixz"], inertia["iyz"], inertia["izz"]]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    return dict(
        mass_kg=mass, inertia_kg_m2=inertia,
        com_m=np.fromstring(inertial.find("origin").get("xyz"), sep=" ").tolist(),
        principal_inertia_kg_m2=eigenvalues.tolist(),
        principal_gyration_radius_mm=(np.sqrt(eigenvalues / mass) * 1000).tolist(),
        valid_inertia_conditions=bool(mass > 0 and eigenvalues[0] > 0
                                      and eigenvalues[2] <= eigenvalues[:2].sum() + 1e-12),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        names = [name for name in archive.namelist() if name.endswith(".urdf")]
        assert len(names) == 1
        root = ET.fromstring(archive.read(names[0]))
        base = root.find("link[@name='base_link']")
        filename = Path(base.find("visual/geometry/mesh").get("filename")).name
        candidates = [name for name in archive.namelist() if name.endswith("/meshes/" + filename)]
        assert len(candidates) == 1
        mesh = trimesh.load(BytesIO(archive.read(candidates[0])), file_type="stl", process=False)
    current = values(root)
    current.update(mesh_bounds_m=mesh.bounds.tolist(), mesh_extents_mm=(mesh.extents * 1000).tolist(),
                   mesh_faces=len(mesh.faces))
    reference = values(ET.parse(args.reference).getroot())
    result = dict(
        input_archive=str(args.archive), reference_urdf=str(args.reference),
        archive_base_link=current, reference_base_link=reference,
        scope="Static parameter/mesh bounds comparison. CAD materials and measured mass distribution remain unverified.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
