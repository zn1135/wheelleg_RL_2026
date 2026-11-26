"""Check only the 001 CAD thigh-stop/calf-plate pairs across knee rotation.

The component face counts identify the exact source meshes used for this 001
export. This is a static local stop check, not a whole-robot collision check.
"""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

DIRECTORY = Path("artifacts/urdf-align/20261126-001-closed")
OUTPUT = Path("artifacts/urdf-align/20261126-001-limit-review")


def pose(element):
    result = np.eye(4)
    result[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    result[:3, :3] = Rotation.from_quat(
        np.fromstring(element.get("quat", "1 0 0 0"), sep=" "),
        scalar_first=True,
    ).as_matrix()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DIRECTORY / "robot_001_closed_mouse_drag.xml")
    parser.add_argument("--output", type=Path, default=OUTPUT / "knee_stops.json")
    args = parser.parse_args()
    root = ET.parse(args.model).getroot()
    results = {}
    for prefix, sizes in (("lf", (236, 1636)), ("rf", (404, 1634))):
        parts = []
        transforms = []
        for name, count in zip((prefix + "0", prefix + "1"), sizes, strict=True):
            body = root.find(f".//body[@name='{name}']")
            geom = body.find("geom")
            mesh = trimesh.load_mesh(DIRECTORY / "meshes" / f"{name}.STL")
            candidates = [part for part in mesh.split(only_watertight=False)
                          if len(part.faces) == count]
            assert len(candidates) == 1 and candidates[0].is_watertight
            parts.append(candidates[0])
            transforms.append(pose(geom))

        knee = root.find(f".//body[@name='{prefix}1']")
        joint = knee.find("joint")
        origin = pose(knee)
        axis = np.fromstring(joint.get("axis"), sep=" ")
        manager = trimesh.collision.CollisionManager()
        manager.add_object("thigh_stop", parts[0], transforms[0])

        def collides(angle):
            hinge = np.eye(4)
            hinge[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
            return manager.in_collision_single(parts[1], origin @ hinge @ transforms[1])

        angles = np.linspace(-np.pi, np.pi, 3001)
        hits = np.array([collides(q) for q in angles])
        transitions = []
        for i in np.flatnonzero(np.diff(hits)):
            left, right = angles[i:i + 2]
            left_hit = hits[i]
            for _ in range(30):
                middle = (left + right) / 2
                if collides(middle) == left_hit:
                    left = middle
                else:
                    right = middle
            transitions.append(float((left + right) / 2))
        ranges = np.fromstring(joint.get("range"), sep=" ")
        sample = np.linspace(*ranges, 1001)
        result = dict(
            scope="001 CAD thigh stop versus calf plate only",
            configured_range_rad=ranges.tolist(),
            local_contact_boundaries_rad=transitions,
            collide_at_configured_ends=[bool(collides(q)) for q in ranges],
            configured_range_collision_samples=int(sum(collides(q) for q in sample)),
            sample_count=1001,
        )
        print(prefix, result, flush=True)
        results[prefix] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
