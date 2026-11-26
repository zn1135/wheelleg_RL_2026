"""Compare paired CAD mass/COM/inertias in aligned frames across the Y mirror.

This is a read-only parameter comparison. It cannot establish true hardware
symmetry or independently validate the CAD material assignments.
"""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET = Path("resources/robots/2027_wheelleg")
OUTPUT = Path("artifacts/2027_wheelleg/20261126-side-mass-review/report.json")


def properties(link):
    inertial = link.find("inertial")
    origin = inertial.find("origin")
    attributes = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    tensor = np.array([
        [attributes["ixx"], attributes["ixy"], attributes["ixz"]],
        [attributes["ixy"], attributes["iyy"], attributes["iyz"]],
        [attributes["ixz"], attributes["iyz"], attributes["izz"]],
    ])
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
    return dict(
        mass=float(inertial.find("mass").get("value")),
        com=np.fromstring(origin.get("xyz"), sep=" "),
        inertia=rotation @ tensor @ rotation.T,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = ET.parse(ASSET / "urdf/wheelleg_2027.urdf").getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    mirror = np.diag([1., -1., 1.])
    pairs = []
    for suffix in ("0", "1", "wheel", "00", "01", "02", "03", "001", "002", "04", "05"):
        left, right = properties(links["lf" + suffix]), properties(links["rf" + suffix])
        reflected = mirror @ left["inertia"] @ mirror
        diff = right["inertia"] - reflected
        mass_delta = right["mass"] - left["mass"]
        row = dict(
            left="lf" + suffix, right="rf" + suffix,
            left_mass_kg=left["mass"], right_mass_kg=right["mass"],
            right_minus_left_mass_g=mass_delta * 1000,
            mass_difference_fraction=mass_delta / ((left["mass"] + right["mass"]) / 2),
            left_Iyy_kg_m2=float(left["inertia"][1, 1]),
            right_Iyy_kg_m2=float(right["inertia"][1, 1]),
            Iyy_difference_fraction=float((right["inertia"][1, 1] - left["inertia"][1, 1]) /
                                          ((right["inertia"][1, 1] + left["inertia"][1, 1]) / 2)),
            mirrored_tensor_relative_difference=float(np.linalg.norm(diff) / np.linalg.norm(reflected)),
            local_com_right_minus_mirrored_left_mm=((right["com"] - mirror @ left["com"]) * 1000).tolist(),
        )
        pairs.append(row)
        print(row["left"], row["right"], "mass kg", row["left_mass_kg"], row["right_mass_kg"],
              "delta g", round(row["right_minus_left_mass_g"], 6),
              "Iyy relative delta %", round(row["Iyy_difference_fraction"] * 100, 4))

    model = mujoco.MjModel.from_xml_path(str(ASSET / "mjlab/wheelleg_2027.xml"))
    spring_mass = {}
    for side in ("left", "right"):
        spring_mass[side] = sum(float(model.body(f"{side}_spring{i}").mass[0]) for i in (1, 2))
    left_total = sum(row["left_mass_kg"] for row in pairs)
    right_total = sum(row["right_mass_kg"] for row in pairs)
    np.testing.assert_allclose(left_total + right_total + properties(links["base_link"])["mass"]
                               + properties(links["imu-link"])["mass"],
                               sum(float(node.get("value")) for node in root.findall(".//mass")), atol=1e-12)
    assert spring_mass["left"] == spring_mass["right"]
    base = properties(links["base_link"])
    report = dict(
        source_urdf=str(ASSET / "urdf/wheelleg_2027.urdf"),
        left_cad_components_kg=left_total, right_cad_components_kg=right_total,
        right_minus_left_g=(right_total - left_total) * 1000,
        total_side_difference_fraction=(right_total - left_total) / ((left_total + right_total) / 2),
        paired_parts=pairs, spring_assembly_mass_kg=spring_mass,
        shared_base_mass_kg=base["mass"],
        shared_base_principal_gyration_radius_mm=(np.sqrt(np.linalg.eigvalsh(base["inertia"]) / base["mass"]) * 1000).tolist(),
        scope="Per-link CAD parameters around their own centers of mass, mirrored through Y. The single base_link combines both chassis sides.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_parts"}, indent=2))


if __name__ == "__main__":
    main()
