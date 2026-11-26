"""Audit CAD mass and inertias and account for added 2027 spring bodies.

This checks arithmetic and physical inertia conditions. Real total weight,
the CAD material assignments and copied spring masses need independent data.
"""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ET.parse(args.urdf).getroot()
    rows = {}
    invalid = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        a = {name: float(value) for name, value in inertial.find("inertia").attrib.items()}
        matrix = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                           [a["ixy"], a["iyy"], a["iyz"]],
                           [a["ixz"], a["iyz"], a["izz"]]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        if mass <= 0 or eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[:2].sum() + 1e-12:
            invalid.append(link.get("name"))
        rows[link.get("name")] = dict(
            mass_kg=mass,
            com_local_m=np.fromstring(inertial.find("origin").get("xyz"), sep=" ").tolist(),
            principal_inertia_kg_m2=eigenvalues.tolist(),
            principal_gyration_radius_m=np.sqrt(eigenvalues / mass).tolist(),
        )
    cad_mass = sum(row["mass_kg"] for row in rows.values())
    side_masses = [sum(value["mass_kg"] for name, value in rows.items() if name.startswith(prefix))
                   for prefix in ("lf", "rf")]
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    mass = model.body_mass.sum()
    com = (model.body_mass[:, None] * data.xipos).sum(axis=0) / mass
    additions = {model.body(i).name: float(value) for i, value in enumerate(model.body_mass)
                 if model.body(i).name not in rows and value > 0}
    assert abs(cad_mass + sum(additions.values()) - mass) < 1e-10
    report = dict(
        cad_mass_kg=cad_mass, mjcf_mass_kg=float(mass),
        base_mass_kg=rows["base_link"]["mass_kg"],
        base_mass_fraction=rows["base_link"]["mass_kg"] / cad_mass,
        left_right_cad_mass_kg=side_masses,
        added_bodies_kg=additions,
        mjcf_com_at_initial_pose_m=com.tolist(),
        links=rows, invalid_mass_or_inertia=invalid,
        pending=[
            f"Compare the complete physical robot weight with URDF {cad_mass:.6f} kg and the MJCF spring additions.",
            "Compare the current base mass/inertia against the selected CAD assembly and its material assignments.",
            "Spring barrel/rod masses are copied from chuanliantui, pending 001 measurement.",
            "400 N per spring is a nominal force; the actual force-length curve and damping remain uncalibrated.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name != "links"},
                     ensure_ascii=False, indent=2))
    assert not invalid, invalid


if __name__ == "__main__":
    main()
