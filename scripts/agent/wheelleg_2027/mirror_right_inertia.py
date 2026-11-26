"""Assign right-side masses and mirrored COM-centred inertias to left components.

Each component keeps its own CAD COM position because the mounting frames
have different axial offsets. The base chassis is a shared rigid body.
"""
from pathlib import Path
from shutil import copy2
from xml.etree import ElementTree as ET
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET = Path("resources/robots/2027_wheelleg")
OUTPUT = Path("artifacts/2027_wheelleg/20261126-right-to-left-verified")
SUFFIXES = ("0", "1", "wheel", "00", "01", "02", "03", "001", "002", "04", "05")
MIRROR = np.diag([1., -1., 1.])


def numbers(text):
    return np.fromstring(text, sep=" ")


def fmt(values):
    return " ".join(f"{value:.17g}" for value in values)


def urdf_tensor(inertial):
    values = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    matrix = np.array([
        [values["ixx"], values["ixy"], values["ixz"]],
        [values["ixy"], values["iyy"], values["iyz"]],
        [values["ixz"], values["iyz"], values["izz"]],
    ])
    rotation = Rotation.from_euler("xyz", numbers(inertial.find("origin").get("rpy"))).as_matrix()
    return rotation @ matrix @ rotation.T


def verify_urdf(path, before):
    root = ET.parse(path).getroot()
    old = ET.parse(before).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    old_links = {link.get("name"): link for link in old.findall("link")}
    for suffix in SUFFIXES:
        left = links["lf" + suffix].find("inertial")
        right = links["rf" + suffix].find("inertial")
        assert left.find("mass").attrib == right.find("mass").attrib
        np.testing.assert_allclose(urdf_tensor(left), MIRROR @ urdf_tensor(right) @ MIRROR,
                                   rtol=1e-12, atol=1e-16)
    for name, link in links.items():
        assert link.find("inertial/origin").attrib == old_links[name].find("inertial/origin").attrib
        for tag in ("visual", "collision"):
            assert ET.tostring(link.find(tag)) == ET.tostring(old_links[name].find(tag))
        if not name.startswith("lf"):
            assert ET.tostring(link) == ET.tostring(old_links[name])
    assert [ET.tostring(j) for j in root.findall("joint")] == [ET.tostring(j) for j in old.findall("joint")]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    paths = [ASSET / "urdf/wheelleg_2027.urdf",
             ASSET / "urdf/wheelleg_2027_train.urdf",
             ASSET / "mjlab/wheelleg_2027.xml"]
    for path in paths:
        saved = OUTPUT / "before" / path.relative_to(ASSET)
        saved.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, saved)

    for path in paths[:2]:
        tree = ET.parse(path)
        links = {link.get("name"): link for link in tree.getroot().findall("link")}
        for suffix in SUFFIXES:
            left = links["lf" + suffix].find("inertial")
            right = links["rf" + suffix].find("inertial")
            # Current canonical link axes are parallel to the chassis axes.
            np.testing.assert_allclose(numbers(left.find("origin").get("rpy")), 0, atol=1e-12)
            left.find("mass").set("value", right.find("mass").get("value"))
            tensor = MIRROR @ urdf_tensor(right) @ MIRROR
            for key, i, j in (("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2),
                              ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2)):
                left.find("inertia").set(key, f"{tensor[i, j]:.17g}")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        verify_urdf(path, OUTPUT / "before" / path.relative_to(ASSET))

    path = paths[2]
    before_model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    tree = ET.parse(path)
    root = tree.getroot()
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    pairs = {"lf" + suffix: "rf" + suffix for suffix in SUFFIXES}
    # In the CAD-derived MJCF these left names still identify the crossed
    # screw/bushing frames. Match physical components, not the suffix alone.
    pairs.update(lf002="rf04", lf04="rf002")
    for left_name, right_name in pairs.items():
        right = before_model.body(right_name)
        left = bodies[left_name].find("inertial")
        rotation = Rotation.from_quat(right.iquat, scalar_first=True).as_matrix()
        tensor = MIRROR @ rotation @ np.diag(right.inertia) @ rotation.T @ MIRROR
        position = left.get("pos")
        left.attrib.clear()
        # Reflect the principal axes with an extra eigenvector sign flip
        # so the orientation remains a proper quaternion rotation.
        orientation = MIRROR @ rotation @ np.diag([-1., 1., 1.])
        quaternion = Rotation.from_matrix(orientation).as_quat(scalar_first=True)
        left.attrib.update(pos=position, mass=f"{right.mass[0]:.17g}",
                           diaginertia=fmt(right.inertia), quat=fmt(quaternion))
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    for left_name, right_name in pairs.items():
        left = model.body(left_name)
        right = before_model.body(right_name)
        left_rotation = Rotation.from_quat(left.iquat, scalar_first=True).as_matrix()
        right_rotation = Rotation.from_quat(right.iquat, scalar_first=True).as_matrix()
        expected = MIRROR @ right_rotation @ np.diag(right.inertia) @ right_rotation.T @ MIRROR
        actual = left_rotation @ np.diag(left.inertia) @ left_rotation.T
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(left.mass, right.mass, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(left.ipos, before_model.body(left_name).ipos)
    for field in ("eq_data", "eq_solref", "eq_solimp", "jnt_range", "jnt_limited",
                  "jnt_axis", "geom_pos", "geom_quat", "body_pos", "body_quat",
                  "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(model, field), getattr(before_model, field))
    np.testing.assert_array_equal(model.body("base_link").inertia, before_model.body("base_link").inertia)
    for index in range(1, model.nbody):
        body = model.body(index)
        if body.name not in pairs:
            np.testing.assert_array_equal(body.mass, before_model.body(body.name).mass)
            np.testing.assert_array_equal(body.inertia, before_model.body(body.name).inertia)
    cad_root = ET.parse(paths[0]).getroot()
    cad_mass = sum(float(m.get("value")) for m in cad_root.findall(".//mass"))
    report = dict(
        urdf_mass_kg=cad_mass, mjcf_mass_kg=float(model.body_mass.sum()),
        previous_mjcf_mass_kg=float(before_model.body_mass.sum()),
        body_mass_kg=float(model.body("base_link").mass[0]),
        component_pairs=pairs, fields=["mass", "mirrored COM-centred inertia tensor"],
        com_source="Each component's existing CAD mounting position",
        scope="Right-to-left parameter assignment, no physical calibration",
    )
    (OUTPUT / "assignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
