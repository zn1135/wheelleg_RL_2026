"""Apply the user-confirmed 2027 CAD screenshot's COM-centred base properties.

The screenshot is in kg, mm and kg*mm^2 with SolidWorks positive
cross-product notation. Its displayed rounding is retained as input precision.
"""
import hashlib
import json
from pathlib import Path
from shutil import copy2
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET = Path("resources/robots/2027_wheelleg")
OUTPUT = Path("artifacts/2027_wheelleg/20261126-cad-base-mass-verified")
SCREENSHOT = Path("/tmp/codex-clipboard-57dd9c1c-db2f-47af-8e41-88da6bda8272.png")


def fmt(values):
    return " ".join(f"{value:.17g}" for value in values)


def main():
    mass = 9.407
    com = np.array([15.656, 8.462, 10.609]) * .001
    diagonal = np.array([116086.939, 138426.327, 211918.467]) * 1e-6
    # SolidWorks positive cross-products have the opposite sign to the
    # off-diagonal matrix elements used by URDF/MuJoCo.
    products = np.array([-148.474, -3083.927, 16.835]) * 1e-6
    inertia = np.diag(diagonal)
    inertia[0, 1] = inertia[1, 0] = -products[0]
    inertia[0, 2] = inertia[2, 0] = -products[1]
    inertia[1, 2] = inertia[2, 1] = -products[2]
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    quaternion = Rotation.from_matrix(axes).as_quat(scalar_first=True)
    displayed_moments = np.array([115986.810, 138427.314, 212017.610]) * 1e-6
    np.testing.assert_allclose(moments, displayed_moments, rtol=0, atol=1e-9)
    assert moments[0] > 0 and moments[2] < moments[:2].sum()
    paths = [ASSET / "urdf/wheelleg_2027.urdf",
             ASSET / "urdf/wheelleg_2027_train.urdf",
             ASSET / "mjlab/wheelleg_2027.xml"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    if SCREENSHOT.is_file():
        copy2(SCREENSHOT, OUTPUT / "source.png")
    manifest = []
    for path in paths:
        before = OUTPUT / "before" / path.relative_to(ASSET)
        before.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, before)
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag == "robot":
            part = root.find("link[@name='base_link']/inertial")
            part.find("mass").set("value", f"{mass:.3f}")
            part.find("origin").attrib.update(xyz=fmt(com), rpy="0 0 0")
            for key, i, j in (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                              ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)):
                part.find("inertia").set(key, f"{inertia[i, j]:.12g}")
        else:
            part = root.find(".//body[@name='base_link']/inertial")
            part.attrib.clear()
            # Supplying the diagonalized tensor avoids the compiler's
            # approximate full-inertia eigen decomposition.
            part.attrib.update(pos=fmt(com), mass=f"{mass:.3f}",
                               diaginertia=fmt(moments), quat=fmt(quaternion))
        tree.write(path, encoding="utf-8", xml_declaration=True)
        # Prove the only XML subtree change is the chassis inertial property.
        check = ET.parse(before)
        if root.tag == "robot":
            old = check.getroot().find("link[@name='base_link']")
            old.remove(old.find("inertial"))
            new = root.find("link[@name='base_link']")
        else:
            old = check.getroot().find(".//body[@name='base_link']")
            old.remove(old.find("inertial"))
            new = root.find(".//body[@name='base_link']")
        inertial = new.find("inertial")
        new.remove(inertial)
        assert ET.tostring(root) == ET.tostring(check.getroot())
        manifest.append(dict(path=str(path), before=hashlib.sha256(before.read_bytes()).hexdigest(),
                             after=hashlib.sha256(path.read_bytes()).hexdigest()))

    model = mujoco.MjModel.from_xml_path(str(paths[2].resolve()))
    body = model.body("base_link")
    rotation = Rotation.from_quat(body.iquat, scalar_first=True).as_matrix()
    actual = rotation @ np.diag(body.inertia) @ rotation.T
    np.testing.assert_allclose(actual, inertia, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(body.ipos, com, rtol=0, atol=1e-15)
    assert float(body.mass[0]) == mass
    report = dict(
        robot="2027_wheelleg", user_confirmed_source="latest 001 CAD, screenshot base-link coordinate frame",
        source_precision="mass displayed to 0.001 kg, COM to 0.001 mm, moments to 0.001 kg*mm^2",
        mass_kg=mass, com_m=com.tolist(), inertia_matrix_kg_m2=inertia.tolist(),
        principal_moments_kg_m2=moments.tolist(),
        principal_gyration_radius_m=np.sqrt(moments / mass).tolist(),
        converted_products_sign="URDF Ixy/Ixz/Iyz are the negatives of screenshot positive cross-products",
        model_mass_kg=float(model.body_mass.sum()), files=manifest,
    )
    (OUTPUT / "cad_source_and_conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
