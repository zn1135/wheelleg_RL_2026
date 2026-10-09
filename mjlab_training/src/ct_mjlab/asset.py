"""The same CAD sites, connect constraints and native spring tendons in CPU/GPU."""
import hashlib
import json
from pathlib import Path
import mujoco
from .config import ASSET, DT
POSES=json.loads(Path(__file__).with_name('poses.json').read_text())

def robot_spec():
    if hashlib.sha256(ASSET.read_bytes()).hexdigest()!=POSES['asset_sha256']:
        raise ValueError('Asset changed: regenerate physical closed-chain pose cache')
    spec=mujoco.MjSpec.from_file(str(ASSET))
    # Scene supplies a single ground; ground must not be a moving entity child.
    spec.delete(spec.geom('floor'))
    for key in list(spec.keys):spec.delete(key)
    # Scene.attach discards child solver options. SimulationCfg is authoritative.
    defaults=mujoco.MjSpec().option
    for name in ('timestep','tolerance','impratio','integrator','cone'):
        setattr(spec.option,name,getattr(defaults,name))
    # Source ground friction is 1 and combines by multiplication. MuJoCo's
    # default max would erase wheel friction DR; robot priority selects its value.
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.priority=1
    return spec

def cpu_model():
    model=mujoco.MjModel.from_xml_path(str(ASSET));model.opt.timestep=DT
    floor=model.geom('floor').id
    model.geom_friction[floor,0]=1.
    collision=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
    model.geom_priority[collision]=1;model.geom_priority[floor]=0
    return model
