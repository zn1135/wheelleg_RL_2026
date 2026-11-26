"""Fudan wheel plant: direct unit-gain torque plus native joint mechanics."""
import hashlib
import json

import mujoco
import numpy as np

from ct_mjlab.envs.chuanliantui.model import MODEL_XML,load_spec

PARAMETERS=MODEL_XML.with_name('wheel_mechanical.json')


def record(profile):
    if profile=='legacy_curve':return dict(version='legacy-wheel-curve-v1')
    if profile!='mechanical':raise ValueError(f'Unknown wheel model: {profile}')
    values=json.loads(PARAMETERS.read_text())
    for name in ('armature','damping','frictionloss'):
        a=np.asarray(values[name],dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any():
            raise ValueError(f'Invalid mechanical wheel {name}')
    return dict(version='fudan-mechanical-wheel-v1',parameter_sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
        wheel_order=['lfwheel','rfwheel'],armature=values['armature'],damping=values['damping'],frictionloss=values['frictionloss'],
        actuator_gain=1.,speed_shaping=False,extra_output_gain=1.)


def mechanical_spec():
    spec=load_spec();values=record('mechanical')
    for i,name in enumerate(values['wheel_order']):
        joint=spec.joint(name)
        joint.armature=values['armature'][i]
        joint.damping[:]=0.;joint.damping[0]=values['damping'][i]
        joint.frictionloss=values['frictionloss'][i]
        motor=spec.actuator(name+'_motor');motor.gaintype=mujoco.mjtGain.mjGAIN_FIXED
        motor.gainprm[:]=0.;motor.gainprm[0]=1.
        # V=1,p=0只给历史诊断提供恒等占位；实际机械路径直接提交力矩。
        motor.gainprm[1]=1.
    return spec
