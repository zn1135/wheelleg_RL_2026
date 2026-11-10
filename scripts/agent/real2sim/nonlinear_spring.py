"""Diagnostic axial force curves; no production defaults or measured-state injection."""
import numpy as np
from functools import lru_cache

from scripts.agent.real2sim.control_leg_pose_gpu import gas_spring_force


@lru_cache(maxsize=256)
def smooth_coefficients(knots,values):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(knots,values).c


def spring_forces(position, velocity, candidates):
    force=gas_spring_force(position,velocity)
    for world in range(len(position)):
        p=candidates[world%len(candidates)]
        if p.get('right_total_length_load_nm') is not None:
            force[world,1]=0.
            continue
        if 'spring_curve_force_n' not in p:continue
        knots=np.asarray(p['spring_curve_compression_m']);values=np.asarray(p['spring_curve_force_n'])
        assert len(knots)==len(values)>=2 and np.all(np.diff(knots)>0)
        assert np.isfinite(values).all() and values.min()>=0 and np.all(np.diff(values)>=-1e-8)
        assert p.get('remove_baseline_right_load',False) or p.get('curve_with_joint_correction',False)
        if p.get('remove_baseline_right_load',False):
            assert not np.any(p['bias_delta_nm']) and all(p.get(k,0)==0 for k in (
                'length_stiffness_delta_nm_rad','cross_stiffness_delta_nm_rad','length_quadratic_delta_nm_rad2'))
        compression=.01-position[world,1]
        if p.get('spring_curve_interpolation')=='pchip':
            coeff=smooth_coefficients(tuple(knots),tuple(values))
            x=np.clip(compression,knots[0],knots[-1]);i=int(np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2))
            z=x-knots[i];elastic=((coeff[0,i]*z+coeff[1,i])*z+coeff[2,i])*z+coeff[3,i]
        else:elastic=np.interp(compression,knots,values)
        force[world,1]=np.clip(elastic-p.get('spring_velocity_damping_ns_m',50.)*velocity[world,1],
                             0,p.get('spring_force_limit_n',600.))
    return force


def total_length_load(length,candidates):
    result=np.zeros((len(length),2))
    for world,x in enumerate(length):
        p=candidates[world%len(candidates)]
        if 'right_total_length_load_nm' not in p:continue
        knots=np.asarray(p['right_total_length_knots_m']);values=np.asarray(p['right_total_length_load_nm'])
        x=np.clip(x,knots[0],knots[-1]);i=int(np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2));z=x-knots[i]
        for motor in range(2):
            coeff=smooth_coefficients(tuple(knots),tuple(values[:,motor]))
            result[world,motor]=((coeff[0,i]*z+coeff[1,i])*z+coeff[2,i])*z+coeff[3,i]
    return result
