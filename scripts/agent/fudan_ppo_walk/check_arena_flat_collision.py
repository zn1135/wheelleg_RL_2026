"""Compare the original and merged RMUC field geometry and red-step replay."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from sim2sim.rmuc2027_arena import REGIONAL_CHECKPOINT, model_cache
from sim2sim.rmuc2027_flat_collision import LAYERS, protected, world_vertices
from .review_arena_four_policies import load


def original_assets(source, candidate):
    groups = json.loads(candidate.with_name('merged.json').read_text())
    audit = {}
    for layer in LAYERS:
        mesh_name = f'{layer}_assets.xml'
        geom_name = f'{layer}_geoms.xml'
        old_meshes, new_meshes = [
            {n.get('name'): n for n in ET.parse(field.with_name(mesh_name)).iter('mesh')}
            for field in (source, candidate)]
        old_geoms, new_geoms = [
            {n.get('name'): n for n in ET.parse(field.with_name(geom_name)).iter('geom')}
            for field in (source, candidate)]
        consumed = {name for group in groups[layer] for name in group['replaces']}
        assert consumed == set(old_geoms)-set(new_geoms)
        assert {group['name'] for group in groups[layer]} == set(new_geoms)-set(old_geoms)
        for name in set(old_geoms)-consumed:
            old, new = old_geoms[name], new_geoms[name]
            assert old.attrib == new.attrib, name
            ref = old.get('mesh')
            if ref:
                assert old_meshes[ref].attrib == new_meshes[ref].attrib, ref
        for name in consumed:
            vertices = world_vertices(old_geoms[name], old_meshes)
            assert len(np.unique(vertices[:, 2].round(6))) == 2
            assert not protected(vertices.min(axis=0), vertices.max(axis=0))
        audit[layer] = dict(merged_groups=len(groups[layer]), replaced=len(consumed),
                            original_geoms_retained=len(old_geoms)-len(consumed))
    for name in ('visual_assets.xml', 'visual_geoms.xml', 'closure_assets.xml', 'closure_geoms.xml'):
        assert source.with_name(name).read_bytes() == candidate.with_name(name).read_bytes()
    return audit


def sample_geometry(checkpoint, arena):
    model = mujoco.MjModel.from_binary_path(str(model_cache(checkpoint, arena)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.geom_rgba[:, 3] = [
        float(model.geom(i).name.startswith('arena/')) for i in range(model.ngeom)]
    groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    down = np.array([0., 0., -1.])
    forward = np.array([0., -1., 0.])
    gid = np.zeros(1, dtype=np.int32)
    def ray(x, y, z, direction):
        return mujoco.mj_ray(model, data, np.array([x,y,z]), direction, groups, True, -1, gid)
    queries = dict(
        field_grid=[ray(x,y,5.,down) for x in np.linspace(-14,14,113) for y in np.linspace(-8,8,65)],
        continuous_slope=[ray(x,y,5.,down) for x in (-5.97,-5.76,-5.55,-5.)
                          for y in np.linspace(-5.4,-1.5,781)],
        red_tread=[ray(x,y,5.,down) for x in (-5.97,-5.76,-5.55)
                   for y in np.linspace(-7.4,-4.2,641)],
        red_face=[ray(x,-4.9,z,forward) for x in (-5.97,-5.76,-5.55)
                  for z in np.linspace(-.14,.14,57)])
    return queries, dict(ngeom=model.ngeom, world_geoms=int(model.body_geomnum[0]),
                         compiled_bvh_visualization=bool(model.vis.global_.bvactive))


def step_metrics(path):
    report, state, roles, contacts = load(path)
    assert report['spawn'][:2] == [-5.76,-4.2] and report['spawn_yaw_deg'] == -90.
    assert report['timing'] == {'physics_dt':.001,'pd_dt':.005,'policy_dt':.01}
    transitions = np.r_[0, np.flatnonzero(roles[1:] != roles[:-1])+1]
    assert roles[transitions].tolist() == ['walk','step','parking','step','walk']
    stop, restart, reset = transitions[2:]
    assert (state[stop:restart,1:3] == 0.).all() and (state[restart:reset,1] == 2.).all()
    parking_speed = float(np.linalg.norm(state[restart-100:restart,7:9], axis=1).max())
    loaded = bool((np.abs(contacts[(restart-100)*10:restart*10,:2])>1.).all())
    parked_y = float(state[restart-1,5])
    assert parked_y < -5.405 and parking_speed <= .05 and loaded
    return dict(parking_end_y_m=parked_y, last_parking_second_peak_m_s=parking_speed,
                both_wheels_loaded=loaded, restart_end_speed_m_s=float(np.linalg.norm(state[reset-1,7:9])),
                nonwheel_contact_during_approach_ms=int((contacts[transitions[1]*10:stop*10,2]>1.).sum()),
                realtime_factor=report['simulated_seconds']/report['wall_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for flag in ('source','candidate','baseline-step','optimized-step','output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    report = dict(xml_audit=original_assets(args.source, args.candidate))
    meta = torch.load(REGIONAL_CHECKPOINT, map_location='cpu', weights_only=False)
    before, report['baseline_model'] = sample_geometry(meta, args.source)
    after, report['optimized_model'] = sample_geometry(meta, args.candidate)
    report['rays'] = {}
    for name in before:
        difference = np.abs(np.asarray(before[name])-after[name])
        assert np.array_equal(np.asarray(before[name]) < 0, np.asarray(after[name]) < 0), name
        assert difference.max() < 1e-4, (name, float(difference.max()))
        report['rays'][name] = dict(count=len(difference), max_error_m=float(difference.max()))
    report['baseline_step'], report['optimized_step'] = (step_metrics(path)
        for path in (args.baseline_step,args.optimized_step))
    with np.load(args.baseline_step/'trace.npz') as old, np.load(args.optimized_step/'trace.npz') as new:
        report['step_qpos_max_error'] = float(np.abs(old['qpos']-new['qpos']).max())
    report['speedup'] = report['optimized_step']['realtime_factor']/report['baseline_step']['realtime_factor']
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(report)
