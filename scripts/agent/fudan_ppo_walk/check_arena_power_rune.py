"""Compare the canonical and edited compiled RMUC field: central fixtures, exposed terrain and surrounding structures."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from PIL import Image

from sim2sim.rmuc2027_arena import (ARENA, CHECKPOINT, REGIONAL_CHECKPOINT, ArenaPlayback,
                                   ground_height, model_cache, regional_metadata)
from sim2sim.rmuc2027_field import CENTRAL_DOCKS, CENTRAL_VISUALS, default_field
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def snapshot(field, output):
    _, meta = load_policy(CHECKPOINT)
    model = mujoco.MjModel.from_binary_path(str(model_cache(meta, field)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0., 0., 1.75)
    camera.distance, camera.azimuth, camera.elevation = 5.9, 125., -10.
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output/'centre.png')
    for i in range(model.ngeom):
        model.geom_rgba[i, 3] = float(model.geom(i).name.startswith('arena/'))
    dx = np.array([-1., 1., 0.])/np.sqrt(2.)
    along = np.array([1., 1., 0.])/np.sqrt(2.)
    def probe(origin, direction):
        hits = []
        for group in (2, 3):
            mask = np.zeros(6, dtype=np.uint8)
            mask[group] = 1
            gid = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, origin, direction, mask, True, -1, gid)
            hits.append(float(distance))
        return hits
    swept = []
    for z in np.linspace(1.79, 3.51, 37):
        for u in np.linspace(-.81, .81, 33):
            swept.append(probe(along*u - dx*.7 + [0, 0, z], dx))
    # 水平穿过支架、核心和平台，射线段只取中央装置所在的宽度。
    fixtures = [probe(along*u-dx*1.8+[0, 0, z], dx)
                for z in (.81, 1., 1.2, 1.42, 1.61, 1.7, 2., 2.4, 2.65)
                for u in np.linspace(-1.8, 1.8, 73)]
    down = np.array([0.,0.,-1.])
    adjacent = [probe(np.array([x,y,5.]), down)
                for x, y in ((-5.,-3.),(-5.76,-4.2),(-5.76,-5.75),(5.76,5.75))]
    origins = [along*u + dx*v + [0,0,4.] for u in np.linspace(-1.9,1.9,35)
               for v in np.linspace(-.9,.9,19)]
    if field == ARENA:
        # 独立取得原图裸地高度。保留碰撞块名单来自实际导出的、新场地的 XML。
        terrain = default_field(ARENA).parent
        kept = {e.get('name') for file in terrain.glob('*_geoms.xml')
                for e in ET.parse(file).iter('geom')}
        for i in range(model.ngeom):
            if model.geom(i).name.removeprefix('arena/') not in kept:
                model.geom_rgba[i, 3] = 0.
    floor = [probe(origin, down) for origin in origins]
    # 由内到外扫过两条与原台面等高的接缝，另查孔内的两层高度。
    seams = []
    for sign in (-1, 1):
        for offset in np.linspace(-.01,.01,41):
            for u,v in ((1.238+offset,0.), (1.10,-.115+offset), (1.13,offset)):
                seams.append(probe(sign*(along*u+dx*v)+[0,0,4.], down))
    np.savez(output/'rays.npz', swept=swept, fixtures=fixtures, adjacent=adjacent, floor=floor, seams=seams)
    (output/'geometry.json').write_text(json.dumps(dict(xml=str(field), ngeom=model.ngeom))+'\n')


def check(root):
    with np.load(root/'source'/'rays.npz') as old, np.load(root/'edited'/'rays.npz') as new:
        def counts(rows, extent):
            return ((rows >= 0.) & (rows <= extent)).sum(0).tolist()
        before, after = counts(old['swept'], 1.4), counts(new['swept'], 1.4)
        old_fixtures, new_fixtures = counts(old['fixtures'], 3.6), counts(new['fixtures'], 3.6)
        assert min(old_fixtures) > 0 and max(new_fixtures) == 0, (old_fixtures, new_fixtures)
        holes = (old['floor'] < 0.).all(1)
        assert holes.any() and (new['floor'] > 0.).all()
        levels = np.array([.599, .6015])
        patch_error = float(np.max(np.abs(4. - new['floor'][holes] - levels)))
        seam_error = float(np.max(np.abs(4. - new['seams'] - levels)))
        assert max(patch_error, seam_error) < 1e-5, (patch_error, seam_error)
        floor_delta = float(np.max(np.abs(old['floor'][~holes] - new['floor'][~holes])))
        assert floor_delta < 1e-5, floor_delta
        delta = float(np.max(np.abs(old['adjacent'] - new['adjacent'])))
        assert min(before) > 0 and max(after) == 0 and delta < 1e-5, (before, after, delta)
        assert (new['adjacent'] >= 0.).any(0).all()
        result = dict(queries_per_layer=len(new['swept']), layers=['visual','collision'],
            rune_hits_before=before, rune_hits_after=after, adjacent_queries_per_layer=len(new['adjacent']),
            adjacent_max_distance_error_m=delta, fixture_queries_per_layer=len(new['fixtures']),
            fixture_hits_before=old_fixtures, fixture_hits_after=new_fixtures,
            floor_queries_per_layer=len(new['floor']), filled_empty_samples=int(holes.sum()),
            seam_queries_per_layer=len(new['seams']), seam_max_height_error_m=seam_error,
            floor_max_distance_error_m=floor_delta)
    for case in ('source', 'edited'):
        result[case] = json.loads((root/case/'geometry.json').read_text())
    source_dir = Path(result['source']['xml']).parent
    edited_dir = Path(result['edited']['xml']).parent
    # 对所有保留碰撞块核对名字、变换、材质和顶点；远处地形也纳入检查。
    counts = {}
    for name, tag in [('collision_geoms', 'geom'), ('collision_assets', 'mesh'),
                      ('surface_repair_geoms', 'geom'), ('surface_repair_assets', 'mesh')]:
        source = {n.get('name'): n.attrib for n in ET.parse(source_dir/(name+'.xml')).iter(tag)}
        edited = {n.get('name'): n.attrib for n in ET.parse(edited_dir/(name+'.xml')).iter(tag)}
        assert all(source[key] == attrs for key, attrs in edited.items())
        counts[name] = len(edited)
    original = ET.parse(source_dir/'model.xml')
    body = original.find('worldbody')
    for node in list(body):
        if node.tag == 'body' and node.get('name') in CENTRAL_DOCKS:
            body.remove(node)
    edited = ET.parse(edited_dir/'model.xml')
    edited_world = edited.find('worldbody')
    covers = [n for n in edited_world.findall('geom') if n.get('name','').startswith('rmuc2027_mount_cover_')]
    assert len(covers) == 4
    for n in covers:
        edited_world.remove(n)
    asset = edited.find('asset')
    assert len(asset) == 1 and asset[0].get('name') == 'rmuc2027_mount_cover'
    edited.getroot().remove(asset)
    assert ET.tostring(original.getroot()) == ET.tostring(edited.getroot())
    assert not any(e.get('name') in CENTRAL_VISUALS for e in ET.parse(edited_dir/'visual_geoms.xml').iter('geom'))
    for name in ('closure_geoms.xml', 'closure_assets.xml'):
        assert (source_dir/name).read_bytes() == (edited_dir/name).read_bytes()
    result['identical_retained_entries'] = counts
    before_image = Image.open(root/'source'/'centre.png')
    after_image = Image.open(root/'edited'/'centre.png')
    comparison = Image.new('RGB', (before_image.width * 2, before_image.height))
    comparison.paste(before_image, (0,0))
    comparison.paste(after_image, (before_image.width,0))
    comparison.save(root/'before-after.png')
    assert result['source']['ngeom'] - result['edited']['ngeom'] == 10931+12472+18+18-4
    (root/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)


@torch.inference_mode()
def crossing(output):
    """Drive WALK 6838 across each cover with one wheel aligned to its centre."""
    policy, meta = load_policy(REGIONAL_CHECKPOINT)
    meta = regional_metadata(meta)
    model = mujoco.MjModel.from_binary_path(str(model_cache(meta, default_field(ARENA))))
    env = ArenaPlayback(model, meta)
    results = {}
    for side, sign in [('positive',1.), ('negative',-1.)]:
        u = sign*np.array([1.,1.])/np.sqrt(2.)
        v = sign*np.array([-1.,1.])/np.sqrt(2.)
        # 斜穿补面，另一侧轮子走原台面，避开支架旁原有的落差。
        xy = 1.401*u + .106*v
        heading = np.deg2rad(250.)
        forward = np.cos(heading)*u + np.sin(heading)*v
        env.command[:] = (.7,0.,.26)
        env.reset((*xy,ground_height(model,*xy)), np.arctan2(forward[1],forward[0]))
        ids = [model.body(n).id for n in ('lfwheel','rfwheel')]
        wheel = min(ids, key=lambda body: np.linalg.norm(env.data.xpos[body,:2] - 1.135*u))
        start = float(env.data.xpos[wheel,:2] @ v)
        cover = model.geom(f'arena/rmuc2027_mount_cover_{side}_collision').id
        force = np.empty(6)
        contact_ms = 0
        def sample():
            nonlocal contact_ms
            for i,c in enumerate(env.data.contact):
                if cover not in c.geom or wheel not in model.geom_bodyid[c.geom] or c.efc_address < 0:
                    continue
                mujoco.mj_contactForce(model, env.data, i, force)
                if force[0] > 1.:
                    contact_ms += 1
                    break
        upright = 1.
        for tick in range(150):
            obs = env.observation()
            env.history[-19:-16] = obs[6:9]
            action,_ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
            env.step(action[0].numpy(), sample=sample)
            end = float(env.data.xpos[wheel,:2] @ v)
            upright = min(upright, float(-env.gravity()[2]))
            if end < -.125 or upright < .7:
                break
        results[side] = dict(wheel=model.body(wheel).name, start_v_m=start, end_v_m=end,
            cover_loaded_ms=contact_ms, min_upright_cos=upright, policy_steps=tick+1,
            crossed=start>.119 and end<-.115 and contact_ms>0 and upright>.7)
    (output/'crossing.json').write_text(json.dumps(results, indent=2)+'\n')
    print(results)
    assert all(case['crossed'] for case in results.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('source','edited','check','crossing'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'crossing':
        torch.set_num_threads(1)
        crossing(args.output)
    elif args.phase == 'check':
        check(args.output)
    else:
        dest = args.output/args.phase
        dest.mkdir(parents=True, exist_ok=False)
        snapshot(ARENA if args.phase == 'source' else default_field(ARENA), dest)
