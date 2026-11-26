"""在既有 GPU 回放中只读分部件传感器，归因第一次碰条与落地首秒的非轮接触。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import ContactType
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from . import evaluate as bars
from . import measure_flat as flat
from .scoring import score


def names_at(names, mask):
    return [names[i] for i in np.flatnonzero(mask)]


class SolverPairs:
    """按 worldid 与实际几何编号拆解求解器接触，独立于传感器对象匹配。"""
    def __init__(self, env):
        sim = env.env.sim
        model, data = sim.mj_model, sim.wp_data
        self.names = [model.body(i).name.removeprefix('robot/') for i in range(model.nbody)]
        self.body = torch.tensor(model.geom_bodyid.copy(), device=env.device, dtype=torch.long)
        self.ground = model.geom('terrain').id
        self.bar = model.geom('jump_bar').id if env.has_bar else -1
        self.pairs = wp.to_torch(data.contact.geom)
        self.world = wp.to_torch(data.contact.worldid)
        self.count = wp.to_torch(data.nacon)
        self.kind = wp.to_torch(data.contact.type)
        self.index = torch.arange(len(self.world),device=env.device)
        self.worlds = env.num_envs

    def frame(self):
        a,b = self.pairs.unbind(1)
        valid = (self.index < self.count[0]) & ((self.kind & int(ContactType.CONSTRAINT)) != 0)
        values = []
        for target in (self.ground,self.bar):
            matches = valid & (target>=0) & ((a==target) | (b==target))
            other = torch.where(a==target,b,a).long().clamp(0,len(self.body)-1)
            size = self.worlds*len(self.names)
            slots = torch.where(matches,self.world.long()*len(self.names)+self.body[other],size)
            counts = torch.zeros(size+1,device=a.device,dtype=torch.int32)
            counts.scatter_add_(0,slots,matches.int())
            values.append(counts[:-1].reshape(self.worlds,-1)>0)
        return torch.cat(values,1)


def collect(output, original, height, policies, solver_pairs, *, flat_reference=None, landing_handoff=False):
    """沿用原批次与策略调用顺序，读取未合并的 found / netforce 槽位。"""
    output.mkdir()
    chunks, ground_names, bar_names, wheel_names = [], [], [], []
    trace_frame = JumpEnv.trace_frame

    def capture(env):
        nonlocal solver
        row = trace_frame(env)
        ground = env.env.scene['jump_ground']
        if not ground_names:
            ground_names.extend(ground.primary_names)
            wheel_names.extend(env.env.scene['jump_wheel'].primary_names)
        force = ground.data.force.norm(dim=-1)
        if env.has_bar:
            bar = env.env.scene['jump_bar_touch']
            if not bar_names:
                bar_names.extend(bar.primary_names)
            force = torch.cat((force, bar.data.found > 0), dim=1)
        if solver_pairs:
            if solver is None:
                solver = SolverPairs(env)
            force = torch.cat((force,solver.frame()),dim=1)
        chunks.append(force)
        return row

    result = []
    solver = None
    with TemporaryDirectory() as tmp, patch.object(JumpEnv, 'trace_frame', capture):
        tmp = Path(tmp)
        if height:
            bars.collect_height(tmp, height, *policies, 'cuda:0')
        else:
            flat.collect_profile(tmp/'flat', 'walk_jump_walk', *policies,landing_handoff=landing_handoff)
        data = torch.stack(chunks).cpu().numpy()
        assert data.shape[1:] == (bars.WORLDS, len(ground_names)+len(bar_names)+(2*len(solver.names) if solver else 0))
        for i in range(bars.WORLDS):
            case = bars.case_id(height, i) if height else f'start{i:02}'
            ref_dir = original/'bars' if height else original/'flat/walk_jump_walk'
            if flat_reference is not None:
                ref_dir = flat_reference
            path = (tmp if height else tmp/'flat')/(case+'.npz')
            refpath = ref_dir/(case+'.npz')
            with np.load(path) as new, np.load(refpath) as old:
                t, ref, mode = new['trace'], old['trace'], new['mode']
                assert all(np.array_equal(new[k],old[k]) for k in ('qpos0','qvel0','history0')),case
                ground = data[:len(t),i,:len(ground_names)]
                limit = len(ground_names)+len(bar_names)
                hits = data[:len(t),i,len(ground_names):limit].astype(bool)
                assert np.array_equal(ground.max(1),t[:,10])
                assert np.array_equal(hits.any(1),t[:,20].astype(bool))
                # 使用各自真实轨迹定事件；逐毫秒接触与原轨迹的差异单独保存。
                aligned = t.shape == ref.shape
                row = dict(case=case, reference_sha256=sha(refpath), initial_exact=True,
                    trace_exact=bool(np.array_equal(t,ref)), mode_exact=bool(np.array_equal(mode,old['mode'])),
                    trace_max_abs_delta=float(np.abs(t-ref).max()) if aligned else None,
                    aggregate_bar_trace_exact=bool(np.array_equal(t[:,20],ref[:,20])),
                    first_bar=[], landing=None)
                extra = {}
                if solver:
                    raw_ground,raw_bar = np.split(data[:len(t),i,limit:].astype(bool),2,axis=-1)
                    checked_ground = raw_ground[:,[solver.names.index(n) for n in ground_names]]
                    row['loaded_ground_without_raw_pair_samples'] = int(((ground>1.) & ~checked_ground).sum())
                    row['raw_first_bar_ms'] = None
                    row['raw_first_bar_parts'] = []
                    raw_hits = np.flatnonzero(raw_bar.any(1))
                    if raw_hits.size:
                        row['raw_first_bar_ms'] = int(raw_hits[0])
                        row['raw_first_bar_parts'] = names_at(solver.names,raw_bar[raw_hits[0]])
                    row['sensor_vs_raw_bar_flags_exact'] = (bool(np.array_equal(raw_bar[:,[solver.names.index(n) for n in bar_names]],hits))
                        if height else not raw_bar.any().item())
                    extra.update(solver_ground=raw_ground,solver_bar=raw_bar)
                if height:
                    old_result=score(ref,old['mode'],old['distance'],height)
                    new_result=score(t,mode,new['distance'],height)
                    row.update(original_reason=old_result['reason'], replay_reason=new_result['reason'])
                hit = np.flatnonzero(hits.any(1))
                refhit = np.flatnonzero(ref[:,20])
                row['original_first_hit_ms'] = int(refhit[0]) if refhit.size else None
                if hit.size:
                    tick=int(hit[0])
                    row.update(first_hit_ms=tick, first_bar=names_at(bar_names,hits[tick]),
                        hit_after_trigger_ms=tick-10*int(np.flatnonzero(mode)[0]),
                        vertical_speed_at_hit_m_s=float(t[tick,19]),
                        wheel_centers_past_far_edge_m=(t[tick,[0,3]]-4.05).tolist())
                events = flat.events(t,mode)
                if events and events[0]['landed']:
                    event=events[0]
                    tick=round(1000*(event['takeoff_s']+event['airtime_s']))
                    window=ground[tick:tick+1000]
                    touch=window>1.
                    contacts=np.flatnonzero(touch.any(1))
                    ref_events=flat.events(ref,old['mode'])
                    row['landing']=dict(wheel_touch_ms=tick, observed_ms=len(window),
                        ref_wheel_touch_ms=round(1000*(ref_events[0]['takeoff_s']+ref_events[0]['airtime_s']))
                        if ref_events and ref_events[0]['landed'] else None,
                        first_nonwheel_after_wheel_ms=int(contacts[0]) if contacts.size else None,
                        first_nonwheel=names_at(ground_names,touch[contacts[0]]) if contacts.size else [],
                        nonwheel_any=names_at(ground_names,touch.any(0)),
                        peak_force_N={n:float(v) for n,v in zip(ground_names,window.max(0)) if v>1.},
                        contact_ms={n:int(v) for n,v in zip(ground_names,touch.sum(0)) if v})
                np.savez_compressed(output/(case+'.npz'),ground_force_N=ground,bar_found=hits,
                                    trace=t,mode=mode,**extra)
                result.append(row)
    def counts(key):
        return dict(Counter(part for row in result for part in row[key]))
    landings=[r['landing'] for r in result if r['landing']]
    summary=dict(cases=len(result), exact_traces=sum(r['trace_exact'] for r in result),
        exact_modes=sum(r['mode_exact'] for r in result),exact_bar_flags=sum(r['aggregate_bar_trace_exact'] for r in result),
        bar_hit_cases=sum(bool(r['first_bar']) for r in result),first_bar_by_part=counts('first_bar'),
        measured_landing_windows=sum(r['observed_ms']==1000 for r in landings),
        first_ground_by_part=dict(Counter(n for r in landings for n in r['first_nonwheel'])),
        any_ground_by_part=dict(Counter(n for r in landings for n in r['nonwheel_any'])))
    if solver:
        summary.update(raw_first_bar_by_part=counts('raw_first_bar_parts'),
            sensor_raw_exact_cases=sum(r['sensor_vs_raw_bar_flags_exact'] for r in result),
            loaded_ground_without_raw_pair_samples=sum(r['loaded_ground_without_raw_pair_samples'] for r in result))
    (output/'cases.json').write_text(json.dumps(dict(ground_names=ground_names,bar_names=bar_names,wheel_names=wheel_names,
        solver_body_names=solver.names if solver else [],cases=result),indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--solver-pairs',action='store_true')
    args=p.parse_args()
    protocol=json.loads((args.reference/'bars/protocol.json').read_text())
    assert protocol['jump_sha256']==sha(args.checkpoint) and protocol['walk_sha256']==sha(bars.WALK)
    for name,digest in protocol['source_sha256'].items():
        assert sha(bars.ROOT/name)==digest,name
    for case,entry in protocol['cases'].items():
        assert sha(args.reference/'bars'/(case+'.npz'))==entry['sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    walk,jump,_=bars.policies(args.checkpoint,'cuda:0')
    report=dict(checkpoint_sha256=protocol['jump_sha256'],walk_sha256=protocol['walk_sha256'],
        optimizer_updates=0,backend='mujoco_warp_gpu',dt_s=.001,script_sha256=sha(__file__))
    for label,height in [('bar05',.05),('bar10',.10),('flat',None)]:
        report[label]=collect(args.output/label,args.reference,height,(walk,jump),args.solver_pairs)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
