"""已对齐两次实机落地的冻结策略输入替换，动作反算只用于离线诊断。"""
import argparse
import csv
from math import comb
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train import load_jump_policy
from .compare_landing_lead import ARCHIVE, ALIGN, jump_metrics
from .analyze_real022 import matrix, read_csv
from .probe_takeoff_work import ROOT, CHECKPOINT, CHECKPOINT_SHA, collect_profile, policies, sha, write_json

GROUPS = {'gyro':[0,1,2], 'gravity':[3,4,5], 'command':[6,7,8], 'leg_angle':[9,10,11,12],
          'leg_speed':[13,14,16,17], 'wheel_speed':[15,18], 'previous_action':list(range(19,25))}
DETAIL = dict(gyro_x=[0],gyro_y=[1],gyro_z=[2],gravity_x=[3],gravity_y=[4],left_angle=[9,10],right_angle=[11,12],left_leg_speed=[13,14],right_leg_speed=[16,17],left_wheel_speed=[15],right_wheel_speed=[18],previous_right_leg=[22,23],previous_left_leg=[19,20],previous_wheels=[21,24])
AUDIT = ROOT/'artifacts/fudan_ppo_jump/20261125-landing-lead-01'


def capture(out):
    out.mkdir(parents=True,exist_ok=False)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    walk,jump,_=policies(CHECKPOINT,'cuda:0')
    original=jump.act_inference
    frames=[]
    histories=[]
    def recorded(obs,history):
        frames.append(obs.detach().cpu().numpy().copy())
        histories.append(history.detach().cpu().numpy().copy())
        return original(obs,history)
    jump.act_inference=recorded
    summary=collect_profile(out/'probe',walk,jump,probe=True,height=.22,speed=0.)
    write_json(out/'summary.json',summary)
    np.savez_compressed(out/'actor-input.npz',obs=np.asarray(frames),history=np.asarray(histories))


def infer(policy, x):
    a,_=policy.act_inference(x[:,0], x[:,1:].reshape(len(x),125))
    return a


def right_target(action):
    # 两个右髋的PD位置目标差，正方向为伸腿。沿用实机实际位置目标限幅。
    front=(.604720554+.5*action[:,3]).clip(.664720554-3.14,.664720554+3.14)
    rear=(-.608324173+.5*action[:,4]).clip(-.667594582-3.14,-.667594582+3.14)
    return front-rear


def stats(v, select):
    return float(np.median(v[select]))


@torch.inference_mode()
def analyze(out):
    torch.set_num_threads(1)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    policy,_=load_jump_policy(CHECKPOINT,'cpu'); inputs=np.load(out/'actor-input.npz')
    sim=np.concatenate([inputs['obs'][:,:,None,:],inputs['history'].reshape(600,20,5,25)],axis=2)
    land=[];sim_errors=[]
    for j in range(20):
        with np.load(out/'probe'/f'start{j:02d}.npz') as raw:
            land.append(jump_metrics(raw)['land']);active=raw['mode'];pred=infer(policy,torch.tensor(sim[active,j])).numpy();sim_errors.append(float(abs(pred-raw['actions'][active]).max()))
    assert max(sim_errors)<1e-4
    alignment=json.loads(ALIGN.read_text());assert sha(ARCHIVE)==alignment['archive_sha256']
    rows=[]; summaries=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for trial in alignment['trials']:
            i=trial['trial'];net=read_csv(z,f'试次{i}-模型输入动作.csv');matched=np.load(AUDIT/f'real{i}.npz');seq=matched['network_sequence'];idx=np.searchsorted(net['sequence'],seq);assert np.array_equal(net['sequence'][idx],seq)
            obs=matrix(net,[f'obs_{j}' for j in range(25)]);hist=matrix(net,[f'history_{j}' for j in range(125)]).reshape(-1,5,25);x=np.concatenate([obs[:,None,:],hist],axis=1).astype('float32');action=matrix(net,[f'action_{j}' for j in range(6)])
            real_act=infer(policy,torch.from_numpy(x[idx])).numpy();error=float(abs(real_act-action[idx]).max());assert error<1e-4
            t=matched['t_ms'];lo,hi=matched['contact_ms'];contact=(t>=lo)&(t<=hi);actual=matched['actual'][:,1,None];mid=np.array(trial['touchdown_bracket_frames']).mean()/30-trial['video_offset_s']
            for offset in (0,-33,33):
                tick=np.floor((np.array(land)[None,:]+1000*(net['rel_s'][idx,None]-mid)+offset)/10).astype(int)
                s=torch.from_numpy(sim[tick,np.arange(20)]).reshape(-1,6,25)
                r=torch.from_numpy(np.repeat(x[idx,None],20,axis=1)).reshape(-1,6,25)
                def outcome(blend):return right_target(infer(policy,blend)).numpy().reshape(-1,20)
                v0=outcome(r); sim_target=outcome(s); results={}; groups=list(GROUPS);K=len(groups)
                for mask in range(1<<K):
                    blend=r.clone();cols=[c for k,g in enumerate(groups) if mask&(1<<k) for c in GROUPS[g]]
                    blend[:,:,cols]=s[:,:,cols];results[mask]=outcome(blend)
                effect={};summary={}
                for k,g in enumerate(groups):
                    value=np.zeros_like(v0)
                    for mask in range(1<<K):
                        if not mask&(1<<k):value+=(results[mask|(1<<k)]-results[mask])/(K*comb(K-1,mask.bit_count()))
                    effect[g]=value
                np.testing.assert_allclose(sum(effect.values()),sim_target-v0,atol=3e-5)
                for g,cols in dict(GROUPS,**DETAIL).items():
                    blend=r.clone();blend[:,:,cols]=s[:,:,cols];single=outcome(blend);change=single-v0
                    back=s.clone();back[:,:,cols]=r[:,:,cols];reverse=outcome(back)-sim_target
                    summary[g]=dict(single_rad=stats(change,contact), reverse_rad=stats(reverse,contact),
                        extension_fraction=float(np.mean((single-actual)[contact]>0)),
                        shapley_mean_rad=float(effect[g][contact].mean()) if g in effect else None)
                    if offset==0 and g in GROUPS:
                        for j,at in enumerate(t):rows.append(dict(trial=i,touch_ms=float(at),group=g,real_error=float(v0[j,0]-actual[j,0]),single_error=float(np.median(single[j]-actual[j])),single_delta=float(np.median(change[j])),shapley_delta=float(effect[g][j].mean())))
                combos={}
                for label,names in [('angles_tilt',['leg_angle','gravity']),('tilt_wheels',['gravity','wheel_speed']),('angles_tilt_gyro',['leg_angle','gravity','gyro']),('tilt_and_motor_speeds',['gravity','leg_speed','wheel_speed']),('all',['gyro','gravity','command','leg_angle','leg_speed','wheel_speed','previous_action'])]:
                    mask=sum(1<<groups.index(g) for g in names)
                    mixed=results[mask]-actual
                    combos[label]=dict(median_error_rad=stats(mixed,contact),extension_fraction=float(np.mean(mixed[contact]>0)),per_pd_median_rad=np.median(mixed[contact],axis=1).tolist())
                values={}
                for g,cols in GROUPS.items():values[g]={'real_obs':np.median(r[:,0,cols].numpy().reshape(-1,20,len(cols))[contact],axis=(0,1)).tolist(),'sim_obs':np.median(s[:,0,cols].numpy().reshape(-1,20,len(cols))[contact],axis=(0,1)).tolist()}
                sim_q=1.213044727+s[:,0,11].numpy().reshape(-1,20)-s[:,0,12].numpy().reshape(-1,20)
                summaries.append(dict(trial=i,sim_offset_ms=offset,matched_pd_samples=int(contact.sum()),distinct_actions=int(len(np.unique(seq[contact]))),max_replay_error=error,
                    real_error_rad=stats(v0-actual,contact),sim_error_vs_own_obs_rad=stats(sim_target-sim_q,contact),sim_target_vs_real_pose_rad=stats(sim_target-actual,contact),
                    target_total_change_mean_rad=float((sim_target-v0)[contact].mean()),effects=summary,inputs=values,combinations=combos))
    write_json(out/'attribution.json',dict(analyses=summaries,groups=GROUPS,sim_action_replay_max_error=max(sim_errors),sha256=CHECKPOINT_SHA))
    with (out/'per-tick.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('capture','analyze'));p.add_argument('out',type=Path)
    args=p.parse_args()
    if args.stage=='capture':capture(args.out)
    else:analyze(args.out)
