"""冻结5200，测试下降段伸腿目标提前一拍的GPU反事实。"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .probe_takeoff_work import CHECKPOINT, CHECKPOINT_SHA, ROOT, collect_profile, policies, sha, write_json
from .analyze_real022 import read_csv, matrix
from .analyze_real_landing_power import extension, MODEL_SIGNS
from .measure_flat import events

PREVIOUS = ROOT/'artifacts/fudan_ppo_jump/20261123-height022-01/v0_h0.22'
ARCHIVE = ROOT/'artifacts/fudan_ppo_jump/20261125-full-trials-replay-02/input.zip'
ALIGN = ROOT/'artifacts/fudan_ppo_jump/20261125-video-log-tuck-01/aligned-final/result.json'
POWER = ROOT/'artifacts/fudan_ppo_jump/20261125-real-landing-height-01/final-v2'


def samples(folder, index):
    with np.load(folder/'probe'/f'start{index:02}.npz') as record:
        return {k:record[k] for k in record.files}


def jump_metrics(record):
    t = record['trace']
    event = events(t,record['mode'])[0]
    lift = round(event['takeoff_s']*1000)
    land = lift+round(event['airtime_s']*1000)
    apex = lift+int(np.argmax(t[lift:land,16]))
    dual = np.flatnonzero((t[land:land+1000,8:10]>.5).all(1))
    return dict(lift=lift, apex=apex, land=land,
                gap_cm=float(abs(t[land,6]-t[land,7])*100),
                dual_delay_ms=int(dual[0]) if len(dual) else None,
                sink_cm=float((t[land,16]-t[land:land+101,16].min())*100),
                peak_cm=event['peak_lower_wheel_m']*100,
                nonwheel_ms=int((t[land:land+1000,10]>1).sum()),stable=event['stable_1s'])


def real_alignment(out):
    out.mkdir(parents=True, exist_ok=True)
    alignment=json.loads(ALIGN.read_text()); assert sha(ARCHIVE)==alignment['archive_sha256']
    summary=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        for trial in alignment['trials']:
            i=trial['trial'];p=read_csv(archive,f'试次{i}-H7控制.csv');s=read_csv(archive,f'试次{i}-H7反馈.csv');n=read_csv(archive,f'试次{i}-模型输入动作.csv')
            land=np.array(trial['touchdown_bracket_frames'])/30-trial['video_offset_s'];mid=land.mean()
            low=trial['legs'][1]['air_min_log_s'];ids=np.flatnonzero((p['rel_s']>=low)&(p['rel_s']<=land[1]+.05))
            t=p['rel_s'][ids];q=matrix(p,[f'current_{j}' for j in range(4)])[ids]
            target=matrix(p,[f'target_{j}' for j in range(4)])[ids]
            tau=matrix(p,[f'pd_Nm_{j}' for j in range(4)])[ids]
            nearest=np.abs(s['rel_s'][:,None]-t).argmin(0); assert abs(s['rel_s'][nearest]-t).max()<.001
            feedback=matrix(s,[f'hip_feedback_Nm_{j}' for j in range(4)])[nearest]*MODEL_SIGNS
            net=matrix(n,[f'action_{j}' for j in (0,1,3,4)])
            nominal=np.array([-.604720554,.608324173,.604720554,-.608324173]);ref=np.array([-.664720554,.667594582,.664720554,-.667594582])
            inferred=(nominal+.5*net).clip(ref-3.14,ref+3.14)
            # 根据实际下达目标匹配该拍或此前30ms的推理，保留匹配失败计数。
            matched=[];errors=[]
            for at,goal in zip(t,target):
                recent=np.flatnonzero((n['rel_s']<=at)&(n['rel_s']>=at-.03)&(n['model']=='5200'))
                distance=abs(inferred[recent]-goal).max(1);idx=distance.argmin();matched.append(recent[idx]);errors.append(distance[idx])
            raw=np.load(POWER/f'trial{i}.npz');gap=np.interp(t,raw['relative_s'],raw['left_minus_right_wheel_bottom_m'])*100
            right_error=extension(target-q)[:,1]; req=extension(tau)[:,1];fb=extension(feedback)[:,1]
            both=(right_error<0)&(req<0);touch=np.flatnonzero((t>=land[0])&(t<=land[1]));j=touch[len(touch)//2]
            a=j
            while a>0 and both[a-1]:a-=1
            b=j
            while b+1<len(t) and both[b+1]:b+=1
            summary.append(dict(trial=i,touch_interval=land.tolist(),low_s=low,matched=sum(np.array(errors)<1e-5).item(),frames=len(t),
                first_sustained_target_and_request_negative_s=float(t[a]) if both[j] else None,
                last_sustained_negative_s=float(t[b]) if both[j] else None,
                contact_right_target_error_rad=float(right_error[j]),contact_right_requested_Nm=float(req[j]),contact_right_feedback_Nm=float(fb[j]),
                right_target_short_fraction_in_contact=float((right_error[touch]<0).mean()),max_target_match_error_rad=float(max(errors))))
            np.savez_compressed(out/f'real{i}.npz',t_ms=(t-mid)*1000,target=extension(target),actual=extension(q),network_target=extension(inferred[matched]),request=extension(tau),feedback=extension(feedback),gap_cm=gap,contact_ms=(land-mid)*1000,
                target_angles_rad=target,network_target_angles_rad=inferred[matched],actual_angles_rad=q,
                requested_motor_Nm=tau,reported_motor_Nm=feedback,network_sequence=n['sequence'][matched])
    write_json(out/'real.json',summary);print(summary,flush=True)


def landing_modifier(reference):
    """只提前原基线伸展向的增量，共同转动和轮动作来自当前闭环策略。"""
    deltas=[];source_apex=[]
    for idx in range(20):
        data=samples(reference,idx);m=jump_metrics(data);goals=extension(data['target'][::10]);change=np.zeros_like(goals)
        # 只读取首次触地前的后继策略拍；不能把承重后的目标提前搬到空中。
        for tick in range((m['apex']+9)//10,(m['land']-1)//10):
            change[tick]=np.maximum(goals[tick+1]-goals[tick],0.)
        deltas.append(change);source_apex.append((m['apex']+9)//10)
    delta=torch.tensor(np.array(deltas),device='cuda:0');apex=torch.tensor(source_apex,device='cuda:0')
    first=torch.full_like(apex,-1);closed=torch.zeros_like(apex,dtype=torch.bool);world=torch.arange(20,device='cuda:0');log=[]
    def apply(env, action, active, tick):
        if tick<200 or not env.substeps:return action
        previous=torch.stack(env.substeps);touch=(previous[:,:,8:10]>.5).any(dim=(0,2));air=~touch
        closed.logical_or_((first>=0)&touch)
        descending=(previous[-1,:,19]<0)&(previous[-1,:,6:8].amin(1)>.01)
        begin=(first<0)&active&air&descending
        first[begin]=tick
        gate=active&air&descending&~closed&(first>=0)
        inc=delta[world,(apex+tick-first).clamp(0,delta.shape[1]-1)]*gate[:,None]
        # 只改两轴差值，按单轴可用位置指令余量限制，避免截断改变摆角。
        legs=[0,1,3,4];a=env.action;scale=a.cfg.leg_position_scale
        center=a._reference_pos[:,legs];nom=a._nominal_pos[:,legs];current=(nom+scale*action[:,legs]).clamp(center-3.14,center+3.14)
        direction=action.new_tensor([-1.,1.,1.,-1.]);room=3.14-direction*(current-center)
        inc=torch.minimum(inc,2*room.reshape(20,2,2).amin(2).clamp_min(0))
        shift=inc.repeat_interleave(2,dim=1)*direction*.5
        changed=action.clone();changed[:,legs]=torch.where(shift!=0,(current+shift-nom)/scale,action[:,legs]);log.append((tick,inc.cpu().numpy()))
        return changed
    return apply,log


def collect(group, out):
    folder=out/group;folder.mkdir(parents=True);walk,jump,saved=policies(CHECKPOINT,'cuda:0');assert saved['iterations']==5200 and sha(CHECKPOINT)==CHECKPOINT_SHA
    before=[{k:v.cpu().clone() for k,v in p.state_dict().items()} for p in (walk,jump)]
    modifier,log=(landing_modifier(out/'baseline') if group=='lead10' else (None,[]))
    summary=collect_profile(folder/'probe',walk,jump,probe=True,height=.22,speed=0.,modify_action=modifier)
    assert all(torch.equal(v.cpu(),before[j][k]) for j,p in enumerate((walk,jump)) for k,v in p.state_dict().items())
    if log:np.savez_compressed(folder/'intervention.npz',ticks=[x[0] for x in log],added_diff_rad=np.stack([x[1] for x in log]))
    write_json(folder/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('real','baseline','lead10','repeat'));parser.add_argument('out',type=Path);args=parser.parse_args()
    if args.stage=='real':real_alignment(args.out)
    else:collect(args.stage,args.out)
