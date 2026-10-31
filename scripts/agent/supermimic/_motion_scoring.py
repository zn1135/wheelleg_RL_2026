"""V12评分的共同预测与聚合过程；指标由各版本的 motion_arrays 提供。"""
import torch
from ct_mjlab.supermimic.motion import physical47_from_full64
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic import score_supermimic_physical_prediction as core

@torch.no_grad()
def predict(world,w,teacher,batch=64):
    truth,mask=w['truth'],w['mask'];names=('path','state','features','reward','failure','frame','continuation');all_values={k:[] for k in names}
    for first in range(0,len(w['plan']),batch):
        last=min(first+batch,len(w['plan']));state=truth['state'][first:last,0].clone();steps={k:[] for k in names}
        command=truth['history'][first:last,0,-19:-16].clone();previous=truth['previous'][first:last,0];previous_previous=truth['previous_previous'][first:last,0]
        parameters=truth['parameters'][first:last,0];context=truth['context'][first:last,0]
        for t in range(24):
            valid=mask[first:last,t,None]
            if teacher:state=torch.where(valid,truth['state'][first:last,t],state)
            action=torch.where(valid,truth['action'][first:last,t],previous)
            parameters=torch.where(valid,truth['parameters'][first:last,t],parameters);context=torch.where(valid,truth['context'][first:last,t],context)
            previous=torch.where(valid,truth['previous'][first:last,t],previous);previous_previous=torch.where(valid,truth['previous_previous'][first:last,t],previous_previous)
            next_command=torch.where(valid,truth['next_frame'][first:last,t,6:9],command)
            noise=torch.where(valid,truth['next_state'][first:last,t,NOISE],state[...,NOISE])
            state,frame,reward,failure,features,path=world.step(state,parameters,action,next_command,context,previous,previous_previous,noise,return_path=True)
            for name,value in zip(names,(path,state,features,reward,failure,frame,world.adapter.continuation(path))):steps[name].append(value.cpu())
            command,previous_previous,previous=next_command,previous,action
        for name in names:all_values[name].append(torch.stack(steps[name],1))
    return {name:torch.cat(values) for name,values in all_values.items()}


def summarize(pred,w,motion_arrays):
    arrays=motion_arrays(pred,w['truth']);mask=w['mask'];lengths=mask.sum(-1);plan=w['plan'];groups={'all':torch.ones(len(plan),dtype=torch.bool)}
    for field in ('case','phase','context_group'):
        for value in sorted({p[field] for p in plan}):groups[field+'/'+value]=torch.tensor([p[field]==value for p in plan])
    counts=(~torch.isfinite(physical47_from_full64(pred['path']))).flatten(2).sum(-1)
    def at(h,selected):
        valid=mask[selected,h-1];metrics={}
        for name,(p,t) in arrays.items():
            pp,tt=p[selected,h-1],t[selected,h-1]
            expanded=valid if pp.ndim==1 else valid[:,None].expand_as(pp)
            metrics[name]=core.stats(pp.flatten(),tt.flatten(),expanded.flatten())
        return dict(planned_windows=int(selected.sum()),complete_windows=int((lengths[selected]>=h).sum()),
            valid_positions=int(mask[selected,:h].sum()),endpoint_seconds=h*.01,LINK_feature_seconds=h*.01-.002,
            retained_motion_nonfinite_values=int(torch.where(mask[selected,:h],counts[selected,:h],0).sum()),metrics=metrics)
    return dict(groups={name:{str(h):at(h,selected) for h in (1,8,24)} for name,selected in groups.items()},
        curve_24_policy_steps=[at(h,groups['all']) for h in range(1,25)],
        aggregation='Each joint/vector group pools scalar components; MAE retains rad, m, rad/s or m/s. No wheel-position/force/torque metric.')
