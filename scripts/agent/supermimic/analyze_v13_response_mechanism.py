"""Decompose wheel response and audit zero-delay dependence on action history."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.state import NOISE,unit_quat

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def decomposition(minus,plus):
    q0=unit_quat(minus[:,-2,3:7]);q1=unit_quat(plus[:,-2,3:7])
    v0=minus[:,-2,25:28];v1=plus[:,-2,25:28]
    a=quat_apply_inverse(q0,v0)[:,0];b=quat_apply_inverse(q0,v1)[:,0];c=quat_apply_inverse(q1,v1)[:,0]
    return dict(velocity_contribution_mean=float((b-a).mean()),rotation_contribution_mean=float((c-b).mean()),total_mean=float((c-a).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source_path=ROOT/'artifacts/supermimic/supermimic-v13-wheel-response-20261101-r1/response.pt'
    hist_path=ROOT/'artifacts/supermimic/supermimic-v13-wheel-history-control-20261101-r1/response.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    hist=torch.load(hist_path,map_location='cpu',weights_only=False)
    assert hist['intervention']=='previous_wheel_history_only'
    rows=[];proof=[]
    for n,filename in ((4200,'actor_004200.pt'),(5200,'final.pt')):
        path=RUN/filename;saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        world=WorldModel().eval().requires_grad_(False);world.load_state_dict(saved['world'],strict=True)
        proof.append(dict(model=n,checkpoint_sha256=sha(path),config={k:saved['config'][k] for k in ('buffer_steps','window','horizon','num_envs')}))
        for s in source['samples']:
            aa=[s[k].clone().requires_grad_(True) for k in ('action','previous','previous_previous')]
            pred=world.step(s['state'],s['parameters'],aa[0],s['context'][:,:3],s['context'],aa[1],aa[2],s['state'][:,NOISE],return_path=True)
            grad=torch.autograd.grad(pred[4][:,0].sum(),aa)
            for wheel in (2,5):
                paths=[]
                for delta in (-.1,.1):
                    a=s['action'].clone();a[:,wheel]+=delta
                    with torch.no_grad():result=world.step(s['state'],s['parameters'],a,s['context'][:,:3],s['context'],s['previous'],s['previous_previous'],s['state'][:,NOISE],return_path=True)
                    paths.append(result[-1])
                    derived=quat_apply_inverse(unit_quat(result[-1][:,-2,3:7]),result[-1][:,-2,25:28])[:,0]
                    torch.testing.assert_close(derived,result[4][:,0],atol=1e-6,rtol=1e-6)
                rows.append(dict(model=n,time_s=s['time_s'],wheel=wheel,
                    current_previous_previous2_gradient=[float(g[:,wheel].mean()) for g in grad],
                    predicted=decomposition(*paths),
                    native=decomposition(s['paths'][f'{wheel}_-0.10'],s['paths'][f'{wheel}_+0.10'])))
    history=[]
    for s in hist['samples']:
        assert (s['parameters'][:,24:26]==0).all()
        noise=torch.maximum(abs(s['true_outputs']['base']-s['true_outputs']['repeat']),abs(s['native_outputs']-s['true_outputs']['base']))
        for wheel in (2,5):
            for eps in (.05,.1):
                native=s['true_outputs'][f'{wheel}_{eps:+.2f}']-s['true_outputs'][f'{wheel}_{-eps:+.2f}']
                predicted=s['predicted_outputs'][f'{wheel}_{eps:+.2f}']-s['predicted_outputs'][f'{wheel}_{-eps:+.2f}']
                floor=noise[:,0]*5+1e-5
                history.append(dict(time_s=s['time_s'],wheel=wheel,epsilon=eps,
                    native_forward_delta_mean=float(native[:,0].mean()),native_forward_delta_max_abs=float(native[:,0].abs().max()),
                    predicted_forward_delta_mean=float(predicted[:,0].mean()),
                    native_above_noise_count=int((native[:,0].abs()>floor).sum()),
                    model_above_noise_count=int((predicted[:,0].abs()>floor).sum()),
                    noise_max=float(noise[:,0].max())))
    report=dict(models=proof,coordinate_decomposition=rows,history_negative_control=history,
        input_sha256={str(p):sha(p) for p in (source_path,hist_path)},parameter_updates=0,
        scope='Same A5200 states for A4200/A5200; not A4200 own distribution. History negative control uses separately frozen captured states. Fixed physical state/current action and zero actuator delay; previous-action FIFO perturbation should have no measurable physical effect.',
        limitations='No full training replay retained; data correlation and moving-distribution explanations remain hypotheses. Local response gradients do not establish the full actor objective gradient.')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(history_negative_control=history,coordinate_decomposition=rows),ensure_ascii=False))


if __name__=='__main__':main()
