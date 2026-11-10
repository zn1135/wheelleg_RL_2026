"""Verify saved PACE precheck targets/PD and summarize boundary hits; no physics steps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import precheck_pace_capture as precheck


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.run/'report.json').read_text())
    with np.load(args.run/'traces.npz') as saved:
        times,q,v,torque,targets=[saved[k] for k in ('time_s','q','v','request_nm','target_rad')]
    assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(targets).all()
    curve_error=pd_error=0.
    for i,scenario in enumerate(report['scenarios']):
        center=np.array(scenario['target_center_rad'])
        expected=np.broadcast_to(center,targets[:,i].shape).copy()
        # 独立重写窗口和线性扫频公式，核对保存目标的参数与起止约束。
        segments=range(len(report['stages'])*4) if report.get('continuous_sequence') else [0]
        for segment in segments:
            stage=report['stages'][segment//4] if report.get('continuous_sequence') else next(s for s in report['stages'] if s[0]==scenario['stage'])
            axis=segment%4 if report.get('continuous_sequence') else ('LF','LR','RF','RR').index(scenario['axis'])
            local=times-30.*segment-5.;active=(local>=0)&(local<=20.)
            phase=2*np.pi*(stage[2]*local+(stage[3]-stage[2])*local**2/40.)
            ramp=np.clip(np.minimum(local,20.-local)/2.,0.,1.)
            envelope=6*ramp**5-15*ramp**4+10*ramp**3
            expected[:,axis]+=np.where(active,stage[1]*envelope*np.sin(phase),0.)
        curve_error=max(curve_error,float(abs(expected-targets[:,i]).max()))
        stop=scenario['result']['first_violation']
        before=times<stop['time_s'] if stop else np.ones(len(times),bool)
        expected_tau=np.clip(report['kp']*(targets[:,i]-q[:,i])-report['kd']*v[:,i],-report['torque_limit_nm'],report['torque_limit_nm'])
        if before.any():pd_error=max(pd_error,float(abs(expected_tau[before]-torque[before,i]).max()))
        assert not stop or np.max(abs(torque[~before,i]))==0.
        bound=max(s[1] for s in report['stages']) if report.get('continuous_sequence') else stage[1]
        assert abs(targets[:,i]-center).max()<=bound+1e-7
        assert abs(targets[0,i]-center).max()<1e-8 and abs(targets[-1,i]-center).max()<1e-8
    assert curve_error<1e-9 and pd_error<1e-4
    for source,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    stages={}
    names=['full_sequence'] if report.get('continuous_sequence') else [s[0] for s in report['stages']]
    for name in names:
        rows=[s for s in report['scenarios'] if s['stage']==name]
        stages[name]=dict(cases=len(rows),limits_passed=sum(s['result']['limits_passed'] for s in rows),
            boundary_hits=sum(s['result']['first_violation'] is not None for s in rows),
            buffer_entries=sum(s['result'].get('first_buffer_entry') is not None for s in rows),
            peak_request_nm=max(s['result']['peak_request_nm'] for s in rows),
            peak_speed_rad_s=max(s['result']['max_abs_speed_rad_s'] for s in rows),
            saturated_cases=sum(s['result']['saturated_control_steps']>0 for s in rows))
    result=dict(numeric_integrity_passed=True,target_formula_max_error_rad=curve_error,pd_max_error_nm=pd_error,
        stages=stages,new_environment_steps=0,hardware_ready=False,
        limitation='Synthetic model assumptions; not hardware acceptance')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3 if report.get('continuous_sequence') else len(report['stages']),2,figsize=(11,9),sharex=True,squeeze=False)
    plot_stages=([report['stages'][0]]*3 if report.get('continuous_sequence') else report['stages'])
    for row,(name,*_) in enumerate(plot_stages):
        for col,joint in enumerate(('LF','LR')):
            i=row if report.get('continuous_sequence') else next(i for i,s in enumerate(report['scenarios']) if s['stage']==name and s['axis']==joint
                   and s['starting_length_m']==.265 and s['spring_assumption']=='shared_model')
            scenario=report['scenarios'][i];axis=('LF','LR','RF','RR').index(joint)
            stop=scenario['result']['first_violation'];mask=times<=stop['time_s'] if stop else np.ones(len(times),bool)
            ax=axes[row,col]
            ax.plot(times,np.degrees(targets[:,i,axis]),ls='--',color='tab:blue',label='Target')
            ax.plot(times[mask],np.degrees(q[mask,i,axis]),color='tab:orange',label='Simulated')
            if stop:ax.axvline(stop['time_s'],color='red',label='Boundary stop')
            title=scenario['spring_assumption'] if report.get('continuous_sequence') else name
            ax.set_title(title+' / '+joint);ax.set_ylabel('Joint angle (deg)');ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=8)
            if row==2:ax.set_xlabel('Time (s)')
    fig.suptitle('PACE '+report.get('profile','doubled')+': synthetic265mm start')
    fig.tight_layout();fig.savefig(args.output/'targets-and-response.png',dpi=140)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
