"""Length-only response summary; never uses angle in ranking or acceptance."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base=Path('artifacts/length-response-fit');run=base/'20261110-length-only-smooth-final-01'
    proof=json.loads((run/'verification.json').read_text());assert proof['software_verified']
    assert proof['response_objective']=='length_only_recorded_torque'
    report=json.loads((run/'report.json').read_text())
    search=json.loads((base/'20261110-length-only-smooth-fit-01/progress.json').read_text())
    assert all(x['objective']=='length' for x in search)
    output=base/'20261110-length-only-smooth-result-02';output.mkdir(exist_ok=False)
    with np.load(base/'20261109-inputs-02/inputs.npz') as a:
        t=a['time_s'][:17501];real=a['real_length'][:17501,1]*1000;mq=a['mq'].astype(int);sign=a['sign'];zero=a['zero']
    with np.load(base/'20261110-nonlinear-elastic-only-01/gpu/signals.npz') as a:old=a['length_m'][:,0]*1000
    with np.load(run/'traces.npz') as a:
        from scripts.agent.real2sim.control_leg_pose_gpu import leg_pose
        new=leg_pose(a['qpos'][:,:,mq]*sign+zero)[0][:,0,1]*1000
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot(t,real,'k',lw=1.4,label='Real');axes[0].plot(t,old,label='Current model');axes[0].plot(t,new,label='Length-only fit')
    axes[0].set(ylabel='Right leg length [mm]',title='Same recorded motor torque; only length is scored')
    axes[1].plot(t,old-real,label='Current error');axes[1].plot(t,new-real,label='Length-only error');axes[1].axhline(0,color='black',lw=.6)
    axes[1].set(xlabel='Time [s]',ylabel='Simulated - real length [mm]')
    for ax in axes:ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'length-only.png',dpi=150);plt.close(fig)
    old_score=search[-1]['baseline_score'];new_score=report['trials'][0]['branches']['recorded_torque']['length_rmse_mm']
    p=Path('resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json')
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest=='e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f'
    extra=json.loads((base/'20261110-length-only-start-stop-01/gpu/report.json').read_text())
    best_extra=min(extra['trials'],key=lambda x:x['branches']['recorded_torque']['length_rmse_mm'])
    result=dict(objective='length_only_recorded_torque',angle_used_in_ranking=False,seconds=35,
        baseline_length_rmse_mm=old_score,fitted_length_rmse_mm=new_score,improvement_percent=(old_score-new_score)/old_score*100,
        fitted_length_max_error_mm=report['trials'][0]['branches']['recorded_torque']['length_max_error_mm'],
        software_verified=True,adopted_parameters_unchanged_sha256=digest,policy_training_updates=0,hardware_commands=0,
        new_gpu_environment_steps=search[-1]['new_gpu_environment_steps']+report['new_gpu_environment_steps']+extra['new_gpu_environment_steps'],
        start_stop_grid_trials=len(extra['trials']),start_stop_grid_best_length_rmse_mm=best_extra['branches']['recorded_torque']['length_rmse_mm'],
        curve='Monotone shape-preserving cubic elastic force; existing joint correction retained',
        empirical_curve='../20261110-length-torque-relation-02/average-torque-length.csv',
        limitations=['Small same-trajectory35s gain; no independent or full-hold validation',
            'A smooth mean torque/length relation mixes motion and holding states and is not a unique passive load'])
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[Path(__file__),Path('scripts/agent/real2sim/nonlinear_spring.py'),Path('scripts/agent/real2sim/refine_nonlinear_spring.py'),
        Path('scripts/agent/real2sim/report_length_response_fit.py'),Path('scripts/agent/real2sim/summarize_length_torque_relation.py')]
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    (output/'source-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
