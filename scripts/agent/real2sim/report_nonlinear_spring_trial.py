"""Summarize bounded nonlinear-spring experiments from saved GPU traces only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base=Path('artifacts/length-response-fit')
    output=base/'20261110-nonlinear-axial-result-01';output.mkdir(exist_ok=False)
    run=base/'20261110-nonlinear-axial-final-01'
    proof=json.loads((run/'verification.json').read_text());assert proof['software_verified']
    report=json.loads((run/'report.json').read_text())
    control=base/'20261110-nonlinear-elastic-only-01/gpu'
    grid=json.loads((control/'report.json').read_text())
    inverse=json.loads((base/'20261110-nonlinear-axial-seed-01/report.json').read_text())
    with np.load('artifacts/length-response-fit/20261109-inputs-02/inputs.npz') as src:
        time=src['time_s'][:17501];real_l=src['real_length'][:17501,1]*1000;real_a=np.degrees(src['real_angle'][:17501,1])
    with np.load(control/'signals.npz') as src:
        old_l=src['length_m'][:,0]*1000;old_a=np.degrees(src['angle_rad'][:,0]);old_target=src['length_m'][:,10]*1000
    with np.load(run/'traces.npz') as src:
        from scripts.agent.real2sim.control_leg_pose_gpu import leg_pose
        with np.load(base/'20261109-inputs-02/inputs.npz') as a:mq=a['mq'].astype(int);sign=a['sign'];zero=a['zero']
        length,angle,_=leg_pose(src['qpos'][:,:,mq]*sign+zero)
    rows=[dict(family='current_linear_foundation_with_joint_load',metrics=grid['trials'][0]['branches']),
          dict(family='same_linear_curve_piecewise_control',metrics=grid['trials'][1]['branches']),
          dict(family='nonlinear_axial_only',metrics=report['trials'][0]['branches']),
          dict(family='nonlinear_elastic_with_fixed_joint_load',metrics=report['trials'][1]['branches'])]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    knots=np.array(report['trials'][0]['parameters']['spring_curve_compression_m'])
    axes[0,0].plot(knots*1000,150*knots/.09,label='Current linear foundation')
    for i,label in enumerate(('Nonlinear axial only','Nonlinear + fixed joint load')):
        p=report['trials'][i]['parameters'];axes[0,0].plot(knots*1000,p['spring_curve_force_n'],'o-',label=label)
    axes[0,0].set(xlabel='Compression [mm]',ylabel='Axial elastic force [N]',title='Candidate curves; no independent force measurement')
    for ax,actual,old,sim,label in [(axes[0,1],real_l,old_l,length[:,:,1]*1000,'Leg length [mm]'),
                                  (axes[1,0],real_a,old_a,np.degrees(angle[:,:,1]),'Leg angle [deg]')]:
        ax.plot(time,actual,'k',lw=1.4,label='Real');ax.plot(time,old,label='Current model')
        ax.plot(time,sim[:,0],label='Nonlinear axial only');ax.plot(time,sim[:,1],label='Nonlinear + fixed joint load')
        ax.set(xlabel='Time [s]',ylabel=label,title='Same recorded torque; free integration')
    ax=axes[1,1];ax.plot(time,real_l,'k',lw=1.4,label='Real');ax.plot(time,old_target,label='Current model')
    ax.plot(time,length[:,2,1]*1000,label='Nonlinear axial only');ax.plot(time,length[:,3,1]*1000,label='Nonlinear + fixed joint load')
    ax.set(xlabel='Time [s]',ylabel='Leg length [mm]',title='Same target/controller; feedback can hide plant error')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    datasets=[base/'20261110-nonlinear-axial-grid-01/report.json',
        base/'20261110-nonlinear-axial-refine-01/round-00/report.json',
        base/'20261110-nonlinear-axial-refine-01/round-01/report.json',control/'report.json',run/'report.json',
        base/'20261110-nonlinear-axial-smoke-02/report.json']
    steps=sum(json.loads(p.read_text())['new_gpu_environment_steps'] for p in datasets)
    formal=Path('resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json')
    digest=hashlib.sha256(formal.read_bytes()).hexdigest()
    assert digest=='e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f'
    result=dict(scope='Finite35-second same-data experiment, not independent gas spring calibration',
        software_verified=True,rows=rows,compression_range_m=inverse['compression_range_m'],
        inverse_axial_unexplainable_rmse_nm=inverse['trials'][0]['axial_unexplainable_component_rmse_nm'],
        new_gpu_environment_steps=steps,policy_training_updates=0,hardware_commands=0,
        adopted_training_parameters_unchanged_sha256=digest,
        findings=['No tested nonlinear curve beat the current model under the predefined combined free-response score',
            'Strict axial-only model also removes existing joint-potential compensation; this is a separate model-family test',
            'Controlled elastic-only comparison keeps all other adopted parameters fixed',
            'Good target tracking does not establish the correctness of the spring force',
            'Finite search failure does not prove all nonlinear spring models impossible'],
        limitations=['One main retraction trajectory does not uniquely identify pressure versus friction',
            'Only about28-43mm compression covered; curve held constant outside this range',
            '600N bound is diagnostic numerical scope, not a measured physical ceiling',
            'No189-second extension or independent trajectory run because35-second candidates already regressed'])
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[Path(__file__),Path('scripts/agent/real2sim/replay_length_response_gpu.py'),Path('scripts/agent/real2sim/nonlinear_spring.py'),
        Path('scripts/agent/real2sim/report_length_response_fit.py'),Path('scripts/agent/real2sim/seed_nonlinear_spring.py'),Path('scripts/agent/real2sim/refine_nonlinear_spring.py')]
    for source in sources:(output/source.name).write_bytes(source.read_bytes())
    (output/'source-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
