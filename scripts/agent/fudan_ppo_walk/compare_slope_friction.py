"""冻结扫描5700的摩擦单因素GPU对照；复用逐1ms驱动记录。"""
import argparse
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from . import record_slope_drive as drive

MODULE='scripts.agent.fudan_ppo_walk.compare_slope_friction'
ARMS={'mu05':.5,'mu08':.8,'mu12':1.2}


def worker(out,mu):
    original_terrain=drive.regional.terrain
    audit=dict(requested_friction=mu,sampled_contacts=0,maximum_contact_friction_error=0.,optimizer_updates=0)
    def terrain(*args,**kwargs):
        # 此时记录器已安装训练快照导入器，避免混入工作区训练实现。
        import mujoco
        from mjlab.managers.event_manager import RecomputeLevel
        from ct_mjlab.fudan_ppo_walk import measurement
        original_configure=measurement.configure_samples
        def configure(wrapper,samples,folder):
            result=original_configure(wrapper,samples,folder)
            sim=wrapper.env.sim;m=sim.mj_model
            bodies=wrapper.robot.indexing.body_ids.cpu().numpy()
            robot=np.flatnonzero(np.isin(m.geom_bodyid,bodies))
            terrain_ids=np.flatnonzero(m.geom_bodyid==m.body('terrain').id)
            geoms=np.union1d(robot,terrain_ids).tolist()
            # 车轮priority=1会覆盖地面；两侧一起设置才是有效轮地摩擦。
            sim.model.geom_friction[:,geoms,0]=mu
            wrapper.env._fudan_parameters['friction'].fill_(mu)
            sim.recompute_constants(RecomputeLevel.set_const)
            sim.forward();wrapper.env.scene.update(dt=0.);sim.sense()
            actual=sim.model.geom_friction.cpu().numpy().copy()
            assert np.allclose(actual[:,geoms,0],mu,rtol=0.,atol=1e-7)
            for i,sample in enumerate(samples):
                path=Path(folder)/(sample['id']+'_initial.npz')
                with np.load(path) as saved:values={k:saved[k].copy() for k in saved.files}
                values['geom_friction']=actual[i]
                np.savez_compressed(path,**values)
            # 独立保存的CPU夹具同步有效材料；CPU仅用于几何复算，不积分。
            fixture=Path(folder)/'model.mjb'
            cpu=mujoco.MjModel.from_binary_path(str(fixture))
            cpu.geom_friction[geoms,0]=mu
            mujoco.mj_saveModel(cpu,str(fixture))
            audit.update(robot_geom_ids=robot.tolist(),terrain_geom_ids=terrain_ids.tolist(),
                actual_geom_friction=actual[0,geoms].tolist(),num_envs=wrapper.num_envs,
                material_scope='All robot/terrain sliding contact, including body contact; torsion/rolling unchanged')
            old_step=sim.step;ticks=[0]
            def step():
                value=old_step();ticks[0]+=1
                if ticks[0]%100==0:
                    data=sim.wp_data;count=int(data.nacon.numpy()[0])
                    pairs=data.contact.geom.numpy()[:count]
                    mask=np.isin(pairs,terrain_ids).any(-1)&np.isin(pairs,robot).any(-1)
                    coefficients=data.contact.friction.numpy()[:count,:2][mask]
                    if len(coefficients):
                        error=float(np.abs(coefficients-mu).max())
                        assert error<1e-7,(mu,error)
                        audit['sampled_contacts']+=len(coefficients)
                        audit['maximum_contact_friction_error']=max(audit['maximum_contact_friction_error'],error)
                return value
            sim.step=step
            return result
        with patch.object(measurement,'configure_samples',configure):
            return original_terrain(*args,**kwargs)
    with patch.object(drive.regional,'terrain',terrain):drive.worker(out)
    assert audit['sampled_contacts']>0
    audit['passed']=True
    drive.frozen.write(out/'effective-friction-audit.json',audit)


def compare(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];pairs={}
    base=out/'mu05/probe/terrain/5700'
    for arm,mu in ARMS.items():
        root=out/arm;current=root/'probe/terrain/5700'
        equal=0;friction_changes=0
        for path in sorted((current/'fixtures').glob('*_initial.npz'))+[current/'initial.npz']:
            with np.load(path) as a,np.load(base/path.relative_to(current)) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:
                    if key=='geom_friction':
                        assert np.array_equal(a[key][...,1:],b[key][...,1:]),'Torsion/rolling friction changed'
                        friction_changes+=int(not np.array_equal(a[key],b[key]));continue
                    assert np.array_equal(a[key],b[key]),(arm,path.name,key)
                    equal+=1
        pairs[arm]=dict(nonfriction_arrays_bitwise_equal=equal,changed_friction_arrays=friction_changes)
        files=list((root/'drive').glob('*.npz'));assert len(files)==32
        force_error=0.
        for path in files:
            with np.load(path) as d:
                assert all(np.isfinite(d[k]).all() for k in d.files),(arm,path.name)
                force_error=max(force_error,float(np.abs(d['actual_joint_torque_Nm']-d['actual_ctrl_Nm']).max()))
        assert force_error<2e-5
        pairs[arm].update(finite_raw_cases=len(files),actual_force_ctrl_max_error_Nm=force_error)
        result=json.loads((current/'result.json').read_text())
        score=next(r for r in result['rows'] if r['id']=='pyramid_3_up_v1')
        summary=json.loads((root/'summary.json').read_text())['summaries']['pyramid_3_up_v1']['2-4s']
        assert score['complete'],'Focus route must complete for the shared time window'
        with np.load(root/'drive/pyramid_3_up_v1.npz') as d:
            t=d['time_start_s'];mask=(t>=2)&(t<4)
            torque=d['actual_joint_torque_Nm'][mask][:,[2,5]]
            estimated=d['encoder_raw'][mask,0]/2
            reference=d['encoder_reference_velocity'][mask,0]
            summary.update(wheel_actual_torque_abs_mean_Nm=np.abs(torque).mean(0).tolist(),
                encoder_error_mean_m_s=float((estimated-reference).mean()),
                encoder_error_MAE_m_s=float(np.abs(estimated-reference).mean()),
                world_forward_displacement_0_to4_m=float(d['qpos_post'][3999,0]-d['qpos_pre'][0,0]))
        rows.append(dict(arm=arm,mu=mu,**summary,route_score=score,
            terrain_complete=sum(r['complete'] for r in result['rows']),
            terrain_reached=sum(r['reached'] for r in result['rows']),
            terrain_passed=result['passed'],terrain_total=result['total']))
    def display_mean(values,axis):
        valid=np.isfinite(values);count=valid.sum(axis=axis)
        return np.divide(np.where(valid,values,0.).sum(axis=axis),count,
            out=np.full(count.shape,np.nan,dtype=float),where=count>0)
    fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
    for arm,mu in ARMS.items():
        with np.load(out/arm/'drive/pyramid_3_up_v1.npz') as d:
            t=d['time_end_s'];slip=np.linalg.norm(d['contact_tangent_velocity_world_m_s'].reshape(-1,2,3),axis=-1)
            found=d['contact_found']>0
            slip=np.where(found,slip,np.nan)
            series=[d['gpu_row_post'][:,3],display_mean(slip,axis=1),d['encoder_raw'][:,0]/2-d['encoder_reference_velocity'][:,0],
                np.abs(d['actual_joint_torque_Nm'][:,[2,5]]).mean(-1),d['gpu_row_post'][:,16]]
            for ax,y in zip(axes,series):
                # 50ms分箱只用于展示，统计/原始记录仍使用1ms。
                length=len(t)//50*50
                ax.plot(t[:length].reshape(-1,50).mean(-1),display_mean(y[:length].reshape(-1,50),axis=1),label=f'mu={mu:g}')
    axes[0].plot([0,4,4,7],[1,1,0,0],'k--',lw=1,label='command')
    for ax,title in zip(axes,['Actual vx (m/s)','Contact slip (m/s)',
                             'Encoder error (m/s)','Abs wheel torque (Nm)','Body contact (N)']):
        ax.set_ylabel(title);ax.axvspan(2,4,color='grey',alpha=.1);ax.grid(alpha=.25);ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)');fig.suptitle('Frozen scan5700 / 20 degree uphill / command 1m/s / h=0.26m')
    fig.tight_layout();fig.savefig(out/'friction-comparison.png',dpi=150);plt.close(fig)
    proof=dict(passed=True,arms=rows,initial_pairing=pairs,window_s=[2,4],
        only_intervention='Robot and terrain sliding friction, all other physical/initial/controller conditions paired',
        optimizer_updates=0,backend='mujoco_warp_gpu',training_started=False)
    drive.frozen.write(out/'comparison.json',proof)
    return proof


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','worker','report'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mu',type=float,choices=tuple(ARMS.values()))
    p.add_argument('--source',type=Path,default=drive.ROOT/'artifacts/fudan_ppo_walk/20261116-slope-contact-allowed-01')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='worker':return worker(out,a.mu)
    if a.mode=='report':return compare(out)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'comparison-source.py')
    shutil.copy2(drive.__file__,out/'recording-source.py')
    drive.frozen.write(out/'plan.json',dict(model='scan5700',arms=ARMS,source=str(a.source),
        routes_per_arm=32,focus='pyramid_3_up_v1',optimizer_updates=0,friction_scope='All robot-terrain sliding friction',
        training_started=False,requested_by='User accepted proposed fixed-friction comparison'))
    try:
        for arm,mu in ARMS.items():
            folder=out/arm;folder.mkdir()
            drive.inspect.audit(folder,5700,'scan')
            drive.run_command(folder,'record',[str(drive.PYTHON),'-u','-m',MODULE,'worker','--output',str(folder),'--mu',str(mu)])
            drive.report(folder,a.source.resolve())
            drive.frozen.write(folder/'status.json',dict(status='completed',optimizer_updates=0,finished_unix=time.time()))
        compare(out)
        drive.frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0,finished_unix=time.time()))
    except BaseException as exc:
        drive.frozen.write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
