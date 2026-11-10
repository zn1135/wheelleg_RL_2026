"""Inspect exact-clone GPU forward stages at frozen GPU states; no integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mw

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); wp.init()
    root = args.experiment/'final-original'; report = json.loads((root/'report.json').read_text())
    profile = report['results'][0]['parameters']; recorded = np.load(root/'physics.npz')
    model = mujoco.MjModel.from_xml_path(str(root/'model.xml'))
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model); model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE; model.opt.iterations=100
    model.opt.tolerance=1e-10; model.opt.impratio=10; model.geom_contype[:]=model.geom_conaffinity[:]=0
    mv = [model.joint(n).dofadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sid = [model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    model.dof_frictionloss[mv+sv]=1e-6; model.dof_solref[mv+sv]=[.002,1.]
    model.dof_solimp[mv+sv]=[.999,.9999,.001,.5,2.]
    row=[]
    for j in range(4): row += [profile[k][j] for k in ('friction_nm','motor_dynamic_positive_nm','motor_dynamic_negative_nm','motor_velocity_scale_rad_s','damping_nms_rad')]
    for side in range(2): row += [profile[k][side] for k in ('spring_reference_force_n','spring_stiffness_n_m','spring_damping_ns_m','spring_friction_n')]
    count=48; results={}
    for seconds in (0.,2.9,3.0,3.02):
        k=round(seconds/.0002); data=mujoco.MjData(model)
        data.qpos[:]=recorded['qpos'][k,0];data.qvel[:]=recorded['qvel'][k,0];data.ctrl[:]=recorded['ctrl'][k,0]
        with wp.ScopedDevice('cuda:0'):
            gm=mw.put_model(model,batch_sizes={'dof_frictionloss':count,'dof_damping':count})
            damping=np.tile(model.dof_damping,(count,1)).astype(np.float32);damping[:,mv]=profile['damping_nms_rad']
            friction=np.tile(model.dof_frictionloss,(count,1)).astype(np.float32);friction[:,sv]=profile['spring_friction_n']
            gm.dof_damping.assign(damping);gm.dof_frictionloss.assign(friction)
            gd=mw.put_data(model,data,nworld=count,nconmax=128,njmax=256)
            ids=[wp.array(x,dtype=wp.int32) for x in (mv,sq,sv,sid)]
            wp.launch(fit.forces,dim=count,inputs=[gd.qpos,gd.qvel,wp.array(np.tile(row,(count,1)),dtype=wp.float32),*ids,gm.dof_frictionloss,gd.ctrl])
            mw.forward(gm,gd)
            assert not np.any(gd.time.numpy())
            values={name:getattr(gd,name).numpy() for name in ('qpos','qvel','M','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_smooth','qacc','solver_niter','nefc')}
            efc={name:getattr(gd.efc,name).numpy() for name in ('type','id','J','D','vel','aref','frictionloss','force','state')}
        assert np.array_equal(values['qpos'],np.broadcast_to(values['qpos'][0],values['qpos'].shape))
        assert np.array_equal(values['qvel'],np.broadcast_to(values['qvel'][0],values['qvel'].shape))
        n=int(values['nefc'][0]); assert np.all(values['nefc']==n)
        canonical={name:[] for name in efc}
        for world in range(count):
            order=sorted(range(n),key=lambda r:(int(efc['type'][world,r]),int(efc['id'][world,r]),r))
            for name in efc: canonical[name].append(efc[name][world,order])
        canonical={k:np.asarray(v) for k,v in canonical.items()}
        maximum=lambda a:float(np.max(abs(a.astype(float)-a[:1].astype(float))))
        result=dict(active_rows=n,raw_row_order_identical=bool(np.array_equal(efc['id'][:,:n],np.broadcast_to(efc['id'][0,:n],(count,n)))),
            stage_max_clone_difference={name:maximum(value) for name,value in values.items()},
            canonical_constraint_max_clone_difference={name:maximum(value) for name,value in canonical.items()})
        results[str(seconds)]=result
        np.savez_compressed(args.output/(f'state-{seconds:g}.npz'),**values,**{'efc_'+k:v for k,v in efc.items()})
        print(json.dumps(dict(seconds=seconds,**result)),flush=True)
    fit.search.write_json(args.output/'report.json',dict(results=results,physical_integration_steps=0,
        scope='Same frozen GPU state broadcast to48 worlds; row sorting is analysis only, not applied to dynamics'))


if __name__=='__main__':main()
