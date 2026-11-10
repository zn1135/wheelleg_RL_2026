"""Verify canonicalization is a within-world row permutation, preserving all equations and states."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mw

from scripts.agent.real2sim import fit_leg_repaired_loop as fit
from scripts.agent.real2sim import ordered_leg_constraints as ordered


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);wp.init()
    source=args.experiment/'final-original'; raw=np.load(source/'physics.npz')
    model=mujoco.MjModel.from_xml_path(str(source/'model.xml'))
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model);model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;model.geom_contype[:]=model.geom_conaffinity[:]=0
    profile=json.loads((source/'report.json').read_text())['results'][0]['parameters']
    mv=[model.joint(n).dofadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    model.dof_frictionloss[mv+sv]=1e-6;model.dof_solref[mv+sv]=[.002,1.]
    model.dof_solimp[mv+sv]=[.999,.9999,.001,.5,2.]
    model.dof_damping[mv]=profile['damping_nms_rad']
    times=np.tile(np.array([0.,1.,2.9,3.,3.02,3.1]),8);ix=np.rint(times/.0002).astype(int)
    q=raw['qpos'][ix,0];v=raw['qvel'][ix,0];ctrl=raw['ctrl'][ix,0];count=len(ix)
    assert len(np.unique(q,axis=0))>1
    with wp.ScopedDevice('cuda:0'):
        gm=mw.put_model(model,batch_sizes={'dof_frictionloss':count})
        gm.dof_frictionloss.assign(raw['friction_capacity'][ix,0])
        gd=mw.put_data(model,mujoco.MjData(model),nworld=count,nconmax=128,njmax=256)
        gd.qpos.assign(q);gd.qvel.assign(v);gd.ctrl.assign(ctrl)
        mw.step1(gm,gd)
        names=['type','id','J','D','pos','margin','vel','aref','frictionloss']
        before={name:getattr(gd.efc,name).numpy() for name in names};n=gd.nefc.numpy()
        ordered.apply(gd,ordered.buffers(gm,gd))
        after={name:getattr(gd.efc,name).numpy() for name in names}
        assert np.array_equal(gd.qpos.numpy(),q) and np.array_equal(gd.qvel.numpy(),v)
        assert np.array_equal(gd.ctrl.numpy(),ctrl) and not np.any(gd.time.numpy())
        assert not np.any(gd.efc.jtdaj_nblock.numpy())
    errors=[];changed=0
    for w in range(count):
        length=int(n[w]);order=sorted(range(length),key=lambda r:(int(before['type'][w,r]),int(before['id'][w,r]),r))
        changed+=order!=list(range(length))
        for name in names: assert np.array_equal(after[name][w,:length],before[name][w,order]),name
        j=before['J'][w,:length].astype(float);d=before['D'][w,:length].astype(float)
        a=after['J'][w,:length].astype(float);b=after['D'][w,:length].astype(float)
        h=j.T@(d[:,None]*j);g=a.T@(b[:,None]*a)
        errors.append(float(abs(h-g).max()/max(1.,abs(h).max())))
    assert max(errors)<1e-12
    result=dict(equation_permutation_passed=True,worlds=count,distinct_input_states=len(np.unique(q,axis=0)),
        worlds_with_changed_constraint_order=changed,all_active_equation_fields_bitwise_preserved=names,
        positions_velocities_controls_bitwise_preserved=True,max_relative_normal_matrix_error=max(errors),
        physical_integration_steps=0,source_sha256=fit.audit.digest(Path(ordered.__file__)),
        no_global_library_modifications=True)
    fit.search.write_json(args.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
