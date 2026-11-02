"""Static geometry feasibility of PD-centred resets; simulator integration is forbidden."""
import argparse,itertools,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS,standing_motor_positions,closed_chain_residual
from ct_mjlab.fudan_ppo_walk.reset_geometry import ResetGeometry


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with patch('mujoco.mj_step',side_effect=AssertionError('No physical steps authorized')):
        solver=ResetGeometry();nominal=np.zeros(6);nominal[[0,1,3,4]]=standing_motor_positions()
        offsets=[np.zeros(6)]
        for corner in itertools.product((-.03,.03),repeat=6):
            offsets.append(np.asarray(corner))
        offsets.extend(np.random.default_rng(431).uniform(-.03,.03,(64,6)))
        for index,offset in enumerate(offsets):
            try:result=solver.solve((nominal+offset).astype(np.float32))
            except ValueError as exc:
                (a.output/'failure.json').write_text(json.dumps(dict(index=index,offset=offset.tolist(),error=str(exc),physics_steps=0),indent=2)+'\n');raise
            q=np.array([result['joint_positions'][n] for n in ACTUATED_JOINTS],dtype=np.float32)
            assert np.array_equal(q,(nominal+offset).astype(np.float32))
            # Validate the values that will actually be written into the float32 simulator buffers.
            data=solver.data;model=solver.model
            data.qpos[:]=model.qpos0
            data.qpos[:7]=np.asarray((0,0,result['height'],1,0,0,0),dtype=np.float32)
            for name,value in result['joint_positions'].items():data.qpos[solver.qaddr[name]]=np.float32(value)
            mujoco.mj_kinematics(model,data)
            residual32=closed_chain_residual(model,data)
            for name in solver.joint_names:
                joint=model.joint(name);value=data.qpos[solver.qaddr[name]]
                assert not joint.limited or joint.range[0]-1e-8<=value<=joint.range[1]+1e-8
            from ct_mjlab.fudan_ppo_walk.reset_geometry import mesh_vertices
            minimum32=min(float(mesh_vertices(model,data,g)[:,2].min()) for g in solver.collision)
            assert residual32<=1e-6 and minimum32>=0.
            assert np.array_equal(data.qpos[[solver.qaddr[n] for n in ACTUATED_JOINTS]],(nominal+offset).astype(np.float32))
            rows.append(dict(index=index,offset=offset.tolist(),height=result['height'],residual=result['closed_chain_residual_m'],
                clearance=result['minimum_ground_clearance_m'],float32_residual_m=residual32,float32_minimum_clearance_m=minimum32))
    report=dict(passed=True,samples=len(rows),heights_m=[min(r['height'] for r in rows),max(r['height'] for r in rows)],
        maximum_closure_error_m=max(r['residual'] for r in rows),zero_action_position_P_exactly_zero=True,float32_writeback_checked=True,
        maximum_float32_closure_error_m=max(r['float32_residual_m'] for r in rows),
        minimum_float32_clearance_m=min(r['float32_minimum_clearance_m'] for r in rows),
        physics_steps=0,optimizer_updates=0,rows=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
