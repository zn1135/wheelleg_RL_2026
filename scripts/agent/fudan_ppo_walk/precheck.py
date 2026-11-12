"""20-environment visible geometry/control gate, charged to the execution budget."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import mujoco
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT,RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,gpu_row,arrays
from ct_mjlab.fudan_ppo_walk.protocol import samples
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
from ct_mjlab.envs.fudan_ppo_walk import posture_reward


def initial_geometry(env,items):
    model=env.env.sim.mj_model;data=env.env.sim.data
    robot=env.robot.indexing.body_ids.cpu().numpy()
    geoms=np.flatnonzero(np.isin(model.geom_bodyid,robot) & ((model.geom_contype!=0)|(model.geom_conaffinity!=0)))
    position=data.geom_xpos.cpu().numpy();rotation=data.geom_xmat.cpu().numpy().reshape(env.num_envs,-1,3,3)
    clearance=np.full(env.num_envs,np.inf)
    for geom in geoms:
        assert model.geom_type[geom]==int(mujoco.mjtGeom.mjGEOM_MESH)
        mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);end=start+int(model.mesh_vertnum[mesh])
        # 从GPU当前几何位置重建网格最低点；不启动CPU物理仿真。
        vertices=model.mesh_vert[start:end][None]@rotation[:,geom].transpose(0,2,1)+position[:,geom,None]
        clearance=np.minimum(clearance,vertices[:,:,2].min(axis=1))
    assert np.isfinite(clearance).all()
    height=data.qpos[:,2].cpu().numpy()
    return [dict(id=item['id'],height=float(height[i]),minimum_collision_mesh_height_m=float(clearance[i]))
            for i,item in enumerate(items)]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--viewer',choices=['viser'],required=True)
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--history-version',choices=('v2','v3'),default='v3')
    p.add_argument('--training-resets',action='store_true',help='Use the actual training reset-on-done path')
    a=p.parse_args();disturbances=from_task_config();a.output.mkdir(parents=True,exist_ok=False)
    reset_contract=RESET_V2_CONTRACT if a.history_version=='v2' else RESET_CONTRACT
    started=time.time();env=None;server=None;report=dict(passed=False,kind='physical-visible-precheck',num_envs=20,reset_contract=reset_contract,optimizer_updates=0,physics_steps=0)
    report['disturbances']=disturbances.record()
    report['command_sampling']=command_config()
    report['posture_reward']=posture_reward.from_task_config()
    report['auto_reset']=a.training_resets
    report['episode_completions']=0
    try:
        if time.time()>=a.deadline:raise TimeoutError('Precheck quota reached before initialization')
        env=FudanEnv(20,seed=30001,randomized=True,reset_contract=reset_contract,disturbances=disturbances)
        fixture=a.output/'fixtures'
        # 预检用独立种子，不提前使用正式留出初态。
        items=[]
        for i,s in enumerate(samples()[:20]):
            sample=dict(s);sample['seed']=30001+i;sample['id']=f'precheck_{i}';items.append(sample)
        obs,critic=configure_samples(env,items,fixture)
        assert obs.shape==(20,25) and critic.shape==(20,141) and env.history.shape==(20,125)
        clean=env.frame().cpu().numpy();initial=initial_geometry(env,items)
        for row in initial:
            if row['minimum_collision_mesh_height_m']<-.0001:raise RuntimeError(f'Invalid reset penetration: {row}')
        np.savez_compressed(a.output/'initial_inputs.npz',clean=clean,obs=obs.cpu().numpy(),history=env.history.cpu().numpy())
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene
        server=viser.ViserServer(host='127.0.0.1',port=a.port,label='Fudan PPO physical precheck')
        scene=MjlabViserScene(server,env.env.sim.mj_model,20,sim_model=env.env.sim.model,expanded_fields=env.env.sim.expanded_fields)
        scene.show_only_selected=False;scene.create_scene_gui(camera_distance=6.,show_debug_viz_control=False)
        server.gui.add_markdown('20 environments; geometric reset / zero-action physical check; no policy trained')
        scene.update(env.env.sim.data)
        print(f'20-environment viewer: http://127.0.0.1:{server.get_port()}',flush=True)
        env.action.record_substeps=True
        rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()]
        steps=max(200,int(np.ceil(disturbances.push_interval_s/.01))+2) if disturbances.pushes else 200
        for step in range(steps):
            if time.time()>=a.deadline:raise TimeoutError('Precheck quota reached')
            tick=time.time()
            # 训练预检按真实倒地/超时复位；非有限状态检查仍在复位前执行。
            result=env.step(torch.zeros(20,6,device=env.device),auto_reset=a.training_resets,resample=False)
            report['episode_completions']+=int(result[3].sum().item())
            rows.extend(torch.stack(env.action.substep_rows).cpu().numpy());poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
            report['physics_steps']+=5
            if step%3==0:scene.update(env.env.sim.data)
            time.sleep(max(0.,.01-(time.time()-tick)))
        rows=np.asarray(rows);poses=np.asarray(poses);origins=env.env.scene.env_origins.cpu().numpy()
        for i in range(20):
            q=poses[:,i].copy();q[:,:3]-=origins[i]
            np.savez_compressed(a.output/f'precheck_{i}.npz',**arrays(rows[:,i],q))
        report.update(passed=True,initial=initial,validation_engine='gpu',
            action_delay_steps=env.action.disturbances.action_steps.cpu().tolist(),
            sensor_delay_steps=env.action.disturbances.sensor_steps.cpu().tolist(),
            push_count=env.action.disturbances.push_count.cpu().tolist(),
            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Valid reset and finite zero-action dynamics; auto_reset is recorded explicitly; not behavioral acceptance')
    except BaseException as exc:
        report['error']=repr(exc)
        if env is not None:
            np.savez_compressed(a.output/'failure_state.npz',qpos=env.env.sim.data.qpos.cpu().numpy(),qvel=env.env.sim.data.qvel.cpu().numpy())
        raise
    finally:
        report['wall_s']=time.time()-started;(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if server is not None:server.stop()
        if env is not None:env.close()

if __name__=='__main__':main()
