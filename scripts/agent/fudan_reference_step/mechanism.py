"""2ms contact/geometry probe for frozen Fudan ONNX only; never trains.

Reuses replay.execute. GPU force data are kept per contact, without reducing
face/corner contacts into one normal. CPU work is kinematics/distance only.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from .replay import execute, write, sha, XML_REL, ONNX_REL, mujoco_rotation

SOURCE=Path('artifacts/fudan_reference_step/20261115-reverse-01')
CASES=['reverse_v0.50_a-10','reverse_v0.30_a-10','reverse_v1.00_a-20','reverse_v0.50_a-20']


class ContactRecorder:
    def __init__(self,model,n,wbids):
        self.model,self.n,self.wbids=model,n,wbids
        self.rows={};self.max_contacts=16

    def record(self,gd,pre,t,pair,world,forces,pos,frame,motors,applied):
        # All solver contacts are stored. 16 is checked, never silently truncated.
        N=self.n;K=self.max_contacts
        points=np.zeros((N,2,K,3));normals=points.copy();fworld=points.copy()
        mu=np.zeros((N,2,K));dist=mu.copy();valid=np.zeros((N,2,K),bool)
        gids=np.full((N,2,K,2),-1,np.int32);counts=np.zeros((N,2),int)
        friction=gd.contact.friction.numpy()[:len(pair)]
        distances=gd.contact.dist.numpy()[:len(pair)]
        for j,(a,b) in enumerate(pair):
            ba,bb=self.model.geom_bodyid[[a,b]]
            if ba and bb:continue
            body=ba or bb
            if body not in self.wbids:continue
            w=int(world[j]);k=int(np.where(self.wbids==body)[0][0]);s=counts[w,k];counts[w,k]+=1
            if s>=K:raise RuntimeError('Detailed wheel contact capacity exceeded')
            sign=-1 if ba else 1
            points[w,k,s]=pos[j];normals[w,k,s]=sign*frame[j].reshape(3,3)[0]
            fworld[w,k,s]=sign*forces[j];mu[w,k,s]=friction[j,0];dist[w,k,s]=distances[j]
            valid[w,k,s]=True;gids[w,k,s]=[a,b]
        row=dict(solver_time=np.full(N,t),solver_qpos=pre['qpos'].copy(),solver_qvel=pre['qvel'].copy(),
            contact_pos=points,contact_normal=normals,contact_force=fworld,contact_mu=mu,
            contact_dist=dist,contact_valid=valid,contact_geom=gids,
            actual_actuator_force=gd.actuator_force.numpy()[:,motors].copy(),applied_actions=applied.copy())
        for k,v in row.items():self.rows.setdefault(k,[]).append(v)

    def save(self,path,i):
        np.savez_compressed(path,**{k:np.asarray(v)[:,i] for k,v in self.rows.items()})


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(SOURCE/'source',out/'source')
    shutil.copy2(SOURCE/'source-manifest.json',out/'source-manifest.json')
    manifest=json.loads((out/'source-manifest.json').read_text())
    for name,entry in manifest['files'].items():assert sha(out/'source'/name)==entry['sha256']
    specs=json.loads((SOURCE/'runs/matched200/metadata.json').read_text())['specs']
    cases=[next(s for s in specs if s['id']==name) for name in CASES]
    write(out/'cases.json',cases)
    write(out/'protocol.json',dict(physics_backend='MuJoCo Warp GPU',physics_dt_s=.002,
        policy_dt_s=.004,optimizer_updates=0,source=str(SOURCE),selected_cases=CASES,
        case_selection='Representative plus two preselected successes (.3/-10 and 1/-20), matched-speed failed .5/-20',
        contact_phase='solver_time/solver_qpos match forces and derived positions; legacy time is postintegration',
        delay='Requested15ms: ZOH ceil to16ms at2ms integration, both action and sensor; no interpolation',
        max_batch=4,source_onnx_sha256=sha(out/'source'/ONNX_REL),cpu_integration_steps=0))


def run(out,arm,deadline):
    cases=json.loads((out/'cases.json').read_text());kwargs={}
    if arm!='baseline':cases=cases[:3]
    if arm=='torque39':kwargs['wheel_cap']=3.9
    elif arm in ('delay15','delay15_zoh'):kwargs['delay_ms']=15.
    elif arm=='delay14':kwargs['delay_ms']=14.
    elif arm=='friction08':kwargs['friction']=.8
    elif arm=='policy10':kwargs['policy_dt']=.010
    elif arm=='freeze':
        base=json.loads((out/'analysis/phases.json').read_text())
        for c in cases:c['freeze_leg_after_s']=base[c['id']]['first_edge_s']-.3
    completed=execute(out,arm,cases,height=.2,width=1.148,seconds=10.,deadline=deadline,
        probe=True,**kwargs)
    if not completed:raise SystemExit('Incomplete GPU replay; partial traces preserved; stop the sequence')


def load(path):
    with np.load(path) as d:return {k:d[k] for k in d.files}


def contact_quantities(p):
    n=p['contact_normal'];f=p['contact_force'];fn=(n*f).sum(-1)
    ft=np.linalg.norm(f-fn[...,None]*n,axis=-1)
    loaded=p['contact_valid']&(fn>1)
    x=p['contact_pos'][...,0];z=p['contact_pos'][...,2]
    # Classify by both point and normal; rounded convex wheel can touch the corner.
    loc=np.zeros(fn.shape,np.int8)
    loc[loaded&(z<.01)]=1
    loc[loaded&(abs(x)<.010)&(abs(z-.2)<.012)]=3
    loc[loaded&(n[...,0]<-.995)&(abs(n[...,2])<.005)&(z>.01)&(z<.203)]=2
    loc[loaded&(n[...,2]>.995)&(abs(n[...,0])<.005)&(x>=-.006)&(abs(z-.2)<.010)]=4
    loc[loaded&(loc==0)]=5
    return fn,ft,loaded,loc


def phases(out):
    result={}
    for path in sorted((out/'runs/baseline').glob('*-probe.npz')):
        name=path.name.removesuffix('-probe.npz');p=load(path);t=load(path.with_name(name+'.npz'))
        fn,ft,loaded,loc=contact_quantities(p);ts=p['solver_time'];w=t['wheel_pos']
        edge=loaded&np.isin(loc,[2,3,4]);hits=np.where(edge.any((1,2))&(ts>1))[0]
        first=int(hits[0]) if len(hits) else None
        if first is None:raise RuntimeError('No edge contact '+name)
        lift=[];top=[]
        for k in range(2):
            ids=np.where((ts>=ts[first]-.15)&(w[:,k,2]>.075))[0];lift.append(int(ids[0]) if len(ids) else None)
            ids=np.where((ts>=ts[first])&(w[:,k,0]>0)&(w[:,k,2]>.24)&((loc[:,k]==4)&loaded[:,k]).any(-1))[0]
            top.append(int(ids[0]) if len(ids) else None)
        first_side=int(np.nanargmin([x if x is not None else np.inf for x in top]))
        second=1-first_side
        events=[('接近',max(0,first-150)),('首次触沿',first)]
        if lift[first_side] is not None:events.append(('抬第一轮',lift[first_side]))
        if top[first_side] is not None:events.append(('第一轮上台',top[first_side]))
        if lift[second] is not None:events.append(('抬第二轮',lift[second]))
        both=(w[:,:,0]>0).all(-1)&(w[:,:,2]>.24).all(-1)&((loc==4)&loaded).any(-1).all(-1)&(ts>=ts[first])
        bi=np.where(both)[0]
        if len(bi):events.append(('双轮上台',int(bi[0])))
        result[name]=dict(first_edge_s=float(ts[first]),first_side=['左','右'][first_side],
            lift_times_s=[None if x is None else float(ts[x]) for x in lift],
            top_times_s=[None if x is None else float(ts[x]) for x in top],
            events=[dict(phase=s,index=int(i),time_s=float(ts[i])) for s,i in events])
    (out/'analysis').mkdir(exist_ok=True);write(out/'analysis/phases.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['prepare','run','phases','geometry','report','video'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--arm',default='baseline')
    parser.add_argument('--deadline',type=float);a=parser.parse_args()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='run':
        if a.deadline is None:parser.error('GPU replay requires --deadline')
        run(a.output,a.arm,a.deadline)
    elif a.mode=='phases':phases(a.output)
    elif a.mode=='geometry':geometry(a.output)
    elif a.mode=='report':
        from .mechanism_report import report
        report(a.output)
    elif a.mode=='video':
        if a.deadline is None:parser.error('EGL GPU rendering requires --deadline')
        video(a.output,a.deadline)



def edge_model(out):
    from .replay import scene
    spec=mujoco.MjSpec.from_file(str(scene(out,'single',.2,1.148)))
    spec.worldbody.add_geom(name='measurement_edge',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(.000001,.574,0.),pos=(0,0,.2),quat=(math.sqrt(.5),math.sqrt(.5),0,0),contype=0,conaffinity=0)
    return spec.compile()


def mesh_points(m,d,g):
    mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];b=a+m.mesh_vertnum[mid]
    return m.mesh_vert[a:b]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]


def overhang(m,d,wids,direction):
    from scipy.spatial import ConvexHull
    wheel_extents=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in wids or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            wheel_extents.append(float((direction*mesh_points(m,d,g)[:,0]).max()))
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis=d.geom_xmat[g].reshape(3,3)[:,2]
            radius,half_length=m.geom_size[g,:2]
            support=radius*np.sqrt(max(0.,1-axis[0]**2))+half_length*abs(axis[0])
            wheel_extents.append(float(direction*d.geom_xpos[g,0]+support))
        else:raise ValueError('Unsupported wheel collision shape')
    wheel_front=max(wheel_extents);rows=[]
    for g in range(m.ngeom):
        bid=int(m.geom_bodyid[g])
        if bid==0 or bid in wids or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
        if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Unsupported nonwheel geometry')
        v=mesh_points(m,d,g);below=v[v[:,2]<.2]
        if not len(below):continue
        # Clip convex collision hull against z=.2; include intersections, not vertices only.
        hull=ConvexHull(v);tri=hull.simplices;edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[0,2]]]),axis=1),axis=0)
        a,b=v[edges[:,0]],v[edges[:,1]];cross=(a[:,2]-.2)*(b[:,2]-.2)<0
        a,b=a[cross],b[cross]
        if len(a):below=np.r_[below,a+((.2-a[:,2])/(b[:,2]-a[:,2]))[:,None]*(b-a)]
        i=int(np.argmax(direction*below[:,0]));point=below[i]
        rows.append(dict(body=m.body(bid).name,geom=g,overhang_m=float(direction*point[0]-wheel_front),
            foremost_point_m=point.tolist(),component_min_z_m=float(v[:,2].min())))
    best=max(rows,key=lambda r:r['overhang_m']) if rows else None
    return dict(direction=direction,wheel_front_m=wheel_front,limiting_component=best,components=rows,
        root_height_m=float(d.qpos[2]),wheel_z_m=d.xpos[wids,2].tolist())


def geometry(out):
    m=edge_model(out);d=mujoco.MjData(m);wids=[m.body(x).id for x in ('l_wheel_Link','r_wheel_Link')]
    base=m.body('base_Link_del').id;edge=m.geom('measurement_edge').id
    gids=[g for g in range(m.ngeom) if m.geom_bodyid[g] and m.geom_bodyid[g] not in wids and (m.geom_contype[g] or m.geom_conaffinity[g])]
    names=[m.body(m.geom_bodyid[g]).name for g in gids]
    hipids=[m.joint(x).id for x in ('lf0_Joint','rf0_Joint')]
    phases_=json.loads((out/'analysis/phases.json').read_text());audit={}
    from .replay import module
    ref=module(out);refargs=ref.build_arg_parser().parse_args([])
    aj=[m.joint(n) for n in ('lf0_Joint','l_wheel_Joint','rf0_Joint','r_wheel_Joint','l20_Joint','r20_Joint')]
    qa=[int(j.qposadr[0]) for j in aj];va=[int(j.dofadr[0]) for j in aj]
    geo=out/'geometry-precise';geo.mkdir(exist_ok=False)
    for path in sorted((out/'runs/baseline').glob('*-probe.npz')):
        name=path.name.removesuffix('-probe.npz');p=load(path);t=load(path.with_name(name+'.npz'))
        rows=[];dist=[];vq=[];max_pair_error=0.
        for i,q in enumerate(p['solver_qpos']):
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            R=d.xmat[base].reshape(3,3);hips=d.xanchor[hipids];vec=(d.xpos[wids]-hips)@R
            com=(d.xipos*m.body_mass[:,None]).sum(0)/m.body_mass.sum()
            distances=[mujoco.mj_geomDistance(m,d,g,edge,10.,None)+.000001 for g in gids]
            rows.append(np.r_[com,vec.ravel()]);dist.append(distances)
            aq=q[qa]
            l=ref.solve_phi3_left(ref.DEFAULT_OFFSET+aq[4],aq[0],refargs.l1,refargs.l2)
            r=ref.solve_phi3_right(ref.DEFAULT_OFFSET-aq[5],-aq[2],refargs.l1,refargs.l2)
            vq.append([aq[0],l,aq[1],aq[2],r,aq[3]])
            max_pair_error=max(max_pair_error,float(np.abs(d.xpos[wids]-t['wheel_pos'][i]).max()))
        rows=np.asarray(rows);dist=np.asarray(dist)
        np.savez_compressed(geo/(name+'.npz'),time=p['solver_time'],com_world=rows[:,:3],hip_wheel_body=rows[:,3:].reshape(-1,2,3),
            edge_distance=dist,edge_distance_body_names=np.array(names),virtual_q_actual=np.asarray(vq),motor_speed=p['solver_qvel'][:,va])
        audit[name]=dict(kinematic_wheel_GPU_max_error_m=max_pair_error,minimum_edge_distance_m=dict(zip(names,dist.min(0).tolist())))
        print('GEOMETRY '+name,flush=True)
    # B: nominal physical parameters, actual command-conditioned settled stance.
    # A virtual PD zero-action reference is not the robot's standing pose.
    from ct_mjlab.envs.chuanliantui.model import load_model
    local=load_model();ld=mujoco.MjData(local);lw=[local.body(x).id for x in ('lfwheel','rfwheel')]
    static=[]
    for side in ('forward','reverse'):
        if side=='reverse':
            p=load(out/'runs/baseline/reverse_v0.50_a-10-probe.npz');idx=int(np.argmin(abs(p['solver_time']-.8)))
            d.qpos[:]=p['solver_qpos'][idx];source=out/'runs/baseline/reverse_v0.50_a-10-probe.npz'
        else:
            source=SOURCE/'runs/matched200/forward_v0.50_a-10.npz';old=load(source)
            idx=int(np.argmin(abs(old['time']-.8)));d.qpos[:]=old['qpos'][idx]
        mujoco.mj_kinematics(m,d)
        r=overhang(m,d,wids,1);r.update(robot='fudan',pose='settled_replay_command',height_command_m=.27,
            time_s=.8,source=str(source),source_sha256=sha(source),body_direction=side)
        static.append(r)
    p=load(out/'runs/baseline/reverse_v0.50_a-10-probe.npz')
    localroot=Path('artifacts/fudan_ppo_step/20261115-reverse-diagnostic-01/matrix')
    # Archived matrix uses same nq/order. Kinematic positions are relative to its terrain origin.
    for direction,side in ((1,'forward'),(-1,'reverse')):
        f=localroot/'traces'/f'{side}_h0.200_v0.30_a+0.npz';lt=load(f);idx=79
        ld.qpos[:]=lt['qpos'][idx];mujoco.mj_kinematics(local,ld)
        r=overhang(local,ld,lw,1);r.update(robot='local',pose='settled_replay_command',height_command_m=.30,
            time_s=.8,source=str(f),source_sha256=sha(f),body_direction=side)
        # World travel is +X in both actual archived poses.
        static.append(r)
    idx=phases_['reverse_v0.50_a-10']['events'][1]['index'];d.qpos[:]=p['solver_qpos'][idx];mujoco.mj_kinematics(m,d)
    dynamic=overhang(m,d,wids,1);dynamic.update(time_s=float(p['solver_time'][idx]),pose='first_edge_actual_dynamic')
    write(geo/'audit.json',dict(rows=audit,static=static,dynamic=dynamic,cpu_integration_steps=0,
        edge_distance_definition='MuJoCo convex collision geometry to finite upper edge x=0,z=.2,y in[-.574,.574]; query-only capsule radius1um, distance corrected by+1um',
        overhang_definition='max travel projection of nonwheel collision hull clipped at z<.2 minus exact foremost wheel collision point (mesh support or finite cylinder including width/yaw); positive means nonwheel leads wheel front',
        nominal_note='Standing means nominal physical parameters and .8s settled pose under the actual replay height command (.27 Fudan/.30 local). Virtual PD zero-action reference is not a standing pose; body orientation is retained, no CPU integration.'))



def video(out,deadline):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    target=out/'media-verified';target.mkdir(exist_ok=False)
    name='reverse_v0.50_a-10';p=load(out/'runs/baseline'/(name+'-probe.npz'));t=load(out/'runs/baseline'/(name+'.npz'))
    m=mujoco.MjModel.from_binary_path(str(out/'runs/baseline/model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=800;m.vis.global_.offheight=600
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name=='l_wheel_Link':m.geom_rgba[g]=[.1,.7,.9,1.]
        elif m.body(m.geom_bodyid[g]).name=='r_wheel_Link':m.geom_rgba[g]=[.9,.2,.5,1.]
    renderer=mujoco.Renderer(m,height=600,width=800);cam=mujoco.MjvCamera();cam.distance=1.30;cam.elevation=-12
    fps=50;nframes=round(len(p['solver_time'])*.002*fps);path=target/'reverse-contact-color-1x.mp4'
    fn,ft,loaded,loc=contact_quantities(p);labels=['none','ground','face','corner','top','other']
    def frame(i):
        d.qpos[:]=p['solver_qpos'][i];mujoco.mj_kinematics(m,d);parts=[]
        for az,title in ((90,'View from +Y'),(270,'View from -Y')):
            cam.azimuth=az;cam.lookat[:]=[d.qpos[0]+.02,d.qpos[1],d.qpos[2]+.01]
            renderer.update_scene(d,camera=cam)
            for k in range(2):
                for j in np.where(loaded[i,k])[0]:
                    color=np.array([.1,1.,.3,1.]) if k==0 else np.array([1.,.1,.9,1.])
                    s=renderer.scene;g=s.geoms[s.ngeom];s.ngeom+=1
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004]*3),p['contact_pos'][i,k,j],np.eye(3).ravel(),color)
            im=Image.fromarray(renderer.render());pen=ImageDraw.Draw(im)
            pen.rectangle((0,0,800,105),fill='#17202b')
            pen.text((8,6),f'{title} | Fudan Upstairs ONNX | GPU | 1x | t={p["solver_time"][i]:.3f}s',fill='white')
            pen.text((8,25),'200mm step | green=L contact | magenta=R contact | cameras follow robot',fill='white')
            for k in range(2):
                ix=np.where(loaded[i,k])[0];ss=','.join(sorted(set(labels[int(x)] for x in loc[i,k,ix]))) or 'air'
                pen.text((8,44+k*18),f'{"L" if k==0 else "R"}: {ss} | Fn={fn[i,k,ix].sum():.1f}N Ft(sum)={ft[i,k,ix].sum():.1f}N | z={t["wheel_pos"][i,k,2]:.3f}m',fill='white')
            pen.text((8,81),f'root z={d.qpos[2]:.3f}m | body={t["base_force"][i]:.1f}N leg={t["leg_force"][i]:.1f}N',fill='white')
            parts.append(im)
        panel=Image.new('RGB',(1600,600));panel.paste(parts[0],(0,0));panel.paste(parts[1],(800,0));return panel
    writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None)
    try:
        for j in range(nframes):
            if time.time()>=deadline:raise TimeoutError('Video wall deadline; partial video kept')
            writer.append_data(np.asarray(frame(min(j*10,len(p['solver_time'])-1))))
        ev=json.loads((out/'analysis/phases.json').read_text())[name]['events']
        for j,e in enumerate(ev):frame(e['index']).save(target/f'phase-{j:02d}.png')
    finally:writer.close();renderer.close()
    reader=imageio.get_reader(path);count=0
    for im in reader:count+=1
    info=reader.get_meta_data();reader.close()
    assert count==nframes and info['fps']==fps
    write(target/'video.json',dict(frames=count,fps=fps,duration_s=count/fps,sha256=sha(path),
        full_decode_verified=True,deadline_unix_s=deadline,cpu_integration_steps=0,camera_note='Two opposing close side views; true solver poses/contact dots; no slow motion'))
    print('VIDEO '+str(path),flush=True)



if __name__=='__main__':main()
