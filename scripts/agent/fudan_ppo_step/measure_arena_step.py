"""Original-field grid/ray measurements, wheel mesh dimensions and static bounds; no mj_step."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.ndimage import label
import torch
from sim2sim.rmuc2027_arena import ARENA, CHECKPOINT, model_cache


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rules',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cache=model_cache(saved,ARENA)
    if not cache.exists():raise FileNotFoundError('Original arena cache missing; do not silently compile or simplify')
    model=mujoco.MjModel.from_binary_path(str(cache));data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    field=[g for g in range(model.ngeom) if model.geom(g).name.startswith('arena/')]
    field_set=set(field)
    robot=[g for g in range(model.ngeom) if g not in field_set]
    model.geom_rgba[:,3]=0.;model.geom_rgba[field,3]=1.
    group=np.array([0,0,0,1,0,0],dtype=np.uint8)
    def height(x,y):
        d=mujoco.mj_ray(model,data,np.array([x,y,5.]),np.array([0.,0.,-1.]),group,True,-1,None)
        return 5.-d if d>=0 else np.nan
    xs=np.arange(-15.,15.0001,.2);ys=np.arange(-8.,8.0001,.2)
    H=np.array([[height(x,y) for x in xs] for y in ys])
    np.savez_compressed(a.output/'field_grid.npz',x=xs,y=ys,height=H)
    edges=[]
    for axis in (0,1):
        v=H if axis==0 else H.T;cross=xs if axis==0 else ys;along=ys if axis==0 else xs
        for i in range(2,v.shape[0]-3):
            delta=v[i+1]-v[i]
            valid=(abs(delta)>.12)&(abs(delta)<.4)&(abs(v[i]-v[i-1])<.012)&(abs(v[i-1]-v[i-2])<.012)&(abs(v[i+2]-v[i+1])<.012)&(abs(v[i+3]-v[i+2])<.012)
            groups,n=label(valid)
            for k in range(1,n+1):
                j=np.where(groups==k)[0]
                if len(j)>=3:edges.append(dict(axis='y' if axis==0 else 'x',edge=float((along[i]+along[i+1])/2),
                    span=[float(cross[j.min()]-.1),float(cross[j.max()]+.1)],coarse_height_jump=float(np.median(delta[j]))))
    candidates=[]
    for name,x,edge,sign in [('red',-5.76,-5.405,-1),('blue',5.76,5.405,1)]:
        profile_y=np.arange(-8.,8.0001,.005);profile_z=np.array([height(x,y) for y in profile_y])
        fine_y=edge+np.arange(-.02,.02001,.0001);fine_z=np.array([height(x,y) for y in fine_y])
        np.savez_compressed(a.output/f'{name}_profiles.npz',y=profile_y,z=profile_z,fine_y=fine_y,fine_z=fine_z)
        # 两侧平面的截距差；将场地1.5度坡度与真实台阶高度分开。
        top_y=edge+sign*np.linspace(.08,.3,45);low_y=edge-sign*np.linspace(.08,.3,45)
        slope=np.polyfit(top_y,np.array([height(x,y) for y in top_y]),1)[0]
        top=np.median([height(x,y)-slope*y for y in top_y]);low=np.median([height(x,y)-slope*y for y in low_y])
        width_x=x+np.arange(-.9,.9001,.001)
        width_ok=np.array([abs(height(xx,edge-sign*.15)-slope*(edge-sign*.15)-low)<.003 and
            abs(height(xx,edge+sign*.2)-slope*(edge+sign*.2)-top)<.003 for xx in width_x])
        wl,_=label(width_ok);chosen=np.where(wl==wl[np.argmin(abs(width_x-x))])[0]
        clear_width=float(width_x[chosen[-1]]-width_x[chosen[0]]+.001)
        depth_t=np.arange(.02,4.,.005)
        upper_ok=np.array([abs(height(x,edge+sign*t)-slope*(edge+sign*t)-top)<.003 for t in depth_t])
        back_t=float(depth_t[np.where(~upper_ok)[0][0]])
        refined=np.arange(back_t-.005,back_t+.00001,.0001)
        back_t=float(next(t for t in refined if abs(height(x,edge+sign*t)-slope*(edge+sign*t)-top)>.003))
        upper_front=(fine_y-edge)*sign
        front_t=float(upper_front[(abs(fine_z-slope*fine_y-top)<.001)&(upper_front>-.01)].min())
        depth=back_t-front_t
        approach=np.linspace(.02,5.4,540)
        approach_error=max(abs(height(x+dx,edge-sign*t)-slope*(edge-sign*t)-low)
                           for t in approach for dx in (-.3,0.,.3))
        assert approach_error<.006
        candidates.append(dict(name=name,edge_xy=[x,edge],height_vertical_m=float(top-low),
            height_normal_m=float((top-low)/np.sqrt(1+slope*slope)),slope_deg=float(np.degrees(np.arctan(slope))),
            platform_depth_m=depth,profile_width_with_3mm_tolerance_m=clear_width,
            dimension_source='0.1mm edge rays and narrow side-closure mesh bounds',
            approach_verified_length_m=5.4,approach_width_m=.6,approach_plane_max_error_m=float(approach_error),
            match='screenshot red boundary and adjacent corrugations' if name=='red' else 'blue symmetric alternative'))
    def vertices(g):
        m=model.geom_dataid[g];s=model.mesh_vertadr[m];n=model.mesh_vertnum[m]
        return model.mesh_vert[s:s+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
    closures=[g for g in field if 'side_closure_' in model.geom(g).name and
              (model.geom_contype[g] or model.geom_conaffinity[g])]
    for candidate in candidates:
        x,y=candidate['edge_xy'];matches=[]
        for g in closures:
            v=vertices(g);lo,hi=v.min(0),v.max(0)
            if lo[0]<=x<=hi[0] and .5<hi[0]-lo[0]<2. and abs((lo[1]+hi[1])/2-y)<.02:
                matches.append((abs((lo[1]+hi[1])/2-y),g,lo,hi))
        if not matches:raise ValueError('No narrow edge closure matching candidate; width needs review')
        _,g,lo,hi=min(matches,key=lambda row:row[0])
        candidate.update(clear_width_m=float(hi[0]-lo[0]),width_collision_geom=model.geom(g).name,
                         width_endpoints_x=[float(lo[0]),float(hi[0])])
    radii={}
    for name in ('lfwheel','rfwheel'):
        b=model.body(name).id;g=next(g for g in robot if model.geom_bodyid[g]==b and (model.geom_contype[g] or model.geom_conaffinity[g]))
        q=vertices(g)-data.xanchor[model.joint(name).id];axis=data.xaxis[model.joint(name).id]
        radii[name]=float(np.linalg.norm(q-np.outer(q@axis,axis),axis=1).max())
    bid=model.body('base_link').id
    verts=np.concatenate([vertices(g) for g in robot if model.geom_bodyid[g]==bid and (model.geom_contype[g] or model.geom_conaffinity[g])])
    local=(verts-data.xpos[bid])@data.xmat[bid].reshape(3,3)
    mass=float(sum(model.body_mass[b] for b in range(1,model.nbody) if not model.body(b).name.startswith('arena/')))
    report=dict(kind='static_step0_measurement',physics_steps=0,optimizer_updates=0,
        cache=str(cache),ngeom=model.ngeom,arena_revision='ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001',
        samples=int(H.size),valid_samples=int(np.isfinite(H).sum()),coarse_edges=edges,candidates=candidates,
        selected_candidate='red; user visual confirmation still applicable',wheel_radius_m=radii,mass_kg=mass,
        upright_clearance=[dict(assumed_root_height_m=h,base_clearance_m=h+float(local[:,2].min())) for h in (.22,.26,.30)],
        estimates=dict(height_radius_ratio=.2/np.mean(list(radii.values())),potential_energy_J=mass*9.81*.2,
            energy_equivalent_speed_m_s=float(np.sqrt(2*9.81*.2)),required_approach_speed=None,
            speed_note='Energy equivalence is not a minimum or recommended approach velocity',
            assumptions='Rigid wheels; upright chassis; nominal mass; no body support. Not dynamic feasibility'),
        action_std=saved['model_state_dict']['std'].tolist(),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        elapsed_s=time.perf_counter()-start)
    if a.rules:
        report['rules']=dict(path=str(a.rules),sha256=hashlib.sha256(a.rules.read_bytes()).hexdigest(),
            printed_page=57,figure='4-35',height_mm=200,width_mm=1148,depth_mm=2090,
            tolerance_reference='printed page29: dimensions>=100mm ±5%')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,6));im=ax.pcolormesh(xs,ys,H,vmin=-.2,vmax=.6,shading='auto')
    for row in candidates:
        x,y=row['edge_xy'];ax.plot(x,y,'rx');ax.annotate(row['name'],(x,y),xytext=(8,8),textcoords='offset points')
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal',title='Original RMUC2027 collision heights (static rays)')
    fig.colorbar(im,ax=ax,label='World Z (m)');fig.savefig(a.output/'height_map.png',dpi=160);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
