"""报告基线、下降提前10ms和重复基线的配对结果。"""
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compare_landing_lead import PREVIOUS, samples, jump_metrics, extension
from .probe_takeoff_work import CHECKPOINT, CHECKPOINT_SHA, WALK, sha, write_json

GROUPS = ('baseline','lead10','repeat')
LABELS = ('原5200','伸腿提前10 ms','原5200重复')
MEASURES = ('gap_cm','dual_delay_ms','sink_cm','peak_cm','nonwheel_ms','stable')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def replay(out, index):
    model=mujoco.MjModel.from_binary_path(str(out/'baseline/model.mjb'))
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=1.35;camera.azimuth=8.;camera.elevation=-10.
    font=ImageFont.truetype(FONT,20)
    records=[samples(out/name,index) for name in GROUPS[:2]]; metrics=[jump_metrics(r) for r in records]
    count=0
    with mujoco.Renderer(model,height=320,width=480) as renderer,imageio.get_writer(out/'paired-slow.mp4',fps=50,codec='libx264',quality=8) as writer:
        for tick in range(2000,max(m['land'] for m in metrics)+351,5):
            canvas=Image.new('RGB',(960,400),'#18242f');draw=ImageDraw.Draw(canvas)
            for col,(raw,m) in enumerate(zip(records,metrics)):
                data.qpos[:]=raw['qpos_pre'][tick];mujoco.mj_forward(model,data);camera.lookat[:]=[data.qpos[0],data.qpos[1],.44]
                renderer.update_scene(data,camera=camera);canvas.paste(Image.fromarray(renderer.render()),(col*480,40))
                draw.text((col*480+10,4),f'{LABELS[col]}  ·  同初态 #{index:02d}',font=font,fill='white')
                draw.text((col*480+8,365),f'触发后 {tick-2000} ms   距首次承重 {tick-m["land"]:+d} ms',font=font,fill='white')
            writer.append_data(np.asarray(canvas));count+=1
    with imageio.get_reader(out/'paired-slow.mp4') as reader:
        assert reader.get_meta_data()['fps']==50 and sum(1 for _ in reader)==count
    return dict(case=index,frames=count,fps=50,speed=.25,full_decode=True)


def run(out):
    plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    stats={g:[] for g in (*GROUPS,'historical')};curves={g:[] for g in (*GROUPS,'historical')};audit=[]
    intervention=np.load(out/'lead10/intervention.npz');meta=json.loads((out/'baseline/metadata.json').read_text());qcol=np.array(meta['motor_qadr'])[[0,1,3,4]]
    rel=np.arange(-350,151);real=json.loads((out/'real.json').read_text())
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    for i in range(20):
        inputs=None
        for group in stats:
            path=PREVIOUS if group=='historical' else out/group
            raw=samples(path,i);m=jump_metrics(raw);stats[group].append(m)
            initial=[raw[k] for k in ('qpos0','qvel0','history0')]
            if inputs is None:inputs=initial
            else:assert all(np.array_equal(a,b) for a,b in zip(initial,inputs))
            t=raw['trace'];land=m['land'];assert len(t)==6000 and np.isfinite(t).all() and land+1100<len(t)
            idx=land+rel;target=extension(raw['target'][idx]);actual=extension(raw['qpos_pre'][idx][:,qcol]);requested=extension(raw['pd_torque'][idx][:,[0,1,3,4]]);force=extension(raw['actual_force'][idx][:,[0,1,3,4]])
            assert abs(raw['actual_force'][:,:6]-raw['pd_torque']).max()<1e-4 and abs(raw['actual_force'][:,[0,1,3,4]]).max()<=40.00001
            curves[group].append(np.stack([target[:,1],actual[:,1],requested[:,1],force[:,1],(t[idx,6]-t[idx,7])*100,(t[idx,16]-t[land,16])*100]))
            if group=='lead10':
                delta=intervention['added_diff_rad'][:,i];ticks=intervention['ticks'][(delta>1e-6).any(1)]*10
                assert len(ticks) and np.all(raw['mode'][ticks//10]) and (ticks>=m['apex']-1).all() and (ticks<land).all()
                audit.append(dict(id=i,modified_ticks=len(ticks),first_relative_apex_ms=int(ticks.min()-m['apex']),last_relative_contact_ms=int(ticks.max()-land),max_added_diff_rad=float(delta.max())))
    rows=[]
    for i in range(20):
        item=dict(case=i)
        for g in GROUPS:
            item.update({f'{g}_{key}':stats[g][i][key] for key in MEASURES})
        item.update({f'delta_{key}':stats['lead10'][i][key]-stats['baseline'][i][key] for key in MEASURES[:-1]})
        item['stable_outcome']=('保留' if stats['lead10'][i]['stable'] else '丢失') if stats['baseline'][i]['stable'] else ('新增' if stats['lead10'][i]['stable'] else '均未通过')
        rows.append(item)
    with (out/'paired.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    medians={g:{key:float(np.median([x[key] for x in v])) for key in MEASURES[:-1]} for g,v in stats.items()}
    counts={g:sum(int(x['stable']) for x in v) for g,v in stats.items()}
    # 选取与轮底高差配对变化中位数最接近的初态，所有图表给出同一编号。
    diff=np.array([x['delta_gap_cm'] for x in rows]);i=int(np.argmin(abs(diff-np.median(diff))))
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True,constrained_layout=True)
    for col,trial in enumerate((1,2)):
        s=np.load(out/f'real{trial}.npz');series=np.median(curves['historical'],axis=0)
        for j,key in enumerate(('target','request','feedback','gap_cm')):
            ax=axes[j,col];values=s[key] if j==3 else s[key][:,1]
            ax.plot(s['t_ms'],values,label='实机',color='tab:red');ax.plot(rel,series[[0,2,3,4][j]],label='历史GPU中位',color='tab:blue')
            if j==0:
                ax.plot(s['t_ms'],s['actual'][:,1],ls=':',color='tab:red',label='实机实际角差');ax.plot(rel,series[1],ls=':',color='tab:blue',label='GPU实际角差')
            ax.axvspan(*s['contact_ms'],alpha=.1,color='green');ax.axvline(0,color='grey',lw=.7);ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
            ax.set_ylabel(('右腿伸展角差 / rad','右腿请求差动力矩 / N·m','右腿反馈差动力矩 / N·m','左减右轮底 / cm')[j])
        axes[0,col].set_title(f'试次{trial} · 实机零点为视频触地窗口中点')
        axes[-1,col].set_xlabel('相对首次承重 / ms')
    fig.savefig(out/'real-vs-sim.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for g,label in zip(GROUPS,LABELS):
        data=curves[g][i]
        for ax,k in zip(axes,(0,4,5)):ax.plot(rel,data[k],label=label)
    for ax,title in zip(axes,('右腿伸展目标 / rad','左减右轮底 / cm','机身相对首次承重高度 / cm')):
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.axvline(0,color='grey',lw=.7);ax.legend()
    axes[0].set_title(f'同初态 #{i:02d} · 各自首次承重对齐');axes[-1].set_xlabel('相对首次承重 / ms');fig.savefig(out/'paired-curves.png',dpi=140);plt.close(fig)
    result=dict(medians=medians,stable_counts=counts,intervention=audit,real=real,video=replay(out,i),
        peak_reference_sha=CHECKPOINT_SHA,walk_sha=sha(WALK),optimizer_updates=0,physics_dt_ms=1,pd_dt_ms=5,policy_dt_ms=10,
        comparisons={key:dict(improved=int(sum(row['delta_'+key]<0 for row in rows)),unchanged=int(sum(row['delta_'+key]==0 for row in rows)),worse=int(sum(row['delta_'+key]>0 for row in rows))) for key in ('gap_cm','sink_cm','dual_delay_ms')})
    write_json(out/'report.json',result);print(json.dumps(dict(medians=medians,stable=counts),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out)
