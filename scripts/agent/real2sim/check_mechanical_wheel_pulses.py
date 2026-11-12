"""Historical suspended-wheel pulse diagnostic with a reduced fixed rotor; no hardware."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'artifacts/real2sim/fixed-h7-fixed300-data-20261021'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
PARAMS=XML.with_name('wheel_mechanical.json')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);source=ET.parse(XML).getroot();mechanical=json.loads(PARAMS.read_text())
    results=[];curves=[];hashes={}
    for split in ('fit-complete','validation-complete'):
        directory=DATA/(split+'-analysis');summary=json.loads((directory/'summary.json').read_text())
        path=directory/'outputs.csv';hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f:
            rows=[r for r in csv.DictReader(f) if r['kind']=='control']
        stamps=np.array([int(r['begin_ns']) for r in rows],dtype=np.int64)
        for pulse in summary['pulses']:
            side=0 if pulse['side']=='left' else 1;name=('lfwheel','rfwheel')[side]
            keep=(stamps>=pulse['on_ns']-100000000)&(stamps<=pulse['off_ns']+2500000000)
            chosen=[r for r,ok in zip(rows,keep) if ok];stamp=stamps[keep];start=int(stamp[0]);dt=.002
            actual=np.array([float(r[f'wheel_{side}_speed_rad_s']) for r in chosen])
            age=np.array([float(r[f'wheel_{side}_rx_age_us']) for r in chosen])
            measured_time=(stamp-start-age*1000)/1e9
            grid=np.arange(int(np.ceil((stamp[-1]-start)/1e9/dt))+1)*dt
            index=np.searchsorted(stamp,start+np.rint(grid[:-1]*1e9).astype(np.int64),side='right')-1
            request=np.array([float(r[f'request_{side+4}_firmware_nm']) for r in chosen])[index]
            on=(pulse['on_ns']-start)/1e9;off=(pulse['off_ns']-start)/1e9
            mask=((measured_time>=on+.005)&(measured_time<off-.005))|((measured_time>=off+.005)&(measured_time<=off+2.45))
            body=source.find(f'.//body[@name="{name}"]');joint=body.find('joint');inertial=body.find('inertial')
            gain=np.fromstring(source.find(f'.//general[@name="{name}_motor"]').attrib['gainprm'],sep=' ')
            for profile in ('original_curve','curve045','mechanical'):
                root=ET.Element('mujoco');ET.SubElement(root,'option',timestep=str(dt),gravity='0 0 0',integrator='implicitfast',solver='Newton',iterations='100',tolerance='1e-10')
                b=ET.SubElement(ET.SubElement(root,'worldbody'),'body',name=name)
                fields={f:float(joint.attrib[f]) for f in ('armature','damping','frictionloss')}
                if profile=='mechanical':fields={f:mechanical[f][side] for f in fields}
                ET.SubElement(b,'joint',name=name,type='hinge',axis=joint.attrib['axis'],limited='false',**{k:str(v) for k,v in fields.items()})
                ET.SubElement(b,'inertial',**inertial.attrib)
                ET.SubElement(ET.SubElement(root,'actuator'),'general',name=name+'_motor',joint=name,gear='1',ctrlrange='-3.9 3.9',gaintype='fixed',gainprm=str(1. if profile=='mechanical' else gain[0]))
                model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model);data.qvel[0]=actual[0]
                predicted=[actual[0]]
                for u in request:
                    ctrl=float(np.clip(u,-3.9,3.9))
                    if profile!='mechanical':
                        shape=max(0.,1-max(0.,np.sign(ctrl)*data.qvel[0])/gain[1])**gain[2]
                        ctrl*=shape*(.45 if profile=='curve045' else 1.)
                    data.ctrl[0]=ctrl;mujoco.mj_step(model,data);predicted.append(float(data.qvel[0]))
                prediction=np.interp(measured_time,grid,predicted);error=prediction[mask]-actual[mask]
                results.append(dict(split=split,pulse=pulse['number'],side=pulse['side'],profile=profile,
                    rmse_rad_s=float(np.sqrt(np.mean(error**2))),mae_rad_s=float(np.mean(abs(error))),samples=len(error)))
                if split=='validation-complete':curves.append(dict(pulse=pulse['number'],side=pulse['side'],profile=profile,time=measured_time-on,predicted=prediction,actual=actual))
    aggregates=[]
    for split in ('fit-complete','validation-complete'):
        for profile in ('original_curve','curve045','mechanical'):
            selected=[r for r in results if r['split']==split and r['profile']==profile];n=sum(r['samples'] for r in selected)
            aggregates.append(dict(split=split,profile=profile,rmse_rad_s=float(np.sqrt(sum(r['rmse_rad_s']**2*r['samples'] for r in selected)/n))))
    result=dict(scope='Reduced fixed one-hinge rotor, actual nominal commands and receive-age timestamps; gravity disabled, not full-body GPU acceptance',
        dt_s=.002,mechanical_parameters=mechanical,rows=results,aggregate=aggregates,input_sha256=hashes,training_updates=0,
        limitations=['Historical3Nm/300ms suspended pulses only','No contact/load or low-command calibration','Reduced rotor ignores small gravity moment and measured support pose'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for i,ax in enumerate(axes):
        rows=[r for r in curves if r['pulse']==i+1]
        for r in rows:ax.plot(r['time'],r['predicted'],label=r['profile'])
        ax.plot(rows[0]['time'],rows[0]['actual'],'k--',lw=1,label='measured');ax.set_title(f'Pulse{i+1} {rows[0]["side"]}');ax.set_ylabel('rad/s');ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Seconds from command onset');fig.tight_layout();fig.savefig(args.output/'pulse-comparison.png',dpi=140)
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
