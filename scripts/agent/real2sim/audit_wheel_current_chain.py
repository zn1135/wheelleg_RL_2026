"""Read-only source/current-code audit and legacy pulse feedback inventory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    driver=root/'deployment/H7_RL_ros/imcalib/drivers/dji.c';header=driver.with_suffix('.h')
    source=driver.read_text();start=source.index('int16_t Dji_Torque_To_Current(');opening=source.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    function=source[start:end]
    constants='\n'.join(line for line in header.read_text().splitlines() if line.startswith('#define DJI_') and any(key in line for key in ('CURRENT_MAX','NM_FULL','RATIO_STD')))
    harness='#include <stdint.h>\n#include <stdio.h>\n#define DJI_MOTOR_NUM 2\n#define DJI_M2006 0\n'+constants+'\nstruct config { uint8_t dji_type; float dji_gear_ratio; };\nstatic struct config cfg={1,15.3333333f};\nstatic const struct config *machine=&cfg;\n'+function+'\nint main(void){float x[]={-6,-3.9f,-1,0,1,3.9f,6};for(int i=0;i<7;i++)printf("%.9g,%d\\n",x[i],Dji_Torque_To_Current(0,x[i]));}\n'
    (out/'conversion.c').write_text(harness)
    compiler=shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    values=np.asarray([-6,-3.9,-1,0,1,3.9,6],dtype=np.float32)
    if compiler:
        subprocess.run([compiler,'-std=c99',str(out/'conversion.c'),'-o',str(out/'conversion')],check=True)
        result=subprocess.run([str((out/'conversion').resolve())],check=True,text=True,capture_output=True).stdout
    else:
        per_float=np.float32(np.float32(.30)*np.float32(20)/np.float32(16384))*np.float32(np.float32(15.3333333)/np.float32(19.2))
        result='\n'.join(f'{float(v)},{int(np.clip(np.trunc(v/per_float),-16384,16384))}' for v in values)
    per_raw=(.3*20/16384)*(15.3333333/19.2);rows=[]
    for line in result.splitlines():
        torque,raw=map(float,line.split(','));expected=int(np.clip(np.trunc(torque/per_raw),-16384,16384))
        assert int(raw)==expected
        rows.append(dict(request_Nm=torque,current_code=int(raw),nominal_command_A=float(raw*20/16384)))
    old=root/'artifacts/real2sim/wheel-bench-3p00-300ms-speed30-test-20261021/wheel-bench-3p00-speed30-pulses-20261021/analysis/trajectory.csv'
    fields=['output_tick_ms']+[prefix+wheel+suffix for wheel in ('lfwheel','rfwheel')
        for prefix,suffix in (('gated_torque_','_firmware_nm'),('wheel_current_','_raw'),('motor_age_','_ms'))]
    with old.open() as f:
        records=list(csv.DictReader(f))
    col={name:np.asarray([float(row[name]) for row in records]) for name in fields};pulses=[]
    for wheel in ('lfwheel','rfwheel'):
        request=col['gated_torque_'+wheel+'_firmware_nm'];current=col['wheel_current_'+wheel+'_raw']
        active=abs(request)>.01;begins=np.flatnonzero(active&~np.r_[False,active[:-1]]);ends=np.flatnonzero(active&~np.r_[active[1:],False])+1
        for s,e in zip(begins,ends):
            nominal=request[s:e]/per_raw;ratio=current[s:e]/nominal
            pulses.append(dict(wheel=wheel,start_output_ms=float(col['output_tick_ms'][s]),samples=int(e-s),
                request_Nm=float(request[s]),feedback_mean_raw=float(current[s:e].mean()),
                feedback_to_reconstructed_current_mean=float(ratio.mean()),
                feedback_to_reconstructed_current_minmax=[float(ratio.min()),float(ratio.max())],
                feedback_age_ms_max=float(col['motor_age_'+wheel+'_ms'][s:e].max())))
    current=root/'artifacts/real2sim/model3000-usb-pd500-20261113-01'
    decoded=json.loads((current/'analysis.json').read_text())['decoding']
    with (current/'rl_valid.csv').open() as f:fields=next(csv.reader(f))
    report=dict(source_scope='Current repository H7_RL_ros source, not verified flashed firmware identity for model3000 recording',
        conversion_function_C_checked=bool(compiler),conversion_check='host C execution' if compiler else 'float32 formula versus independent double calculation; native C compiler unavailable',conversion=rows,nominal_Nm_per_raw=per_raw,
        current_log_frame_types=decoded['valid_frame_types'],current_log_actual_current_fields=[name for name in fields if 'current_raw' in name],
        old_capture='2026-10-21 suspended single-wheel pulses; 100Hz mirror, asynchronous command/feedback; not current3000 run',
        old_pulses=pulses,actual_response_tau_identified=False,
        known_existing_fields=dict(vofa_feedback_raw=[150,151],vofa_feedback_age_ms=[142,143],output_capture_CAN='four actual submitted current-code slots plus CAN ID and begin/end times'),
        stale_env_path=dict(path='/home/zn/文档/H7_clion',git_repository=False,gear_ratio=15.5,current_ros_source_gear_ratio=15.3333333,warning='Neither path alone identifies flashed firmware'),
        limits=['Nm-to-raw arithmetic does not calibrate shaft torque or efficiency.','Legacy feedback/request ratios are transient sampled associations, not fixed output gain.','Current log lacks paired actual CAN bytes and motor feedback current; cannot identify actual current-loop lag.'],
        source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (driver,header,old,current/'rl_valid.csv')},
        hardware_access=False,physics_steps=0,training_updates=0)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,out/'source.py')
    print(json.dumps(dict(conversion=rows,legacy_feedback_ratios=[r['feedback_to_reconstructed_current_mean'] for r in pulses],actual_response_tau_identified=False),indent=2))


if __name__=='__main__':main()
