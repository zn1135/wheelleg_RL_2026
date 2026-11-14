"""Independently decode uploaded USB PD500 data and verify policy/PD offline."""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MOTORS=('lf','lr','rf','rr','lw','rw')
FLAGS=('rl_selected','rl_engaged','action_ready','pd_valid','dispatched','obs_valid','warmup','drive_permit','dm_submitted','wheel_submitted')
NAMES=['h7_us','sequence','reset_count','flags','fault','board_drops','obs_sequence','obs_publish_ms',*FLAGS,
    *[f'obs_{i}' for i in range(25)],*[f'{field}_{m}' for field in ('target','current','velocity_rad_s','pd_model_nm','request_firmware_nm') for m in MOTORS]]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def quantiles(value):return np.quantile(value,[0,.5,.95,1.]).tolist()


def decode(path):
    blob=path.read_bytes();cursor=0;rx=[];kinds={}
    while cursor<len(blob):
        assert len(blob)-cursor>=16,'Truncated capture record header'
        kind,size,host=struct.unpack_from('<IIQ',blob,cursor);cursor+=16
        assert size<=len(blob)-cursor,'Truncated capture record body'
        data=blob[cursor:cursor+size];cursor+=size;kinds[kind]=kinds.get(kind,0)+1
        if kind==1:rx.append(data)
    stream=b''.join(rx);cursor=0;errors=skipped=0;types={};rows=[];sessions=[]
    while cursor+32<=len(stream):
        marker=stream.find(b'JRL2',cursor)
        if marker<0:break
        skipped+=marker-cursor;cursor=marker
        magic,version,kind,size,session,reset,sequence,us=struct.unpack_from('<4sBBHQIIQ',stream,cursor)
        if version!=2 or size>348:errors+=1;cursor+=1;continue
        length=32+size+4
        if cursor+length>len(stream):break
        expected=struct.unpack_from('<I',stream,cursor+length-4)[0]
        if zlib.crc32(stream[cursor:cursor+length-4])!=expected:errors+=1;cursor+=1;continue
        types[kind]=types.get(kind,0)+1
        if kind==5:
            assert size==240
            flags,fault,drops,obs_seq,obs_ms,*values=struct.unpack_from('<5I55f',stream,cursor+32)
            rows.append([us,sequence,reset,flags,fault,drops,obs_seq,obs_ms,*[int(bool(flags&(1<<i))) for i in range(10)],*values])
            sessions.append(session)
        cursor+=length
    assert rows
    return np.asarray(rows,dtype=np.float64),np.asarray(sessions,dtype=np.uint64),dict(capture_record_types=kinds,
        valid_frame_types=types,crc_or_header_errors=errors,resync_skipped_bytes=skipped,trailing_rx_bytes=len(stream)-cursor)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();out=a.experiment.resolve()
    torch.set_num_threads(1);x,sessions,decoding=decode(out/'raw/records.bin');col={name:i for i,name in enumerate(NAMES)}
    engaged=x[:,col['rl_engaged']]==1
    valid=engaged&(x[:,col['pd_valid']]==1)&(x[:,col['obs_valid']]==1)&(x[:,col['warmup']]==0)
    checks={}
    for label,mask in (('rl_engaged',engaged),('rl_valid',valid)):
        with (out/(label+'.csv')).open() as f:
            reader=csv.reader(f);assert next(reader)==NAMES;supplied=np.asarray(list(reader),dtype=float)
        checks[label+'_CSV_exactly_matches_raw']=bool(np.array_equal(x[mask],supplied));assert checks[label+'_CSV_exactly_matches_raw']
    with (out/'decoded-all.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(NAMES);writer.writerows(x)
    gaps=np.r_[True,(np.diff(x[:,0])<=0)|(np.diff(x[:,0])>10000)|((np.diff(x[:,1]).astype(np.int64)&0xffffffff)!=1)|
        (np.diff(x[:,2])!=0)|(sessions[1:]!=sessions[:-1])]
    starts=np.flatnonzero(gaps);segments=[]
    for start,end in zip(starts,np.r_[starts[1:],len(x)]):
        segments.append(dict(first_row=int(start),last_row=int(end-1),rows=int(end-start),
            start_h7_us=int(x[start,0]),end_h7_us=int(x[end-1,0]),duration_s=float((x[end-1,0]-x[start,0])/1e6),
            engaged_rows=int(engaged[start:end].sum()),valid_rows=int(valid[start:end].sum())))
    selected=np.flatnonzero(valid);d={name:x[valid,i] for i,name in enumerate(NAMES)}
    assert np.isfinite(x[valid]).all()
    assert np.all(np.diff(d['sequence'])==1) and np.all(np.diff(d['h7_us'])>0),'Valid rows span a discontinuity'
    unique=np.r_[0,np.flatnonzero(np.diff(d['obs_sequence'])!=0)+1]
    hold=np.diff(np.r_[unique,len(selected)])
    zero=(abs(d['obs_6'])<1e-6)&(abs(d['obs_7'])<1e-6);bound=np.r_[0,np.flatnonzero(np.diff(zero))+1,len(selected)]
    t=(d['h7_us']-d['h7_us'][0])/1e6
    zero_segments=[dict(start_s=float(t[i]),end_s=float(t[j-1]),rows=int(j-i),duration_s=float((d['h7_us'][j-1]-d['h7_us'][i])/1e6))
        for i,j in zip(bound[:-1],bound[1:]) if zero[i]]
    timing=dict(valid_rows=len(selected),valid_duration_s=float(t[-1]),pd_hz=float((len(selected)-1)/t[-1]),
        valid_dt_us=quantiles(np.diff(d['h7_us'])),observation_updates=len(unique),
        observation_update_dt_us=quantiles(np.diff(d['h7_us'][unique])),PD_rows_per_observation=quantiles(hold),
        associated_observation_age_ms=quantiles((d['h7_us']-1000*d['obs_publish_ms'])/1000),
        sequence_breaks=int(np.count_nonzero(np.diff(d['sequence'])!=1)),
        board_drop_counter_first_last=[int(d['board_drops'][0]),int(d['board_drops'][-1])],
        board_drop_counter_increment=int(d['board_drops'][-1]-d['board_drops'][0]),fault_values=np.unique(d['fault']).astype(int).tolist())
    motors={}
    for m in MOTORS:
        error=d['target_'+m]-d['current_'+m];v=d['velocity_rad_s_'+m];pd=d['pd_model_nm_'+m]
        wheel=m in ('lw','rw');limit=3.9 if wheel else 40.;expected=np.clip((.1 if wheel else 60.)*error-(0. if wheel else 2.)*v,-limit,limit)
        free=abs(pd)<limit-1e-4;mat=error[free,None] if wheel else np.c_[error[free],-v[free]]
        gain=np.linalg.lstsq(mat,pd[free],rcond=None)[0];firmware=d['request_firmware_nm_'+m]
        sign=float(np.dot(pd,firmware)/np.dot(pd,pd))
        motors[m]=dict(fitted_pd=gain.tolist(),known_pd_max_abs_error_nm=float(abs(pd-expected).max()),
            limited_torque_fraction=float(np.mean(~free)),rms_tracking_error=float(np.sqrt(np.mean(error**2))),
            firmware_request_over_model_pd=sign,firmware_mapping_max_abs_error=float(abs(firmware-sign*pd).max()))
    # 推理数据以一个观测为一组，下一组previous_action对应当前组的策略动作。
    obs=np.stack([d[f'obs_{i}'][unique] for i in range(25)],axis=1).astype('float32')
    assert np.all(np.diff(d['obs_sequence'][unique])==1) and np.all(hold==5)
    history=np.stack([obs[i-4:i+1].reshape(-1) for i in range(4,len(obs)-1)])
    current=obs[4:-1];actual=obs[5:,19:25]
    targets=np.stack([d['target_'+m][unique][4:-1] for m in ('lf','lr','lw','rf','rr','rw')],axis=1)
    predicted_targets=np.empty_like(targets);nominal=np.array([-.6047205544565015,.6083241726212976,.6047205544560037,-.6083241726229305])
    predicted_targets[:,[0,1,3,4]]=nominal+.5*actual[:,[0,1,3,4]]
    predicted_targets[:,[2,5]]=np.clip(10*actual[:,[2,5]],-150,150)
    mapping_error=float(abs(targets-predicted_targets).max());assert mapping_error<1e-5
    models=[]
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    for label,path in (('model3000',ROOT/'artifacts/fudan_ppo_walk/models/20261113_model3000_p60d2_armature2_wheelP01_mechanical.pt'),
        ('old17679',ROOT/'artifacts/fudan_ppo_walk/models/20261112_model17679_p60d2_armature2_wheelP045_output045.pt')):
        with redirect_stdout(io.StringIO()):policy,metadata=load_policy(path,'cpu')
        with torch.inference_mode():pred,_=policy.act_inference(torch.from_numpy(current),torch.from_numpy(history))
        pred=pred.numpy().clip(-100,100);err=pred-actual
        models.append(dict(model=label,checkpoint_sha256=sha(path),iterations=metadata['iterations'],frames=len(pred),
            rms_action_error=float(np.sqrt(np.mean(err**2))),max_abs_action_error=float(abs(err).max()),
            per_action_rms_error=np.sqrt(np.mean(err**2,axis=0)).tolist(),
            allclose_rows_fraction=float(np.mean(np.all(abs(err)<=1e-4+1e-4*abs(actual),axis=1)))))
    assert models[0]['allclose_rows_fraction']==1.
    result=dict(decoding=decoding,CSV_checks=checks,segments=segments,timing=timing,PD_checks=motors,
        model_replay=models,previous_action_to_current_target_max_error=mapping_error,
        command_ranges=dict(vx_m_s=quantiles(d['obs_6']/2),yaw_rad_s=quantiles(d['obs_7']/.25),height_m=quantiles(d['obs_8']/5)),
        zero_command_segments=zero_segments,longest_zero_command_s=max(s['duration_s'] for s in zero_segments),
        observation_absolute_max=float(abs(obs).max()),metadata_record_mode_only=True,
        scope='CPU offline decode/inference only; no physics or learning, no actual motor torque/height/contact measurement')
    write(out/'analysis.json',result)
    plot(out,d,t)
    print(json.dumps(dict(decoding=decoding,timing=timing,PD_checks=motors,model_replay=models,longest_zero_command_s=result['longest_zero_command_s']),indent=2))


def plot(out,d,t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    axes[0].plot(t,d['obs_6']/2,label='vx command (m/s)');axes[0].plot(t,d['obs_7']/.25,label='yaw command (rad/s)',alpha=.7)
    axes[0].set_ylabel('Command');axes[0].legend(ncol=2);axes[0].set_title('Model3000 / valid RL telemetry / PD500Hz, policy100Hz / height command0.22m')
    for ax,m in zip(axes[1:3],('lw','rw')):
        ax.plot(t,d['target_'+m],label='Target',lw=.8);ax.plot(t,d['current_'+m],label='Measured speed',lw=.8)
        ax.set_ylabel(m+' (rad/s)');ax.legend(ncol=2)
    for m in ('lw','rw'):axes[3].plot(t,d['pd_model_nm_'+m],label=m,lw=.8)
    axes[3].axhline(3.9,color='red',ls='--',lw=.6);axes[3].axhline(-3.9,color='red',ls='--',lw=.6)
    axes[3].set_ylabel('Wheel request (Nm)');axes[3].legend(ncol=2);axes[3].set_xlabel('Time since first valid RL row (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'commands-and-wheels.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for ax,m in zip(axes,('lf','lr','rf','rr')):
        ax.plot(t,np.rad2deg(d['target_'+m]),label='Target',lw=.9);ax.plot(t,np.rad2deg(d['current_'+m]),label='Measured position',lw=.9)
        ax.set_ylabel(m+' (deg)');ax.grid(alpha=.2);ax.legend(ncol=2)
    axes[0].set_title('Leg position tracking / motion commands present; not a zero-command oscillation test')
    axes[-1].set_xlabel('Time since first valid RL row (s)');fig.tight_layout();fig.savefig(out/'leg-tracking.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
