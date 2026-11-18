"""离线复核4200轮GPU证据、初态配对与1倍速视频；不推进物理。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def review(root):
    from scripts.agent.motion.acceptance import compare_baseline
    from .review_final3200 import pair,detail
    assert json.loads((root/'execution.json').read_text())['passed']
    receipt=json.loads((root/'input-receipt.json').read_text())
    results={};protocols={}
    for model in ('3200','4200'):
        folder=root/f'stairs-{model}'
        p=json.loads((folder/'protocol.json').read_text());protocols[model]=p
        assert p['source_sha256']==receipt['checkpoints'][model]==sha(p['source_checkpoint'])
        assert p['all_finite'] and p['all_networks_unchanged'] and p['optimizer_updates']==0
        assert json.loads((folder/'audit/audit.json').read_text())['passed']
        metrics=json.loads((folder/'metrics.json').read_text())
        for case in p['cases']:
            assert sha(folder/'traces'/(case['id']+'.npz'))==p['trace_sha256'][case['id']]
        chosen=[r for r in metrics if r['height']>0]
        turns=json.loads((root/f'turns-{model}/acceptance.json').read_text())
        turn_metrics=json.loads((root/f'turns-{model}/metrics.json').read_text())
        assert json.loads((root/f'turns-{model}/receipt.json').read_text())['checkpoint_sha256']==receipt['checkpoints'][model]
        for row in turn_metrics:
            assert sha(root/f'turns-{model}/traces'/(row['id']+'.npz'))==row['trace_sha256']
        results[model]=dict(stairs_total=len(chosen),stairs_landed=sum(r['landed'] for r in chosen),
            stairs_full_parked=sum(r['task_score']['passed'] for r in chosen),
            stairs_strict_parked=sum(r['parked'] for r in chosen),
            by_speed=[dict(speed=v,total=len(batch),full_parked=sum(r['task_score']['passed'] for r in batch))
                for v in (.3,1.,2.5) for batch in [[r for r in chosen if r['speed']==v]]],
            turns=[dict(height=h,category=category,total=len(batch),passed=sum(r['passed'] for r in batch),
                yaw_mae_mean=float(np.mean([r['yaw_mae'] for r in batch])))
                for h in (.26,.35) for category in ('spin','turn','zero')
                for batch in [[r for r in turn_metrics if r['height']==h and r['id'].startswith(category+'_')]]],
            turn_details=turn_metrics,
            stair_details=[detail(folder,case,next(r for r in metrics if r['id']==case['id']))
                           for case in p['cases'] if case['height']>0])
        protocols['turn'+model]=turns
    pairing=pair(root/'stairs-3200',root/'stairs-4200',protocols['3200'],protocols['4200'])
    turn_comparison=compare_baseline(protocols['turn4200'],protocols['turn3200'])['comparison']
    old={r['id']:r for r in results['3200']['stair_details']}
    new={r['id']:r for r in results['4200']['stair_details']}
    flat=json.loads((root/'flat47/acceptance/acceptance.json').read_text())
    assert flat['total']==47 and json.loads((root/'flat47/runtime-audit.json').read_text())['passed']
    # 将已采集的27例交统一入口，保留完整矩阵缺测标记。
    p=protocols['4200'];folder=root/'stairs-4200'
    samples=[dict(id=c['id'],mode='up',height=c['height'],speed=c['speed'],
        angle_deg=c['angle'],friction=c['friction'],reverse=False) for c in p['cases'] if c['height']>0]
    cohort=hashlib.sha256()
    for sample in samples:
        with np.load(folder/'traces'/(sample['id']+'.npz')) as trace:
            initial=trace['initial'].copy();initial[:3]-=trace['origin'];cohort.update(initial.tobytes())
        cohort.update(np.array(sample['friction'],dtype=np.float32).tobytes())
    write(folder/'step-protocol.json',dict(task='fudan_ppo_step',backend='mujoco_warp_gpu',samples=samples,
        duration_s=20.,dt=.01,recipe=p['recipe'],checkpoint_sha256=receipt['checkpoints']['4200'],
        cohort_sha256=cohort.hexdigest(),trace_sha256=p['trace_sha256'],source_fingerprint=p['source_fingerprint'],
        source_protocol_sha256=sha(folder/'protocol.json'),optimizer_updates=0))
    write(root/'flat47/gpu/step-policy-receipt.json',dict(checkpoint_sha256=receipt['checkpoints']['4200'],
        optimizer_updates=0,source_plan_sha256=sha(root/'flat47/plan.json')))
    repository=Path(__file__).resolve().parents[3]
    with (root/'unified-acceptance.log').open('x') as stream:
        subprocess.run([str(repository/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_step',
            '--recording',str(folder),'--output',str(root/'unified-acceptance'),
            '--flat-recording',str(root/'flat47/gpu'),'--flat-baseline',
            str(repository/'artifacts/fudan_ppo_step/20261117-v7-effect200-01/evidence/baseline-flat47/gpu')],
            cwd=repository,stdout=stream,stderr=subprocess.STDOUT,check=True)
    unified=json.loads((root/'unified-acceptance/acceptance.json').read_text())
    assert unified['technical_passed']==results['4200']['stairs_full_parked']
    assert not unified['step_matrix_complete'] and not unified['full_task_accepted']
    result=dict(evidence_verified=True,models=results,stair_pairing=pairing,turn_comparison=turn_comparison,
        stairs_lost=[key for key,row in old.items() if row['full_parked'] and not new[key]['full_parked']],
        stairs_gained=[key for key,row in old.items() if not row['full_parked'] and new[key]['full_parked']],
        flat_passed=flat['technical_passed'],flat_total=47,flat_comparison=flat['comparison'],
        flat_cohort_sha256=flat['cohort_sha256'],optimizer_updates=0,
        full_task_accepted=False,not_tested=['randomized_full_population','full_downstairs_matrix','full_arena','hardware'])
    write(root/'review.json',result)
    print(json.dumps(dict(models={key:{k:v for k,v in val.items() if k not in ('stair_details','turn_details')}
                         for key,val in results.items()},flat_passed=flat['technical_passed'],
                         stair_initials_matched=pairing['all_initials_matched']),ensure_ascii=False))


def video(root,case):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image,ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    output=root/'video-turn';output.mkdir(exist_ok=False)
    panels=[]
    try:
        for model in ('3200','4200'):
            folder=root/f'turns-{model}'
            mj=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'))
            disable_model_sameframe_shortcuts(mj)
            mj.vis.global_.offwidth=640;mj.vis.global_.offheight=480
            data=mujoco.MjData(mj);cam=mujoco.MjvCamera()
            cam.distance=1.6;cam.azimuth=90.;cam.elevation=-45.
            with np.load(folder/'traces'/(case+'.npz')) as saved:trace={k:saved[k] for k in saved.files}
            panels.append((model,mj,data,cam,trace,mujoco.Renderer(mj,width=640,height=480)))
        path=output/'turn-1x.mp4';fps=25;frames=375
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=7,macro_block_size=None) as writer:
            for frame in range(frames):
                t=frame/fps;canvas=Image.new('RGB',(1280,480),'#18212b')
                for col,(label,mj,data,cam,tr,renderer) in enumerate(panels):
                    i=min(len(tr['qpos'])-1,round(t/.01));k=min(len(tr['time'])-1,round(t/.001))
                    data.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(mj,data)
                    cam.lookat[:]=data.qpos[:3]+[0.,0.,.05]
                    renderer.update_scene(data,camera=cam)
                    panel=Image.fromarray(renderer.render());pen=ImageDraw.Draw(panel)
                    pen.rectangle((0,0,640,64),fill='#18212b')
                    pen.text((8,5),f'STEP {label} | GPU replay 1x | h=0.35m | t={t:.2f}s',fill='white')
                    pen.text((8,25),f'yaw cmd={0.5 if t<7 else 0.:.2f}, actual={tr["angular_body"][k,2]:+.2f} rad/s',fill='white')
                    canvas.paste(panel,(640*col,0))
                writer.append_data(np.asarray(canvas))
                if frame in (0,150,374):canvas.save(output/f'key-{frame:03d}.png')
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps']==fps and sum(1 for _ in reader)==frames
        write(output/'media.json',dict(video_sha256=sha(path),frames=frames,fps=fps,duration_s=15.,
            case=case,trace_sha256={model:sha(root/f'turns-{model}/traces'/(case+'.npz')) for model in ('3200','4200')},
            physics_steps=0,optimizer_updates=0))
    finally:
        for panel in panels:panel[-1].close()


def media(root):
    import imageio.v2 as imageio
    rows=[]
    for folder,filename in (('video-turn','turn-1x.mp4'),('video-stair','replay-1x.mp4')):
        meta=json.loads((root/folder/'media.json').read_text());path=root/folder/filename
        assert sha(path)==meta['video_sha256']
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps']==meta['fps']
            count=sum(1 for _ in reader)
        assert count==meta['frames']
        if folder=='video-turn':
            for model,digest in meta['trace_sha256'].items():
                assert sha(root/f'turns-{model}/traces'/(meta['case']+'.npz'))==digest
        else:
            assert sha(root/'stairs-4200/traces'/(meta['case']+'.npz'))==meta['trace_sha256']
        rows.append(dict(file=str(path),frames=count,fps=meta['fps'],full_decode_passed=True))
    write(root/'media-audit.json',dict(passed=True,videos=rows,physics_steps=0,optimizer_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=('review','video','media'))
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--case',default='spin_v+0_y+0.5_h0.35')
    args=parser.parse_args()
    if args.mode=='review':review(args.recording)
    elif args.mode=='video':video(args.recording,args.case)
    else:media(args.recording)
