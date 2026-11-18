"""Frozen6838/7500 GPU comparison: clean47, matched sensor noise, high-stance/parking repeats."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from . import diagnose_gyro_x as core
from . import inspect_randomization_4000 as frozen
from .run_torque_ablation import run_command

ROOT = frozen.ROOT
MODULE = 'scripts.agent.fudan_ppo_walk.evaluate_gyrox7500'
NEW_RUN = ROOT / 'logs/mjlab/fudan_ppo_walk/20261118_gyrox_noise06_resume6838_to7500_01'
TRAINING = ROOT / 'artifacts/fudan_ppo_walk/20261118-gyrox-resume6838-7500-02'
NEW_SHA = '3d56fddb250b85d9ca1dbe2e8c0c045c1c8d90139b4a6e4ea0596d5c496061e8'


def groups(batch):
    result = []
    if batch == 'main':
        result = [dict(id='source_clean', model=6838, noise='clean', seed=71001),
                  dict(id='candidate_clean', model=7500, noise='clean', seed=71001),
                  dict(id='source_sham', model=6838, noise='clean', seed=71001)]
    for seed in ((71001,) if batch == 'main' else (71002, 71003)):
        for noise in ('old', 'new'):
            for model in (6838, 7500):
                result.append(dict(id=f'm{model}_{noise}_s{seed}', model=model, noise=noise, seed=seed))
    return result


def prepare(out):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    assert json.loads((TRAINING / 'completion.json').read_text())['passed']
    assert frozen.sha(NEW_RUN / 'model_final.pt') == NEW_SHA
    out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((core.study.RUN / 'effective-training-config.json').read_text())
    target = json.loads((NEW_RUN / 'effective-training-config.json').read_text())
    restore = json.loads(json.dumps(target)); restore['normalization']['noise_scales'][0] = .075
    assert restore == baseline and target['normalization']['noise_scales'][0] == .15
    for run in (core.study.RUN, NEW_RUN):
        for name, digest in json.loads((run / 'source_sha256.json').read_text()).items():
            if name.startswith('mjlab_training/src/ct_mjlab/'):
                assert frozen.sha(run / 'source/ct_mjlab' / name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest
            elif name.startswith('resources/'):
                assert frozen.sha(ROOT / name) == digest
    a = json.loads((core.study.RUN / 'source_sha256.json').read_text())
    b = json.loads((NEW_RUN / 'source_sha256.json').read_text())
    assert all(b[n] == h for n, h in a.items() if n.startswith('mjlab_training/src/ct_mjlab/'))
    saved = torch.load(NEW_RUN / 'model_final.pt', map_location='cpu', weights_only=False)
    assert saved['iterations'] == 7500 and not saved['partial_update']
    assert saved['primary_updates'] == saved['encoder_updates'] == 150000
    assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
    events = EventAccumulator(str(NEW_RUN), size_guidance={'scalars': 0}).Reload()
    rewards = events.Scalars('Train/reward'); assert [x.step for x in rewards] == list(range(6839, 7501))
    frozen.write(out / 'training-review.json', dict(passed=True, final_iteration=7500, additional_iterations=662,
        final_sha256=NEW_SHA, optimizer_steps_each=150000, source_implementation_identical=True,
        sole_recipe_change={'normalization.noise_scales[0]': [.075, .15]},
        reward_first100=float(np.mean([x.value for x in rewards[:100]])),
        reward_last100=float(np.mean([x.value for x in rewards[-100:]])),
        training_report=json.loads((NEW_RUN / 'training_report.json').read_text())))
    frozen.write(out / 'reward-curve.json', dict(iteration=[x.step for x in rewards], reward=[x.value for x in rewards]))
    shutil.copy2(__file__, out / 'evaluation-source.py')
    for batch in ('main', 'repeat'):
        folder = out / batch; core.ARMS = tuple(g['id'] for g in groups(batch))
        core.prepare(folder, 'model_comparison')
        shutil.copy2(NEW_RUN / 'model_final.pt', folder / 'model_7500.pt')
        plan = json.loads((folder / 'plan.json').read_text())
        plan.update(groups=groups(batch), intervention='Two frozen policies; matched current/five-frame full sensor noise, only gyroX amplitude differs old/new',
            noise_scales=dict(clean=[0.] * 25, old=baseline['normalization']['noise_scales'], new=target['normalization']['noise_scales']),
            model_sha256={'6838': core.SHA, '7500': NEW_SHA},
            scope='Full47 plus6 height cases, one noise stream' if batch == 'main' else 'Only zero/parking/height cases, two additional noise streams',
            limitations='Nominal physics, no pushes/external load; this does not identify or reproduce real mechanical11Hz',
            seed_pairing='Within each noise seed: same uniform draws across policies and old/new amplitudes')
        frozen.write(folder / 'plan.json', plan)


def worker(out, batch):
    folder = out / batch; definitions = groups(batch)
    core.ARMS = tuple(g['id'] for g in definitions)
    plan = json.loads((folder / 'plan.json').read_text()); created = []

    def factory(baseline, meta, n):
        import torch
        from ct_mjlab.fudan_ppo_walk.policy import load_policy
        candidate, other = load_policy(folder / 'model_7500.pt', 'cuda:0')
        for key in ('control', 'control_timing', 'disturbances', 'physical_wheel_model', 'reset_contract',
                    'height_reward', 'posture_reward', 'command_sampling'):
            assert meta.get(key) == other.get(key), key

        class PairedPolicies(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.baseline = baseline; self.candidate = candidate
                self.tick = 0; self.ring = None; self.records = []; self.uniforms = {}
                self.generators = {s: torch.Generator(device='cuda:0').manual_seed(s) for s in {g['seed'] for g in definitions}}
                self.errors = dict(current=0., history=0., clean=0.)
                self.protocol_overrides = {g['id']: dict(checkpoint_sha256=plan['model_sha256'][str(g['model'])],
                    observation_noise=dict(seed=g['seed'], scales=plan['noise_scales'][g['noise']],
                                           applied='After sensor delay, before current clipping; history retains noisy frames'),
                    inference='Frozen deterministic policy mean; no PPO action exploration') for g in definitions}

            def act_inference(self, obs, history):
                # 一个噪声样本同时给两个模型；old/new仅gyroX幅度相差两倍。
                draws = {s: torch.rand((n, 25), device=obs.device, generator=r) * 2 - 1 for s, r in self.generators.items()}
                noise = torch.cat([draws[g['seed']] * obs.new_tensor(plan['noise_scales'][g['noise']]) for g in definitions])
                if self.ring is None:self.ring = noise[:, None].repeat(1, 5, 1)
                else:self.ring = torch.cat((self.ring[:, 1:], noise[:, None]), 1)
                raw_history = history.view(-1, 5, 25)
                assert torch.equal(raw_history[:, -1].clamp(-100., 100.), obs)
                corrupted = (raw_history[:, -1] + noise).clamp(-100., 100.)
                noisy_history = (raw_history + self.ring).reshape(-1, 125)
                for i, g in enumerate(definitions):
                    if g['noise'] == 'clean':
                        take = slice(i*n, (i+1)*n)
                        assert torch.equal(corrupted[take], obs[take]) and torch.equal(noisy_history[take], history[take])
                if self.tick % 100 == 0:
                    self.errors['history'] = max(self.errors['history'], float(abs((noisy_history.view(-1,5,25) - raw_history) - self.ring).max()))
                    assert self.errors['history'] < 2e-5
                actions = obs.new_empty((len(obs), 6)); latent = obs.new_empty((len(obs), 3))
                for model, policy in ((6838, self.baseline), (7500, self.candidate)):
                    ids = torch.cat([torch.arange(i*n, (i+1)*n, device=obs.device) for i,g in enumerate(definitions) if g['model'] == model])
                    a, e = policy.act_inference(corrupted[ids], noisy_history[ids]); actions[ids] = a; latent[ids] = e
                self.records.append(torch.stack((obs[:,0], corrupted[:,0], noisy_history[:,-25]), -1).cpu().numpy())
                for s, u in draws.items():self.uniforms.setdefault(s, []).append(u.cpu().numpy())
                self.tick += 1
                return actions, latent
        result = PairedPolicies(); created.append(result)
        return result

    selector = None if batch == 'main' else lambda s: s.get('diagnostic') or s['case']['category'] == 'zero' or s['case']['category'].startswith('stop_')
    core.worker(folder, policy_factory=factory, sample_filter=selector)
    policy = created[0]
    np.savez_compressed(folder / 'policy-noise-record.npz', values=np.asarray(policy.records),
                        **{f'uniform_{s}': np.asarray(v) for s,v in policy.uniforms.items()})
    assert policy.tick == 3000 and frozen.sha(folder / 'model_7500.pt') == NEW_SHA
    frozen.write(folder / 'policy-noise-audit.json', dict(passed=True, ticks=policy.tick,
        independent_seeds=sorted(policy.uniforms), same_draws_across_policies_and_amplitudes=True,
        current_and_history_clipping='Current clipped100; history noisy before clip (originalv3)',
        maximum_injection_error=policy.errors, weights_unchanged=True, optimizer_updates=0))


def report(out):
    from scripts.agent.motion.acceptance import compare_baseline
    totals = {}; transients = []; parking = []
    for batch in ('main', 'repeat'):
        folder = out / batch; definitions = groups(batch)
        items = json.loads((folder / 'samples.json').read_text())['all_cases']
        accepted = {g['id']: json.loads((folder / g['id'] / 'acceptance.json').read_text()) for g in definitions}
        if batch == 'main':
            clean = compare_baseline(accepted['candidate_clean'], accepted['source_clean'])
            sham = compare_baseline(accepted['source_sham'], accepted['source_clean'])
            frozen.write(out / 'clean-comparison.json', clean); frozen.write(out / 'sham-comparison.json', sham)
        for g in definitions:
            score = accepted[g['id']]; key = g['id']
            totals[key] = dict(passed=score['technical_passed'], total=score['total'], groups=score['groups'])
            if g['model'] == 7500:
                source_id = key.replace('m7500_', 'm6838_')
                if source_id in accepted:
                    comparison = compare_baseline(score, accepted[source_id])
                    totals[key]['comparison'] = comparison['comparison']
            path = folder / g['id']
            with np.load(path / 'motor-pd.npz') as saved:
                t = saved['time']; motor = saved['values']
                for i,s in enumerate(items):
                    if not s.get('diagnostic') and s['case']['category'] != 'zero':continue
                    end=s['case']['duration']; lo,hi=(8.37,13.) if s.get('diagnostic') else (end-5,end)
                    m=(t>=lo)&(t<hi)
                    with np.load(path / 'traces' / (s['id']+'.npz')) as tr:
                        mask=(tr['time']>=lo)&(tr['time']<hi)
                        row=dict(batch=batch, arm=g['id'], model=g['model'], noise=g['noise'], noise_seed=g['seed'],
                            sample_id=s['id'], kind=s.get('diagnostic','zero30'),
                            gyroX_true_band8to14=float(core.band(tr['angular_body'][mask,0],1000.)),
                            LR_target_band8to14=float(core.band(motor[m,i,0,1],200.)),
                            LR_torque_band8to14=float(core.band(motor[m,i,3,1],200.)),
                            horizontal_speed_peak=float(np.linalg.norm(tr['linear_world'][mask,:2],axis=1).max()),
                            pitch_peak_deg=float(abs(tr['roll_pitch_deg'][mask,1]).max()),
                            complete=not bool(tr['terminated'].any() or tr['numerical_failure'].any()))
                        transients.append(row)
            for row in score['samples']:
                if row['case']['category'].startswith('stop_') or row['case']['category']=='zero':
                    with np.load(path/'traces'/(row['sample_id']+'.npz')) as tr:
                        end=row['case']['duration']; last=(tr['time']>=end-5)&(tr['time']<end)
                        fixed=dict(window=[end-5,end],horizontal_peak=float(np.linalg.norm(tr['linear_world'][last,:2],axis=1).max()),
                                   pitch_peak_deg=float(abs(tr['roll_pitch_deg'][last,1]).max()))
                    parking.append(dict(model=g['model'],noise=g['noise'],seed=g['seed'],batch=batch,
                                        sample_id=row['sample_id'],passed=row['passed'],phases=row.get('motion_phases'),
                                        fixed_last5s=fixed))
    frozen.write(out / 'comparison.json', dict(scores=totals, high_stance=transients, parking=parking,
        optimizer_updates=0, limitations='Noise streams2/3 only zero/parking/high-stance; flat47 subset, not full987 or terrain acceptance; GPU did not reproduce real11Hz'))
    plot(out, transients)


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for arm in ('source_clean','candidate_clean','m6838_new_s71001','m7500_new_s71001'):
        folder=out/'main'/arm
        with np.load(folder/'motor-pd.npz') as d:
            items=json.loads((out/'main/samples.json').read_text())['all_cases']
            i=next(i for i,s in enumerate(items) if s['id']=='height_replay__50001');t=d['time'];mask=t<15
            axes[0].plot(t[mask],d['values'][mask,i,0,1],lw=.7,label=arm)
        with np.load(folder/'traces/height_replay__50001.npz') as tr:
            axes[1].plot(tr['time'],tr['angular_body'][:,0],lw=.7,label=arm)
            axes[2].plot(tr['time'],np.linalg.norm(tr['linear_world'][:,:2],axis=1),lw=.7,label=arm)
    for ax,label in zip(axes,('LR target (rad)','True gyroX (rad/s)','Horizontal speed (m/s)')):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);ax.axvspan(7.79,8.37,color='orange',alpha=.15)
    axes[-1].set_xlabel('Simulated seconds');fig.suptitle('Frozen6838 vs7500: identical height schedule and matched sensor noise')
    fig.tight_layout();fig.savefig(out/'high-stance-comparison.png',dpi=150);plt.close(fig)


def render(out, sid, noise):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    folder=out/'main'; definitions=groups('main')
    names=('source_clean','candidate_clean') if noise=='clean' else ('m6838_new_s71001','m7500_new_s71001')
    items=json.loads((folder/'samples.json').read_text())['all_cases'];index=next(i for i,s in enumerate(items) if s['id']==sid)
    sample=items[index];sources=[];traces=[]
    for arm in names:
        path=folder/arm/'traces'/(sid+'.npz')
        with np.load(path) as z:traces.append({k:z[k].copy() for k in z.files})
        sources.append(dict(arm=arm,trajectory=str(path),sha256=frozen.sha(path)))
    model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=480;model.vis.global_.offheight=368
    data=[mujoco.MjData(model),mujoco.MjData(model)]
    cameras=[mujoco.MjvCamera(),mujoco.MjvCamera()]
    for cam in cameras:cam.distance=1.6;cam.azimuth=90.;cam.elevation=-15.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    videos=out/'videos';videos.mkdir(exist_ok=True);path=videos/(noise+'__'+sid+'.mp4')
    assert not path.exists()
    frames=0
    with mujoco.Renderer(model,width=480,height=368) as renderer, imageio.get_writer(path,fps=50,codec='libx264',quality=8) as writer:
        for i in range(0,len(traces[0]['qpos'])-1,2):
            canvas=Image.new('RGB',(960,448),'#18212b');draw=ImageDraw.Draw(canvas)
            draw.text((12,8),f'1x NORMAL | {sid} | gyroX noise {noise} | no external load',font=title,fill='white')
            centre=np.mean([tr['qpos'][i,:2] for tr in traces],axis=0)
            for col,tr in enumerate(traces):
                data[col].qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(model,data[col]);mujoco.mj_comPos(model,data[col])
                cameras[col].lookat[:]=[centre[0],centre[1],.20]
                renderer.update_scene(data[col],camera=cameras[col],scene_option=option)
                panel=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(panel)
                tick=min(len(tr['time'])-1,round(tr['qpos_time'][i]/.001))
                speed=np.linalg.norm(tr['linear_world'][tick,:2]);drift=np.linalg.norm(tr['qpos'][i,:2]-tr['qpos'][0,:2])
                d.rectangle((0,0,480,91),fill='#18212b')
                d.text((8,4),f'{6838 if col==0 else 7500} | P20/D1 | t={tr["qpos_time"][i]:.2f}s',font=title,fill='white')
                d.text((8,31),f'speed={speed:.3f}m/s | XY drift={drift:.3f}m',font=font,fill='white')
                d.text((8,56),f'TRUE gyroX={tr["angular_body"][tick,0]:+.3f}rad/s',font=font,fill='white')
                canvas.paste(panel,(col*480,80))
            writer.append_data(np.asarray(canvas));frames+=1
            if i==round((sample['case']['duration']-1)*100):imageio.imwrite(path.with_suffix('.png'),np.asarray(canvas))
    assert frames==round(sample['case']['duration']*50)
    path.with_suffix('.json').write_text(json.dumps(dict(playback_speed=1,fps=50,frames=frames,duration_s=sample['case']['duration'],
        sources=sources,video_sha256=frozen.sha(path),physics_steps=0,policy_forwards=0,
        note='Saved GPU poses only; common horizontal-follow camera and ground pattern show relative motion'),indent=2)+'\n')


def verify(out):
    import ast
    import imageio.v2 as imageio
    import torch
    original=ast.parse((out/'evaluation-source.py').read_text());current=ast.parse(Path(__file__).read_text())
    for name in ('worker','groups'):
        old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(old)==ast.dump(new),name
    learned=torch.load(NEW_RUN/'model_final.pt',map_location='cpu',weights_only=False)
    assert frozen.sha(NEW_RUN/'model_final.pt')==NEW_SHA
    assert learned['iterations']==7500 and not learned['partial_update']
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in learned[key]['state'].values()}=={150000}
        assert all(torch.isfinite(v).all() for row in learned[key]['state'].values() for v in row.values() if isinstance(v,torch.Tensor))
    results=[]
    for batch in ('main','repeat'):
        folder=out/batch;runtime=json.loads((folder/'runtime-audit.json').read_text())
        assert runtime['passed'] and runtime['physics_steps']==30000 and runtime['optimizer_updates']==0
        arrays=0
        with np.load(folder/'same-initial-state.npz') as z:
            for key in z.files:
                value=z[key];assert all(np.array_equal(value[0],v) for v in value[1:]),key;arrays+=1
        with np.load(folder/'policy-noise-record.npz') as z:
            values=z['values'];n=values.shape[1]//len(groups(batch));error=0.;cliperror=0.
            for i,g in enumerate(groups(batch)):
                a=values[:,i*n:(i+1)*n];scale=0. if g['noise']=='clean' else (.075 if g['noise']=='old' else .15)
                expected=z['uniform_'+str(g['seed'])][:,:,0]*scale
                mask=abs(a[:,:,0])<99.5
                error=max(error,float(abs((a[:,:,1]-a[:,:,0]-expected)[mask]).max()))
                cliperror=max(cliperror,float(abs(a[:,:,1]-a[:,:,2].clip(-100,100)).max()))
            assert error<2e-5 and cliperror==0.
        cohorts=[]
        for g in groups(batch):
            arm=folder/g['id'];a=json.loads((arm/'acceptance.json').read_text())
            assert a['details']['source_protocol_sha256']==frozen.sha(arm/'protocol.json')
            cohorts.append(a['cohort_sha256'])
        assert len(set(cohorts))==1
        results.append(dict(batch=batch,initial_state_arrays_equal=arrays,cohort_sha256=cohorts[0],
                            gyroX_injection_error=error,current_matches_noisy_history_last_error=cliperror,
                            worlds=runtime['worlds'],PD_error_Nm=runtime['PD_error_nm'],friction_error=runtime['friction_max_error']))
    videos=[]
    for path in sorted((out/'videos').glob('*.mp4')):
        meta=json.loads(path.with_suffix('.json').read_text());assert frozen.sha(path)==meta['video_sha256']
        reader=imageio.get_reader(path);container=reader.get_meta_data();count=sum(1 for _ in reader);reader.close()
        assert count==meta['frames'] and meta['fps']==container['fps']==50 and meta['playback_speed']==1
        assert abs(container['duration']-meta['duration_s'])<.05
        videos.append(dict(path=str(path),decoded_frames=count,fps=container['fps'],duration_s=container['duration'],sha256=meta['video_sha256']))
    assert len(videos)>=2
    shutil.copy2(__file__,out/'report-render-source.py')
    frozen.write(out/'verification.json',dict(passed=True,executed_worker_and_group_AST_unchanged=True,
        final_optimizer_states_finite=True,models_unchanged=True,batches=results,videos=videos,
        optimizer_updates=0,physics_steps=0,scope='Offline verification of savedGPU/noise/state/protocol/video evidence'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('run','worker','report','render','verify'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',choices=('main','repeat'),default='main')
    p.add_argument('--sample',default='height_replay__50001');p.add_argument('--noise',choices=('clean','new'),default='new')
    a=p.parse_args();out=a.output.resolve()
    if a.action=='worker':return worker(out,a.batch)
    if a.action=='report':return report(out)
    if a.action=='render':return render(out,a.sample,a.noise)
    if a.action=='verify':return verify(out)
    prepare(out)
    try:
        for batch in ('main','repeat'):
            run_command(out,batch,[str(core.study.PYTHON),'-u','-m',MODULE,'worker','--output',str(out),'--batch',batch])
        report(out);frozen.write(out/'status.json',dict(status='completed',optimizer_updates=0))
    except BaseException as e:
        frozen.write(out/'status.json',dict(status='failed',error=repr(e),automatic_retry=False));raise


if __name__=='__main__':main()
