"""Aligned actor observations and student-PD targets; same RSL actor as PPO."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .smp.dataset import MotionDataset
from .smp.features import inverse_rotate
from .wheel_odometry import WHEEL_OBSERVATION_CONTRACT, WHEEL_ODOMETRY, wheel_displacement
from . import reference_contract as ref_contract
from . import wheel_angles

PAIR_SCHEMA = 'chuanliantui_actor39_pd6_prestep_v1'
FORMAT = 'ct_bc_actor39_pd6_v1'
WHEEL_PAIR_SCHEMA = 'chuanliantui_actor40_pd6_prestep_wheel_v1'
WHEEL_FORMAT = 'ct_bc_actor40_pd6_wheel_v1'
ORDER = [0, 1, 3, 4, 2, 5]
LEGACY_SAMPLING = 'rise-standing-equal-v1'
BRAKING_SAMPLING = 'rise-brake-turn-stop-v2'
PRESERVE_SAMPLING = 'preserve-rise-brake-stop-v1'
PRESERVE_WEIGHTS = {'standing':.25,'braking':.5,'parked':.25}
BRAKING_WEIGHTS = {'rise': .5, 'standing': .125, 'braking': .125, 'turn_correction': .125, 'parked': .125}
LABEL_REFERENCE = dict(schema='ct_bc_parking_reference_v1',
    normal_handover='zero_position_error_and_anchor_current_yaw_once',
    takeover='same_anchor_once_after_continuous_wheel_support_1s',
    continuation='continuous_teacher_history_with_original_air_and_standup_resets',
    labels='actual_teacher_control_only_no_shadow_queries')
ALIGNED_LABEL_REFERENCE = dict(LABEL_REFERENCE, schema='ct_bc_shared_wheel_position_reference_v2',
    normal_handover='first_ground_start_recovered_shared_with_wheel_origin_before_lqr_feedback',
    takeover='phase_sync_keeps_shared_position_origin',
    continuation='freeze_position_before_shared_event_then_integrate_without_internal_position_relocks')


def check_label_reference(manifests, reports, *, required=False):
    sources = [*manifests, *reports]
    references = [source.get('label_reference') for source in sources]
    if not any(reference is not None for reference in references):
        if required:
            raise ValueError('BC training requires unified label_reference manifests; legacy data remain readable for replay')
        return None
    if all(reference==ref_contract.REFERENCE for reference in references):
        return ref_contract.REFERENCE
    if all(reference==ALIGNED_LABEL_REFERENCE for reference in references):
        if any(report.get('teacher_queries_only') is not False for report in reports):
            raise ValueError('Shared wheel reference requires actual teacher control')
        if any(source.get('observation_contract')!=wheel_angles.HANDOVER_OBSERVATION_CONTRACT for source in sources):
            raise ValueError('Shared position reference requires handover wheel observations')
        return ALIGNED_LABEL_REFERENCE
    if any(reference != LABEL_REFERENCE for reference in references):
        raise ValueError('Mixed or unknown BC parking label references')
    if any(report.get('teacher_queries_only') is not False for report in reports):
        raise ValueError('Unified parking labels require actual teacher control, not shadow queries')
    return LABEL_REFERENCE


def build_actor(device='cpu', *, observation_dim=39):
    obs = TensorDict({'actor': torch.zeros(1, observation_dim, device=device)}, batch_size=[1])
    return MLPModel(obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(256, 128, 64),
        activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .05, 'std_type': 'scalar'}).to(device)


def expand_actor(actor, observation_dim, device='cpu'):
    old=actor.state_dict();expanded=build_actor(device,observation_dim=observation_dim).eval()
    state=expanded.state_dict();changed=[]
    for name,value in old.items():
        if value.shape==state[name].shape:
            state[name].copy_(value)
        else:
            if value.ndim!=2 or value.shape[0]!=256 or value.shape[1]>=observation_dim or state[name].shape!=(256,observation_dim):
                raise ValueError('Only appending zero first-layer observation weights is supported')
            state[name].zero_();state[name][:,:value.shape[1]].copy_(value);changed.append(name)
    if len(changed)!=1:
        raise ValueError('Explicit observation expansion requires exactly one enlarged input layer')
    expanded.load_state_dict(state,strict=True)
    return expanded


class ActionDataset:
    def __init__(self, paths, corrections=(), *, action_sampling=LEGACY_SAMPLING, require_reference=False):
        if action_sampling not in (LEGACY_SAMPLING, BRAKING_SAMPLING, PRESERVE_SAMPLING):
            raise ValueError('Unknown BC action sampling profile')
        self.action_sampling = action_sampling
        manifests = [json.loads(Path(p).read_text()) for p in paths]
        reports = [json.loads((Path(p)/'result.json').read_text()) for p in corrections]
        reference = check_label_reference(manifests, reports, required=require_reference)
        self.motion = MotionDataset(paths)
        pair_schema = manifests[0].get('action_pair_schema')
        self.observation_dim = {PAIR_SCHEMA:39,WHEEL_PAIR_SCHEMA:40,ref_contract.PAIR_SCHEMA:41,
                                wheel_angles.PAIR_SCHEMA:39,wheel_angles.HANDOVER_PAIR_SCHEMA:39}.get(pair_schema)
        if self.observation_dim is None:
            raise ValueError('Unknown BC observation/action schema')
        angle_profile=pair_schema in (wheel_angles.PAIR_SCHEMA,wheel_angles.HANDOVER_PAIR_SCHEMA)
        handover_profile=pair_schema==wheel_angles.HANDOVER_PAIR_SCHEMA
        angle_contract=wheel_angles.HANDOVER_OBSERVATION_CONTRACT if handover_profile else wheel_angles.OBSERVATION_CONTRACT
        angle_definition=wheel_angles.PROFILES[angle_contract]
        if angle_profile and any(m.get('observation_contract')!=angle_contract or
                m.get('wheel_angles')!=angle_definition['wheel_angles'] for m in [*manifests,*reports]):
            raise ValueError('Mixed reset-relative wheel-angle observation contracts')
        if self.observation_dim == 40 and any(m.get('observation_contract') != WHEEL_OBSERVATION_CONTRACT or
                m.get('wheel_odometry') != WHEEL_ODOMETRY for m in [*manifests, *reports]):
            raise ValueError('Mixed wheel displacement observation contracts')
        if self.observation_dim==41 and any(m.get('observation_contract')!=ref_contract.OBSERVATION_CONTRACT or
                m.get('wheel_odometry')!=WHEEL_ODOMETRY or m.get('label_reference')!=ref_contract.REFERENCE
                for m in [*manifests,*reports]):
            raise ValueError('Mixed shared reference observation contracts')
        if any(m.get('action_pair_schema') != pair_schema or m.get('observation_noise') or
               m.get('observation_delay') for m in manifests):
            raise ValueError('BC requires explicitly aligned, nominal actor observations')
        self.description = self.motion.description()
        self.description.update(pair_schema=pair_schema, observation_contract=manifests[0]['observation_contract'],
            actor_hidden_dims=[256, 128, 64], activation='elu', action_std=.05,
            action_order=manifests[0]['motor_order'], physics_dt=.002, action_dt=.01,
            control_sources={k: v for k, v in manifests[0]['source_sha256'].items()
                             if k.endswith(('pd_action.py', 'huananhu_action.py'))})
        if any(m['observation_contract'] != self.description['observation_contract'] or
               {k: v for k, v in m['source_sha256'].items() if k in self.description['control_sources']}
               != self.description['control_sources'] for m in manifests):
            raise ValueError('Mixed BC observation or control definitions')
        self.frames, self.classes = {}, {}
        for split, episodes in self.motion.splits.items():
            inputs, targets, classes, offset = [], [], {'rise': [], 'standing': []}, 0
            for e in episodes:
                with np.load(e['path']) as d:
                    o, a = torch.from_numpy(d['observation'].copy()), torch.from_numpy(d['action'].copy())
                    q, dq = torch.from_numpy(d['source_motor_q'].copy()), torch.from_numpy(d['source_motor_dq'].copy())
                    if o.shape != (len(a), self.observation_dim) or a.shape != (len(o), 6) or not torch.isfinite(o).all() or not torch.isfinite(a).all():
                        raise ValueError('Invalid BC observation/action shape or values')
                    np.testing.assert_allclose(d['observation_time'], d['action_time'], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(d['observation_time'], d['time']-.01, atol=1e-8, rtol=0)
                    if self.observation_dim >= 40:
                        torch.testing.assert_close(o[:, 39:40], wheel_displacement(q[:, [2,5]], q[0, [2,5]]), atol=2e-6, rtol=2e-6)
                    if self.observation_dim==41:
                        check_reference_pairs(d,o.numpy())
                    if reference==ALIGNED_LABEL_REFERENCE:
                        check_aligned_position_pairs(d)
                    # 核对观测和目标均对应步进前：上一拍动作、角度、角速度和根姿态。
                    torch.testing.assert_close(o[1:, 22:28], a[:-1, ORDER].clamp(-100, 100), atol=2e-6, rtol=2e-6)
                    assert not o[0, 22:28].any()
                    torch.testing.assert_close(q[1:], torch.from_numpy(d['motor_q'][:-1]), atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(dq[1:], torch.from_numpy(d['motor_dq'][:-1]), atol=2e-6, rtol=2e-6)
                    nominal = torch.from_numpy(d['nominal_motor_positions'].copy())
                    position = (q-nominal)[:, ORDER]
                    position[:, 4:] = (torch.from_numpy(check_handover_wheel_pairs(d,q.numpy(),o.numpy())) if handover_profile else
                        q[:,[2,5]]-q[0,[2,5]] if angle_profile else 0.)
                    torch.testing.assert_close(o[:, 10:16], position.clamp(-100, 100), atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(o[:, 16:22], dq[:, ORDER].clamp(-200, 200)*.1, atol=2e-6, rtol=2e-6)
                    pre_qpos = torch.cat((torch.from_numpy(d['initial_qpos'].copy())[None], torch.from_numpy(d['qpos'][:-1].copy())))
                    pre_qvel = torch.cat((torch.from_numpy(d['initial_qvel'].copy())[None], torch.from_numpy(d['qvel'][:-1].copy())))
                    gravity = torch.zeros(len(o), 3); gravity[:, 2] = -1
                    torch.testing.assert_close(o[:, 7:10], inverse_rotate(pre_qpos[:, 3:7], gravity), atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(o[:, 4:7], pre_qvel[:, 3:6].clamp(-100, 100)*.5, atol=2e-6, rtol=2e-6)
                    time = torch.from_numpy(d['observation_time'].copy())
                    indices = torch.arange(len(o))+offset
                    classes['rise'].append(indices[time < 2.])
                    classes['standing'].append(indices[time >= 2.])
                    inputs.append(o); targets.append(a); offset += len(o)
            self.frames[split] = (torch.cat(inputs), torch.cat(targets))
            self.classes[split] = {k: torch.cat(v) for k, v in classes.items()}
        self.description['fingerprint'] = hashlib.sha256(json.dumps(dict(
            motion=self.motion.fingerprint, pair_schema=pair_schema, sampling='rise-standing-equal-v1'), sort_keys=True).encode()).hexdigest()
        self.description['motion_fingerprint'] = self.motion.fingerprint
        self.description['class_frame_counts'] = {s: {k: len(v) for k, v in c.items()} for s, c in self.classes.items()}
        correction_sources = []
        braking_classes = {name: [] for name in ('braking', 'turn_correction', 'parked')}
        train_groups = {e['group_id'] for e in self.motion.splits['train']}
        if reference==ALIGNED_LABEL_REFERENCE:
            # 未收录完整停车示范的原训练姿态仍可提供合格的实际接管纠偏。
            train_groups=set(manifests[0]['group_split']['train'])
        for directory, report in zip(corrections, reports):
            directory = Path(directory)
            path = directory/'trace.npz'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (report.get('correction_schema') != 'ct_bc_corrections_v1' or report['split'] != 'train' or
                    report.get('base_motion_fingerprint') != self.motion.fingerprint or
                    report['xml_sha256'] != self.motion.xml_sha256 or digest != report['trace_sha256'] or
                    not all(row['group_id'] in train_groups for row in report['rows'])):
                raise ValueError('Corrections must use matching training poses, never heldout poses')
            with np.load(path) as d:
                valid = d['label_valid'] & d['alive']
                if reference==ALIGNED_LABEL_REFERENCE:
                    check_aligned_position_pairs(d,d['alive'])
                obs, target = torch.from_numpy(d['observation'][valid].copy()), torch.from_numpy(d['label'][valid].copy())
                if obs.shape != (len(target), self.observation_dim) or target.shape != (len(obs), 6):
                    raise ValueError('Correction observation/action dimensions differ')
                if self.observation_dim >= 40:
                    q = torch.from_numpy(d['source_q'].copy())
                    expected = wheel_displacement(q[..., [2,5]], q[0][..., [2,5]])
                    np.testing.assert_allclose(d['observation'][..., 39:40][valid], expected.numpy()[valid], atol=2e-6, rtol=2e-6)
                if self.observation_dim==41:
                    check_reference_pairs(d,d['observation'],valid)
                if angle_profile:
                    q=d['source_q'];expected=(check_handover_wheel_pairs(d,q,d['observation'],d['alive']).clip(-100,100)
                        if handover_profile else np.clip(q[...,[2,5]]-q[0][...,[2,5]],-100,100))
                    np.testing.assert_allclose(d['observation'][...,14:16][valid],expected[valid],atol=2e-6,rtol=2e-6)
                if not len(obs) or not torch.isfinite(obs).all() or not torch.isfinite(target).all():
                    raise ValueError('No finite, valid teacher correction labels')
                times = torch.from_numpy(np.broadcast_to((d['time']-.01)[:, None], valid.shape)[valid].copy())
                braking = action_sampling in (BRAKING_SAMPLING,PRESERVE_SAMPLING) and report.get('teacher_queries_only') is False
                if braking or (reference is not None and report.get('teacher_queries_only') is False):
                    qualified = np.array([r['qualified'] for r in report['rows']])
                    allowed=np.broadcast_to(qualified,valid.shape)
                    if angle_profile and report.get('label_scope')=='qualified_full_recoveries_plus_rise_only_v1':
                        rise_qualified=np.array([r['rise_qualified'] for r in report['rows']])
                        # 未通过停车的轨迹仅可提供两秒内的已验收起立动作。
                        allowed=allowed|((d['time']<=2.)[:,None]&rise_qualified[None,:])
                    if (not allowed.any() or (valid&~allowed).any() or
                            valid[d['time'] <= report['takeover_s']].any() or
                            not d['teacher_selected'][valid].all() or
                            not ((d['teacher_phase'][valid] != 4).all() if angle_profile and not braking else
                                 (d['teacher_phase'][valid] == 3).all())):
                        raise ValueError('Corrections require screened, actual teacher-controlled labels')
                    np.testing.assert_array_equal(d['applied_action'][valid], target.numpy())
                    if d['teacher_fault'][valid].any():
                        raise ValueError('Faulted teacher frames in unified parking labels')
                if braking:
                    features = d['features']
                    # 标签对应步进前状态，不能按本拍动作执行后的速度分组。
                    pre = np.concatenate((np.zeros_like(features[:1]), features[:-1]))[valid]
                    turn = np.abs(pre[:, 9]) > .02
                    parked = ((np.linalg.norm(pre[:, 4:6], axis=-1) <= .01) & ~turn &
                        (np.sqrt(np.square(pre[:, [18,19,21,22]]).mean(-1)) <= .02) &
                        (np.abs(pre[:, 0]-.33648) <= .03) & (pre[:, 3] < -.97))
                    masks = {'braking': ~turn & ~parked, 'turn_correction': turn, 'parked': parked}
                    if action_sampling==PRESERVE_SAMPLING:masks={'braking':~parked,'parked':parked}
            offset = len(self.frames['train'][0])
            self.frames['train'] = tuple(torch.cat((a, b)) for a, b in zip(self.frames['train'], (obs, target)))
            indices = torch.arange(len(obs))+offset
            if braking:
                for name, mask in masks.items():
                    braking_classes[name].append(indices[torch.from_numpy(mask)])
            else:
                for name, mask in (('rise', times < 2.), ('standing', times >= 2.)):
                    self.classes['train'][name] = torch.cat((self.classes['train'][name], indices[mask]))
            correction_sources.append(dict(trace_sha256=digest, source_actor_sha256=report['checkpoint_sha256'], frames=len(obs)))
        if correction_sources:
            self.description['corrections'] = correction_sources
            self.description['fingerprint'] = hashlib.sha256(json.dumps(dict(
                base=self.description['fingerprint'], corrections=correction_sources), sort_keys=True).encode()).hexdigest()
            self.description['class_frame_counts']['train'] = {k: len(v) for k, v in self.classes['train'].items()}
        if action_sampling == BRAKING_SAMPLING:
            if any(not chunks for chunks in braking_classes.values()):
                raise ValueError('Braking profile requires qualified braking demonstrations')
            self.classes['train'].update({k: torch.cat(v) for k, v in braking_classes.items()})
            if any(not len(v) for v in self.classes['train'].values()):
                raise ValueError('Braking sampling class is empty')
            self.description.update(action_sampling=action_sampling, train_class_weights=BRAKING_WEIGHTS,
                braking_classification='pre-step: abs(yaw)>0.02=>turn; otherwise speed<=0.01, legRMS<=0.02, heightError<=0.03, gravityZ<-0.97=>parked; remaining=>braking')
            self.description['class_frame_counts']['train'] = {k: len(v) for k, v in self.classes['train'].items()}
            self.description['fingerprint'] = hashlib.sha256(json.dumps(dict(
                base=self.description['fingerprint'], action_sampling=action_sampling,
                weights=BRAKING_WEIGHTS, classification=self.description['braking_classification']), sort_keys=True).encode()).hexdigest()
        if action_sampling==PRESERVE_SAMPLING:
            if not angle_profile or any(not braking_classes[k] for k in ('braking','parked')):
                raise ValueError('Preserved wheel39 training requires actual qualified braking and parked labels')
            self.classes['train'].update({k:torch.cat(braking_classes[k]) for k in ('braking','parked')})
            if any(not len(self.classes['train'][k]) for k in PRESERVE_WEIGHTS):
                raise ValueError('Preserved training class is empty')
            self.description.update(action_sampling=PRESERVE_SAMPLING,train_class_weights=PRESERVE_WEIGHTS,
                rise_labels='separate frozen successful-actor preservation; teacher rise excluded from expert sampling')
            self.description['class_frame_counts']['train']={k:len(v) for k,v in self.classes['train'].items()}
            self.description['fingerprint']=hashlib.sha256(json.dumps(dict(base=self.description['fingerprint'],
                action_sampling=PRESERVE_SAMPLING,weights=PRESERVE_WEIGHTS),sort_keys=True).encode()).hexdigest()
        if reference is not None:
            self.description['label_reference'] = reference
            self.description['fingerprint'] = hashlib.sha256(json.dumps(dict(
                base=self.description['fingerprint'], label_reference=reference), sort_keys=True).encode()).hexdigest()
        if self.observation_dim >= 40:
            self.description.update(observation_dim=self.observation_dim, wheel_odometry=WHEEL_ODOMETRY)
        if self.observation_dim==41:
            self.description['group_split']=manifests[0]['group_split']
        if angle_profile:
            self.description.update(observation_dim=39,wheel_angles=angle_definition['wheel_angles'],
                                    group_split=manifests[0]['group_split'])

    def sample(self, split, size, generator):
        pools = list(self.classes[split].values())
        if split=='train' and self.action_sampling==PRESERVE_SAMPLING:
            pools=[self.classes[split][k] for k in PRESERVE_WEIGHTS]
            labels=torch.multinomial(torch.tensor(list(PRESERVE_WEIGHTS.values())),size,replacement=True,generator=generator)
        elif split == 'train' and self.action_sampling == BRAKING_SAMPLING:
            weights = torch.tensor([BRAKING_WEIGHTS[k] for k in self.classes[split]])
            labels = torch.multinomial(weights, size, replacement=True, generator=generator)
        else:
            labels = torch.randint(2, (size,), generator=generator)
        indices = torch.empty(size, dtype=torch.long)
        for k, pool in enumerate(pools):
            mask = labels == k
            indices[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), generator=generator)]
        return tuple(values[indices] for values in self.frames[split])


def load_actor(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('format') not in (FORMAT,WHEEL_FORMAT,ref_contract.FORMAT,wheel_angles.FORMAT,wheel_angles.HANDOVER_FORMAT) or saved.get('iteration', 0) < 1:
        raise ValueError('Not a trained aligned BC actor')
    dim={FORMAT:39,WHEEL_FORMAT:40,ref_contract.FORMAT:41,wheel_angles.FORMAT:39,wheel_angles.HANDOVER_FORMAT:39}[saved['format']]
    if saved['format'] in (wheel_angles.FORMAT,wheel_angles.HANDOVER_FORMAT):
        profile=wheel_angles.PROFILES.get(saved['dataset']['observation_contract'])
        if profile is None or profile['model_format']!=saved['format'] or saved['dataset'].get('wheel_angles')!=profile['wheel_angles']:
            raise ValueError('Unknown wheel-angle checkpoint contract')
    if dim == 40 and (saved['dataset']['observation_contract'] != WHEEL_OBSERVATION_CONTRACT or
                     saved['dataset'].get('wheel_odometry') != WHEEL_ODOMETRY):
        raise ValueError('Unknown wheel displacement checkpoint contract')
    if dim==41 and (saved['dataset']['observation_contract']!=ref_contract.OBSERVATION_CONTRACT or
                   saved['dataset'].get('label_reference')!=ref_contract.REFERENCE):
        raise ValueError('Unknown shared reference checkpoint contract')
    actor = build_actor(device, observation_dim=dim)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    return actor.eval(), saved


def check_handover_wheel_pairs(data,q,observation,valid=None):
    origin=data['wheel_angle_origin'];latched=data['wheel_angle_handover_latched'];recovered=data['wheel_angle_recovered']
    wheels=q[...,[2,5]]
    if origin.shape!=wheels.shape or latched.shape!=q.shape[:-1] or recovered.shape!=latched.shape:
        raise ValueError('Invalid handover wheel reference arrays')
    mask=np.ones(latched.shape,dtype=bool) if valid is None else valid
    expected=np.maximum.accumulate(recovered,axis=0)
    np.testing.assert_array_equal(latched[mask],expected[mask])
    event=latched & ~np.concatenate((np.zeros_like(latched[:1]),latched[:-1]),axis=0)
    index=np.arange(len(q)).reshape((-1,)+(1,)*(latched.ndim-1))
    index=np.maximum.accumulate(np.where(event,index,0),axis=0)
    expected_origin=np.take_along_axis(wheels,np.broadcast_to(index[...,None],wheels.shape),axis=0)
    np.testing.assert_allclose(origin[mask],expected_origin[mask],atol=2e-6,rtol=2e-6)
    angles=wheels-origin
    np.testing.assert_allclose(observation[...,14:16][mask],angles.clip(-100,100)[mask],atol=2e-6,rtol=2e-6)
    return angles


def check_aligned_position_pairs(data,valid=None):
    latched=data['wheel_angle_handover_latched']
    event=latched & ~np.concatenate((np.zeros_like(latched[:1]),latched[:-1]),axis=0)
    mask=np.ones(latched.shape,dtype=bool) if valid is None else valid
    np.testing.assert_array_equal(data['lqr_position_reference_event'][mask],event[mask])
    integrating=data['lqr_position_integrating']
    assert not (integrating & ~latched)[mask].any()
    assert not data['lqr_position_error_used'][event & mask].any()
    assert not data['lqr_position_contribution'][event & mask].any()
    position=np.zeros_like(data['lqr_position_error_used'][0])
    for tick in range(len(latched)):
        position=np.clip(position+integrating[tick]*data['lqr_velocity_error_used'][tick]*.01,-3.5,3.5)
        position=np.where(event[tick],0.,position)
        np.testing.assert_allclose(np.asarray(data['lqr_position_error_used'][tick])[mask[tick]],
            np.asarray(position)[mask[tick]],atol=3e-6,rtol=3e-6)


def check_reference_pairs(data, observation, valid=None):
    state=data['reference_state'];error=data['reference_error']
    np.testing.assert_array_equal(observation[...,40:41],error)
    np.testing.assert_allclose(error[...,0],state[...,6]-state[...,5],atol=2e-6,rtol=2e-6)
    before=np.where(state[...,3]!=0,0.,state[...,0])
    expected=np.clip(before+state[...,2]*state[...,1]*.01,-3.5,3.5)
    expected=np.where(state[...,4]!=0,0.,expected)
    mask=np.ones(expected.shape,dtype=bool) if valid is None else valid
    np.testing.assert_allclose(state[...,5][mask],expected[mask],atol=2e-6,rtol=2e-6)
    if not np.isfinite(observation[mask]).all():
        raise ValueError('Nonfinite aligned reference pairs')
