"""Qualified complete episodes, with grouped splits and windows that never cross episodes."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .features import FEATURE_SCHEMA, FEATURE_DIM, SAMPLE_DT, WINDOW, motion_features


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior_name(initial, schedule):
    if initial == 'ground':
        return 'standup'
    vx,yaw = schedule[0]['vx'],schedule[0]['yaw']
    if vx and yaw:
        return 'moving_turn'
    if vx:
        return 'forward' if vx > 0 else 'backward'
    if yaw:
        return 'turn_left' if yaw > 0 else 'turn_right'
    return 'standing'


class MotionDataset:
    def __init__(self, manifests, *, validation_fraction=.2, split_seed=42):
        if not 0 < validation_fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        self.episodes = []
        self.xml_sha256 = None
        self.ground_start_hold_s = None
        identities, hashes = set(), []
        declared_splits=[]
        for manifest_path in manifests:
            manifest_path = Path(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            declared_splits.append(manifest.get('group_split'))
            if (manifest.get('schema') != 'chuanliantui_smp_episodes_v1' or
                    manifest.get('feature_schema') != FEATURE_SCHEMA or
                    manifest.get('feature_dim') != FEATURE_DIM or manifest.get('sample_dt') != SAMPLE_DT):
                raise ValueError(f'Incompatible motion dataset: {manifest_path}')
            if self.xml_sha256 is not None and self.xml_sha256 != manifest['xml_sha256']:
                raise ValueError('Cannot mix different robot XML versions')
            self.xml_sha256 = manifest['xml_sha256']
            for record in manifest['episodes']:
                if not record.get('accepted') or record.get('rejection_reasons'):
                    raise ValueError('Rejected episode in training manifest')
                identity = record['episode_id']
                if identity in identities:
                    raise ValueError('Duplicate episode (possible train/validation leakage)')
                identities.add(identity)
                path = (manifest_path.parent/record['path']).resolve()
                if not path.is_relative_to(manifest_path.parent.resolve()):
                    raise ValueError('Episode path escapes dataset directory')
                if file_hash(path) != record['sha256']:
                    raise ValueError(f'Episode hash mismatch: {path}')
                with np.load(path, allow_pickle=False) as data:
                    episode_metadata = json.loads(str(data['metadata']))
                    if (episode_metadata.get('episode_id') != identity or
                            episode_metadata.get('group_id') != record['group_id'] or
                            episode_metadata.get('xml_sha256') != self.xml_sha256 or
                            episode_metadata.get('feature_schema') != FEATURE_SCHEMA or
                            not episode_metadata.get('accepted')):
                        raise ValueError('Episode metadata disagrees with manifest')
                    time = data['time']
                    if len(time) < WINDOW or not np.allclose(np.diff(time), SAMPLE_DT, atol=1e-8, rtol=0):
                        raise ValueError('Episode is too short or not sampled at 100Hz')
                    arrays = {name: torch.from_numpy(data[name].copy()).float() for name in
                              ('root','motor_q','motor_dq','wheel_position_w','ground_height')}
                    features = motion_features(**arrays)
                    if features.shape != (len(time), FEATURE_DIM) or not torch.isfinite(features).all():
                        raise ValueError('Invalid motion features')
                    np.testing.assert_allclose(features.numpy(), data['features'], atol=2e-6, rtol=2e-6,
                                               err_msg='Offline/online motion features disagree')
                    # 旧清单已保存初态和命令，可从原始元数据恢复行为标签。
                    behavior = episode_metadata.get('behavior') or behavior_name(
                        episode_metadata.get('initial','standing'),
                        episode_metadata.get('schedule',[{'vx':0.,'yaw':0.}]))
                    if record.get('behavior',behavior) != behavior:
                        raise ValueError('Episode behavior disagrees with manifest')
                    if behavior == 'standup':
                        hold = episode_metadata.get('ground_start_hold_s', 0.)
                        if (not math.isfinite(hold) or hold < 0. or
                                hold != manifest.get('ground_start_hold_s', 0.)):
                            raise ValueError('Invalid or inconsistent ground-start hold metadata')
                        if self.ground_start_hold_s is not None and hold != self.ground_start_hold_s:
                            raise ValueError('Cannot mix standup demonstrations with different startup holds')
                        self.ground_start_hold_s = hold
                    if 'schedule' in episode_metadata:
                        schedule = episode_metadata['schedule']
                        if len(schedule) != 1 or behavior_name(episode_metadata['initial'],schedule) != behavior:
                            raise ValueError('Behavior does not match fixed episode command')
                        expected_command = np.asarray([schedule[0][key] for key in ('vx','yaw','leg_length')])
                        np.testing.assert_allclose(data['command'],
                            np.broadcast_to(expected_command,(len(time),3)),atol=1e-6,rtol=0,
                            err_msg='Recorded per-environment command changed or disagrees with metadata')
                    phase = torch.from_numpy(data['phase'].copy()).long() if 'phase' in data else None
                    if behavior == 'standup' and (phase is None or phase.shape != (len(time),)):
                        raise ValueError('Standup episode needs recorded controller phases')
                    self.episodes.append(dict(record, behavior=behavior, features=features,
                                              phase=phase, time=torch.from_numpy(time.copy()), path=str(path)))
                hashes.append((identity,record['sha256']))
        # 同一批清单换顺序也应恢复相同采样序列。
        self.episodes.sort(key=lambda episode:episode['episode_id'])
        groups = sorted({e['group_id'] for e in self.episodes})
        if len(groups) < 2:
            raise ValueError('Need at least two qualified seed/command groups for a nonleaking split')
        rng = torch.Generator().manual_seed(split_seed)
        # 每种行为都按完整种子组划分，避免验证集只剩静站或训练集没有起立。
        group_behavior = {}
        for episode in self.episodes:
            group = episode['group_id']
            if group in group_behavior and group_behavior[group] != episode['behavior']:
                raise ValueError('A seed group mixes different behavior labels')
            group_behavior[group] = episode['behavior']
        valid_groups = set()
        for behavior in sorted(set(group_behavior.values())):
            candidates = [group for group in groups if group_behavior[group] == behavior]
            if len(candidates) < 2:
                raise ValueError(f'Need two qualified seed groups for each behavior: {behavior}')
            order = torch.randperm(len(candidates), generator=rng).tolist()
            valid_count = max(1,min(len(candidates)-1,round(len(candidates)*validation_fraction)))
            valid_groups.update(candidates[i] for i in order[:valid_count])
        if any(value is not None for value in declared_splits):
            declared=declared_splits[0]
            if any(value!=declared for value in declared_splits) or not isinstance(declared,dict):
                raise ValueError('Mixed explicit motion group splits')
            training=set(declared['train']);validation=set(declared['validation'])
            if training&validation or not set(groups)<=training|validation or not validation&set(groups):
                raise ValueError('Invalid explicit group split or no qualified validation group')
            valid_groups=validation
        self.splits = {name: [e for e in self.episodes if (e['group_id'] in valid_groups) == (name == 'validation')]
                       for name in ('train','validation')}
        self.fingerprint = hashlib.sha256(json.dumps(dict(episodes=sorted(hashes),
            validation_groups=sorted(valid_groups),schema=FEATURE_SCHEMA,
            sampling='uniform_motion_classes_v2'),sort_keys=True).encode()).hexdigest()
        self._frames, self._starts, self._class_starts = {}, {}, {}
        for name, episodes in self.splits.items():
            offset, starts, frames, classes = 0, [], [], {}
            for episode in episodes:
                f = episode['features']
                valid_starts = torch.arange(len(f)-WINDOW+1)+offset
                starts.append(valid_starts)
                if episode['behavior'] == 'standup':
                    from ..teacher.standup import DONE, FAILED
                    phase = episode['phase']
                    preparing = (phase != DONE) & (phase != FAILED)
                    if not preparing.any():
                        raise ValueError('Standup episode has no recorded rise window')
                    # DONE只是交接；前2秒的伸腿／落稳过程也属于起立示范。
                    rising = (preparing | (episode['time'] <= 2.)).unfold(0,WINDOW,1).any(-1)
                    for class_name,mask in (('standup',rising),('standing',~rising)):
                        if mask.any():
                            classes.setdefault(class_name,[]).append(valid_starts[mask])
                else:
                    classes.setdefault(episode['behavior'],[]).append(valid_starts)
                frames.append(f)
                offset += len(f)
            self._frames[name] = torch.cat(frames)
            self._starts[name] = torch.cat(starts)
            self._class_starts[name] = {key:torch.cat(classes[key]) for key in sorted(classes)}

    def sample(self, split, batch_size, generator):
        classes = list(self._class_starts[split].values())
        labels = torch.randint(len(classes),(batch_size,),generator=generator)
        selected = torch.empty(batch_size,dtype=torch.long)
        for index,starts in enumerate(classes):
            mask = labels == index
            selected[mask] = starts[torch.randint(len(starts),(int(mask.sum()),),generator=generator)]
        return self._frames[split][selected[:,None]+torch.arange(WINDOW)]

    def normalization(self):
        # 仅用训练回合统计；每帧只计一次，验证回合不影响归一化。
        train = self._frames['train'].double()
        return train.mean(0).float(), train.std(0, correction=0).clamp_min(.2).float()

    def description(self):
        return dict(fingerprint=self.fingerprint,xml_sha256=self.xml_sha256,feature_schema=FEATURE_SCHEMA,
                    ground_start_hold_s=self.ground_start_hold_s,
                    sample_dt=SAMPLE_DT,window=WINDOW,sampling='uniform_motion_classes_v2',
                    class_window_counts={split:{name:len(starts) for name,starts in classes.items()}
                                         for split,classes in self._class_starts.items()},
                    splits={name:[{k:e[k] for k in ('episode_id','group_id','behavior','sha256','path')}
                                  for e in episodes] for name,episodes in self.splits.items()})
