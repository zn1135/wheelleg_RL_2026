"""Explicit legacy39 migration and train-only rise/local-feedback preservation."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .bc import FORMAT,ALIGNED_LABEL_REFERENCE
from . import wheel_angles

SCHEMA='ct_bc_preserve_rise_feedback_v1'


def migrate_wheel_actor(actor,saved,description):
    profile=wheel_angles.PROFILES.get(description['observation_contract'])
    if (saved['format']!=FORMAT or actor.mlp[0].in_features!=39 or
            profile is None or description.get('wheel_angles')!=profile['wheel_angles'] or
            saved['dataset']['xml_sha256']!=description['xml_sha256'] or
            saved['dataset']['control_sources']!=description['control_sources']):
        raise ValueError('Explicit wheel migration requires legacy39 with identical XML and PD controls')
    if (description['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT and
            description.get('label_reference')!=ALIGNED_LABEL_REFERENCE):
        raise ValueError('Shared wheel migration requires aligned teacher position references')
    old_groups={s:{e['group_id'] for e in saved['dataset']['splits'][s]} for s in ('train','validation')}
    if any(old_groups[s]!=set(description['group_split'][s]) for s in old_groups):
        raise ValueError('Wheel migration cannot change training or heldout pose membership')
    with torch.no_grad():actor.mlp[0].weight[:,14:16]=0.
    return actor


class PreservationDataset:
    def __init__(self,path,description,checkpoint_sha256):
        path=Path(path);manifest=json.loads(path.read_text());source=path.parent/manifest['path']
        if (manifest['schema']!=SCHEMA or manifest.get('canonical_precision')!='ieee_fp32' or manifest['heldout_used'] or
                manifest['checkpoint_sha256']!=checkpoint_sha256 or
                manifest['base_motion_fingerprint']!=description['motion_fingerprint'] or
                manifest['xml_sha256']!=description['xml_sha256'] or
                manifest['observation_contract']!=description['observation_contract'] or
                (description['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT and
                 manifest.get('label_reference')!=description.get('label_reference')) or
                hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['sha256']):
            raise ValueError('Preservation source/model/control contract mismatch')
        train=set(description['group_split']['train']);heldout=set(description['group_split']['validation'])
        self.values={}
        with np.load(source) as d:
            for kind in ('rise','feedback'):
                groups=set(d[kind+'_group'].tolist())
                if not groups or not groups<=train or groups&heldout:
                    raise ValueError('Preservation must contain training poses only')
            for name in ('rise_observation','rise_target','feedback_plus','feedback_minus','feedback_target_delta','rise_scale','feedback_scale'):
                self.values[name]=torch.from_numpy(d[name].copy())
        dimensions={'rise_observation':39,'rise_target':6,'feedback_plus':39,'feedback_minus':39,'feedback_target_delta':6}
        for name,dim in dimensions.items():
            value=self.values[name]
            if value.ndim!=2 or value.shape[1]!=dim or not len(value) or not torch.isfinite(value).all():
                raise ValueError('Invalid preservation arrays')
        if len(self.values['rise_target'])!=len(self.values['rise_observation']) or any(
                len(self.values[k])!=len(self.values['feedback_plus']) for k in ('feedback_minus','feedback_target_delta')):
            raise ValueError('Preservation pair lengths differ')
        if any(self.values[k].shape!=(6,) or not torch.isfinite(self.values[k]).all() or
                not (self.values[k]>0).all() for k in ('rise_scale','feedback_scale')):
            raise ValueError('Invalid training-only preservation scales')
        self.description=dict(manifest,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rise_loss_weight=1.,feedback_loss_weight=1.,feedback='plus-minus action differences, not absolute parking actions')

    def loss(self,actor,generator,device,size=256):
        d=self.values
        rise=torch.randint(len(d['rise_observation']),(size,),generator=generator)
        feedback=torch.randint(len(d['feedback_plus']),(size,),generator=generator)
        rise_error=actor.mlp(d['rise_observation'][rise].to(device))-d['rise_target'][rise].to(device)
        # 约束纠正反应的变化量，允许停车所需的动作偏置改变。
        delta=actor.mlp(d['feedback_plus'][feedback].to(device))-actor.mlp(d['feedback_minus'][feedback].to(device))
        feedback_error=delta-d['feedback_target_delta'][feedback].to(device)
        return ((rise_error/d['rise_scale'].to(device)).square().mean(),
                (feedback_error/d['feedback_scale'].to(device)).square().mean())
