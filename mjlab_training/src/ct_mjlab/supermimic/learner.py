# SPDX-License-Identifier: MIT
# BPTT and lambda-return structure adapted from supermimic_wheelbipe/train.py.
# Copyright (c) 2026 RadiantForever; see THIRD_PARTY_LICENSE.txt.
"""Motion dynamics fitting, real velocity supervision and imagined policy/value updates."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from .networks import MotionCritic, make_actor, act, ACTION_CLIP
from .motion import WorldModel, physical47_from_full64, delta46
from .motion_adapter import MOTION_AUXILIARY_RECIPE, IMAGINATION_CLAMP_RECIPE
from .analytic_reward import return_bounds
from .state import PHYS_DIM, SUBSTEPS, PHYSICAL, SENSORS, NOISE, MEMORY, WRENCH
from ct_mjlab.envs.chuanliantui_split.runner import contracts, validate as validate_task_checkpoint

V12_CONTRACT = 'chuanliantui-supermimic-motion-only-pd-reward-soft-termination-v12'
V13_HISTORY_CONTRACT = 'chuanliantui-supermimic-motion-only-alternating-bptt-v13'
V13_NO_HISTORY_CONTRACT = 'chuanliantui-supermimic-motion-only-alternating-bptt-v13-no-action-history-v1'
V13_HEADING_CONTRACT = 'chuanliantui-supermimic-motion-only-alternating-bptt-v13-no-history-heading-invariant-v2'
V13_ENDPOINT_CONTRACT = 'chuanliantui-supermimic-control-endpoint-heading-invariant-bptt-v13-v3'
V13_CLAMP_CONTRACT = 'chuanliantui-supermimic-control-endpoint-imagination-clamp-bptt-v13-v4'
ALGORITHM_CONTRACT = 'chuanliantui-supermimic-control-endpoint-state-action-clamp-bptt-v13-v5'
ALTERNATING_RECIPE = dict(version='control-endpoint-world4-real-encoder-actor-bptt-critic-action-clamp-v4',
    world_steps_per_update=4,encoder_real_supervision=True,critic_state_dim=162,
    critic_parameter_offset=66,critic_norm_update_after_actor=True,
    soft_survival_gradients=True,invalid_replay_rows_not_forwarded=True,
    imagination_state_clamp=IMAGINATION_CLAMP_RECIPE,
    action_clip=ACTION_CLIP,collection_action_clip='after_exploration_before_simulation_and_replay')
WORLD_STEPS_PER_UPDATE = 4
WORLD_LOSS_RECIPE = dict(version='control-endpoint-squared-delta-std-heading-invariant-v4',
    motion_state_dim=47,motion_delta_dim=46,substeps=1,physics_substeps=5,
    prediction_dt_s=.01,supervision='next_control_endpoint_only',
    world_input_dim=129,world_action_history_frames=0,
    input_coordinates='height_yaw_removed_rotation6d_body_v_body_w_nonwheel_q_joint_v',
    input_wrench_coordinates='body_force_body_torque',
    excluded_full_physical_columns=[9,14,*range(49,64)],
    coordinates='body_local_pose_velocity_and_all_nonwheel_q_all_joint_v',
    penalty='squared_delta_std',output_delta_normalization='motion46_only',
    internal_reward_and_termination=MOTION_AUXILIARY_RECIPE)
ACTOR_TRANSFER_CONTRACTS = (
    'chuanliantui-supermimic-physical-features-bptt-analytic-reward-v3',
    'chuanliantui-supermimic-goal-isolated-physical-features-bptt-analytic-reward-v5',
    'chuanliantui-supermimic-goal-isolated-bounded-value-bptt-analytic-reward-v6',
    'chuanliantui-supermimic-worldfit4-goal-isolated-bounded-value-bptt-analytic-reward-v7',
    'chuanliantui-supermimic-contact-target-cap5-worldfit4-goal-isolated-bounded-value-bptt-analytic-reward-v8',
    'chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9',
    'chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10',
    'chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11',
    V12_CONTRACT, V13_HISTORY_CONTRACT, V13_NO_HISTORY_CONTRACT, V13_HEADING_CONTRACT,
    V13_ENDPOINT_CONTRACT, V13_CLAMP_CONTRACT, ALGORITHM_CONTRACT)


def checkpoint_action_clip(contract):
    return ACTION_CLIP if contract == ALGORITHM_CONTRACT else None


def asset_digest():
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    return hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    previous = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, value in zip(parameters,previous):
            p.requires_grad_(value)


def apply_gradient(loss, optimizer, parameters, *, max_norm=1.):
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss; no optimizer step performed')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_value_(parameters,.1)
    optimizer.step()
    return norm.detach()


class Learner:
    def __init__(self, device, *, actor_lr=3e-4):
        self.actor, self.world, self.critic = make_actor(device), WorldModel().to(device), MotionCritic().to(device)
        self.opt_actor = torch.optim.Adam(self.actor.mlp.parameters(),lr=actor_lr)
        self.opt_encoder = torch.optim.Adam(self.actor.encoder.parameters(),lr=1e-3)
        self.opt_world = torch.optim.Adam(self.world.parameters(),lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=3e-4)

    def world_loss_terms(self, truth, path):
        # Keep one sample axis so callers pool endpoint and46 coordinates identically.
        before = physical47_from_full64(truth[..., -1:, :])
        after = physical47_from_full64(path[..., -1:, :])
        return (delta46(before,after)/self.world.delta_norm.std()).square()

    @staticmethod
    def _state_finite(state):
        retained = torch.cat((physical47_from_full64(state[...,PHYSICAL]),
            state[...,SENSORS],state[...,NOISE],state[...,MEMORY][...,:1]),-1)
        return retained.isfinite().all(-1)

    @staticmethod
    def _prefix(valid):
        return valid.long().cumprod(1).bool()

    def world_loss(self, w):
        truth = w['path'].reshape(*w['path'].shape[:-1],SUBSTEPS,PHYS_DIM)
        valid = (w['mask'].bool() & physical47_from_full64(truth[..., -1, :]).isfinite().all(-1)
            & (w['context'][...,5] < .5) & w['parameters'].isfinite().all(-1)
            & w['action'].isfinite().all(-1) & w['previous'].isfinite().all(-1)
            & w['previous_previous'].isfinite().all(-1) & w['next_state'][...,NOISE].isfinite().all(-1))
        valid = self._prefix(valid & self._state_finite(w['state'][:,0])[:,None])
        state = torch.zeros_like(w['state'][:,0])
        first = valid[:,0].nonzero().flatten()
        state = state.index_copy(0,first,w['state'][first,0])
        total = next(self.world.parameters()).sum()*0.
        positions = 0
        for t in range(w['action'].shape[1]):
            rows = valid[:,t].nonzero().flatten()
            if not len(rows):
                continue
            result = self.world.step(state[rows],w['parameters'][rows,t],w['action'][rows,t],
                w['next_frame'][rows,t,6:9],w['context'][rows,t],w['previous'][rows,t],
                w['previous_previous'][rows,t],w['next_state'][rows,t,NOISE],return_path=True)
            state = state.index_copy(0,rows,result[0])
            total = total+self.world_loss_terms(truth[rows,t],result[-1]).mean(dim=(-1,-2)).sum()
            positions += len(rows)
        loss = total/max(positions,1)
        return loss,dict(dynamics=loss.detach(),valid_world_positions=positions)

    @staticmethod
    def _valid_states(mask):
        lengths = mask.sum(0).long()
        return ((torch.arange(len(mask)+1,device=mask.device)[:,None] <= lengths)
                & (lengths > 0))

    def _values(self, states, parameters, contexts, state_valid, gamma):
        values = states.new_zeros(states.shape[:2])
        if state_valid.any():
            selected = self.critic(states[state_valid],parameters[state_valid],contexts[state_valid])
            lower,upper = return_bounds(contexts[state_valid],gamma)
            values = values.masked_scatter(state_valid,selected.clamp(lower,upper))
        return values

    def imagination(self, w, gamma=.99, lam=.95):
        same_context = (w['context'][:,:,3:5] == w['context'][:,:1,3:5]).all(-1)
        consistent = ((same_context | ~w['mask'].bool()).all(1)
            & (w['context'][:,0,5] < .5) & self._state_finite(w['state'][:,0])
            & w['history'][:,0].isfinite().all(-1)
            & w['previous'][:,0].isfinite().all(-1)
            & w['previous_previous'][:,0].isfinite().all(-1))
        legal_context = w['context'].isfinite() | (w['context'].isinf()
            & (torch.arange(w['context'].shape[-1],device=w['context'].device)>=7)
            & (torch.arange(w['context'].shape[-1],device=w['context'].device)<19))
        valid = (w['mask'].bool() & consistent[:,None] & legal_context.all(-1)
            & w['parameters'].isfinite().all(-1) & w['next_state'][...,NOISE].isfinite().all(-1))
        command_finite = w['next_frame'][...,6:9].isfinite().all(-1) | w['done'].bool()
        valid = self._prefix(valid & command_finite)
        mask = valid.T.float()
        start = valid[:,0].nonzero().flatten()
        state = torch.zeros_like(w['state'][:,0]).index_copy(0,start,w['state'][start,0])
        history = torch.zeros_like(w['history'][:,0]).index_copy(0,start,w['history'][start,0])
        parameters = torch.zeros_like(w['parameters'][:,0]).index_copy(0,start,w['parameters'][start,0])
        context = torch.zeros_like(w['context'][:,0]).index_copy(0,start,w['context'][start,0])
        previous = torch.zeros_like(w['previous'][:,0]).index_copy(0,start,w['previous'][start,0])
        previous_previous = torch.zeros_like(w['previous_previous'][:,0]).index_copy(0,start,w['previous_previous'][start,0])
        histories,states,parameter_states,contexts = [history],[state],[parameters],[context]
        rewards,continuations = [],[]
        for t in range(w['action'].shape[1]):
            rows = valid[:,t].nonzero().flatten()
            reward = state.new_zeros(len(state));continuation = torch.zeros_like(reward)
            if len(rows):
                action = act(self.actor,history[rows])
                next_command = torch.where(w['done'][rows,t,None],history[rows,-19:-16],
                    w['next_frame'][rows,t,6:9])
                ns,frame,r,_,_ = self.world.step(state[rows],parameters[rows],action,next_command,
                    context[rows],previous[rows],previous_previous[rows],w['next_state'][rows,t,NOISE],
                    imagination=True)
                continuation = continuation.index_copy(0,rows,self.world.adapter.soft_survival(ns[...,PHYSICAL]))
                reward = reward.index_copy(0,rows,r)
                state = state.index_copy(0,rows,ns)
                history = history.index_copy(0,rows,torch.cat((history[rows,25:],frame),-1))
                previous_previous = previous_previous.index_copy(0,rows,previous[rows])
                previous = previous.index_copy(0,rows,action)
                command = torch.stack((next_command[:,0]/2.,next_command[:,1]*4.,next_command[:,2]/5.),-1)
                context = context.index_copy(0,rows,torch.cat((command,context[rows,3:]),-1))
            if t+1 < len(valid[0]):
                continuing = valid[:,t+1].nonzero().flatten()
                if len(continuing):
                    next_params = torch.cat((parameters[continuing,:WRENCH.start],
                        w['parameters'][continuing,t+1,WRENCH]),-1)
                    parameters = parameters.index_copy(0,continuing,next_params)
            histories.append(history);states.append(state);parameter_states.append(parameters);contexts.append(context)
            rewards.append(reward);continuations.append(continuation)
        rewards,continuation = torch.stack(rewards),torch.stack(continuations)
        states,parameters,contexts = torch.stack(states),torch.stack(parameter_states),torch.stack(contexts)
        values = self._values(states,parameters,contexts,self._valid_states(mask),gamma)
        alive = torch.cat((torch.ones_like(continuation[:1]),continuation.cumprod(0)),0)
        discount = gamma**torch.arange(len(rewards),device=states.device)[:,None]
        lengths = mask.sum(0).long();columns = torch.arange(mask.shape[1],device=states.device)
        objective = (discount*alive[:-1]*rewards).sum(0)
        objective = objective+gamma**lengths*alive[lengths,columns]*values[lengths,columns]
        objective = objective.masked_fill(lengths==0,0.)
        loss = -objective.sum()/(lengths>0).float().sum().clamp_min(1)
        return (loss,states.detach(),parameters.detach(),rewards.detach(),continuation.detach(),mask,contexts.detach())

    def critic_loss(self, states, parameters, rewards, continuations, mask, contexts, gamma=.99, lam=.95):
        state_valid = self._valid_states(mask)
        with torch.no_grad():
            if state_valid.any():
                self.critic.norm.update(self.critic.inputs(states[state_valid],parameters[state_valid],contexts[state_valid]))
            values = self._values(states,parameters,contexts,state_valid,gamma)
            target = values[-1]
            targets = []
            for t in reversed(range(len(rewards))):
                target = rewards[t]+gamma*continuations[t]*((1-lam)*values[t+1]+lam*target)
                target = torch.where(mask[t].bool(),target,values[t])
                targets.append(target)
            targets = torch.stack(targets[::-1])
        valid = mask.bool()
        if valid.any():
            prediction = self.critic(states[:-1][valid],parameters[:-1][valid],contexts[:-1][valid])
            loss = (prediction-targets[valid]).square().mean()
        else:
            loss = next(self.critic.parameters()).sum()*0.
        return loss,targets

    def optimizer_update_counts(self):
        counts = {}
        for name in ('world','encoder','actor','critic'):
            state = getattr(self,'opt_'+name).state
            steps = {int(value['step']) for value in state.values()}
            if len(steps)>1:
                raise ValueError('Inconsistent optimizer steps: '+name)
            counts[name] = next(iter(steps)) if steps else 0
        return counts

    def training_phase(self, warmup_updates):
        counts = self.optimizer_update_counts()
        return ('alternating' if counts['actor'] else 'alternating_ready'
            if counts['world']>=warmup_updates else 'world_warmup')

    def update(self, buffer, *, batch=128, window=4, horizon=8, policy=True, encoder=True):
        for _ in range(WORLD_STEPS_PER_UPDATE):
            w = buffer.sample(batch,window)
            self.world.observe(w)
            world_loss,errors = self.world_loss(w)
            if errors['valid_world_positions']:
                apply_gradient(world_loss,self.opt_world,self.world.parameters())
        valid = (w['mask'].bool() & w['history'].isfinite().all(-1) & w['velocity'].isfinite().all(-1))
        h,v = w['history'][valid],w['velocity'][valid]
        if len(h):
            encoder_loss = F.mse_loss(self.actor.encoder(h),v)
            if encoder:
                apply_gradient(encoder_loss,self.opt_encoder,self.actor.encoder.parameters(),max_norm=.1)
        else:
            encoder_loss = next(self.actor.encoder.parameters()).sum()*0.
        metrics = dict(world=float(world_loss.detach()),encoder=float(encoder_loss.detach()),
            **{key:float(value) for key,value in errors.items()})
        if policy:
            starts = buffer.sample(batch,horizon)
            with frozen(self.world,self.critic,self.actor.encoder):
                actor_loss,states,parameters,rewards,continuations,mask,contexts = self.imagination(starts)
                eligible = bool(mask.any())
                grad = (apply_gradient(actor_loss,self.opt_actor,self.actor.mlp.parameters())
                    if eligible else torch.zeros((),device=states.device))
            value_loss,_ = self.critic_loss(states,parameters,rewards,continuations,mask,contexts)
            if eligible:
                apply_gradient(value_loss,self.opt_critic,self.critic.parameters())
            metrics.update(actor=float(actor_loss.detach()),critic=float(value_loss.detach()),actor_grad=float(grad),
                imagination_eligible_fraction=float((mask[0]>0).float().mean()))
        with torch.no_grad():
            _,errors = self.world_loss(buffer.sample(batch,window,validation=True))
            metrics.update({f'heldout_{key}':float(value) for key,value in errors.items()})
        metrics.update({name+'_updates':value for name,value in self.optimizer_update_counts().items()})
        return metrics

    def init_actor(self, path, command, *, task):
        path = Path(path)
        parent_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path,map_location='cpu',weights_only=False)
        expected = self.actor.state_dict()
        actor = saved.get('actor',{})
        source_infos, target_infos = saved.get('infos'), contracts(command.cfg)
        infos_match = source_infos == target_infos
        reward_transfer = None
        # Only the verified SM615 actor may cross this exact reward-only change.
        if (not infos_match and parent_sha256 ==
                '50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'):
            removed = {'lin_vel_z','base_link_contact','wheels_airborne','action_rate','action_smooth'}
            old_recipe = 'split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-v5'
            new_recipe = 'split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-no-vz-contact-rate-smooth-v7'
            weights = source_infos['reward_weights']
            transferred_infos = dict(source_infos,reward_recipe=new_recipe,
                reward_weights={key:value for key,value in weights.items() if key not in removed})
            if (source_infos['reward_recipe'] == old_recipe and len(weights) == 18
                    and removed <= weights.keys() and len(transferred_infos['reward_weights']) == 13
                    and transferred_infos == target_infos):
                infos_match = True
                digest = lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
                reward_transfer = dict(kind='SM615_reward_only_v5_to_v7_actor_transfer',
                    source_recipe=old_recipe,target_recipe=new_recipe,
                    removed_reward_terms=sorted(removed),source_reward_weights=weights,
                    target_reward_weights=transferred_infos['reward_weights'],
                    source_infos_sha256=digest(source_infos),target_infos_sha256=digest(target_infos),
                    all_nonreward_infos_equal=True,world_critic_optimizers_not_inherited=True)
        if (saved.get('algorithm_contract') not in ACTOR_TRANSFER_CONTRACTS
                or saved.get('config',{}).get('mode') != 'train'
                or saved.get('config',{}).get('task') != task
                or not infos_match
                or saved.get('asset_sha256') != asset_digest()
                or not all(name in saved for name in ('world','critic','opt_actor',
                    'opt_encoder','opt_world','opt_critic'))
                or set(actor) != set(expected)
                or any(not isinstance(actor[name],torch.Tensor)
                    or actor[name].shape != value.shape or actor[name].dtype != value.dtype
                    or not torch.isfinite(actor[name]).all() for name,value in expected.items())):
            raise ValueError('Actor transfer requires a matching self-trained SM task/XML/actor checkpoint; PPO is not accepted')
        # Transfer only the full actor state. New world/value/norms/optimizers stay cold.
        self.actor.load_state_dict(actor,strict=True)
        self.actor.supermimic_action_clip = ACTION_CLIP
        result = dict(kind='actor_transfer_world_coldstart',parent_checkpoint=str(path.resolve()),
            parent_sha256=parent_sha256,parent_algorithm_contract=saved['algorithm_contract'],
            inherited=['actor'],reset=['world','critic','opt_actor','opt_encoder','opt_world',
                'opt_critic','iteration','transitions','world_updates','curriculum','replay','rng'],
            fresh_from_zero=False,target_action_clip=ACTION_CLIP,
            source_action_clip=checkpoint_action_clip(saved['algorithm_contract']))
        if reward_transfer is not None:
            result['reward_only_actor_transfer'] = reward_transfer
        return result

    def init_world(self, path, command, *, task):
        path = Path(path).resolve()
        root = Path(__file__).resolve().parents[4]
        source_run = root/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v12_motiononly_retrain_r1'
        trial = root/'artifacts/supermimic-v12-retrain-20261031-r1'
        source_sha = '3a06b3ece5d284d658cdff7b95ddf932fb81f40b26e97e59073e9fb7afe09646'
        if path != source_run/'final.pt' or hashlib.sha256(path.read_bytes()).hexdigest() != source_sha:
            raise ValueError('World transfer requires the verified completed V12 final')
        receipt = json.loads((trial/'train-exit.json').read_text())
        audit = json.loads((trial/'completed-audit-r2.json').read_text())
        audit_exit = json.loads((trial/'completed-audit-r2-exit.json').read_text())
        saved = torch.load(path,map_location='cpu',weights_only=False)
        expected = self.world.state_dict();world = saved.get('world',{})
        counts = self.optimizer_update_counts()
        manifest = json.loads((source_run/'source_sha256.json').read_text())
        snapshots_ok = all(hashlib.sha256((source_run/'snapshot/ct_mjlab'/key).read_bytes()).hexdigest()==value
            for key,value in manifest.items())
        if (receipt.get('returncode') != 0 or receipt.get('final_sha256') != source_sha
                or not receipt.get('final_checkpoint_exists') or Path(receipt['log_dir']) != source_run
                or audit_exit.get('returncode') != 0 or not audit.get('all_checks_passed')
                or not audit.get('checks',{}).get('completion_verified') or not snapshots_ok
                or saved.get('algorithm_contract') != V12_CONTRACT or saved.get('interrupted',False)
                or saved.get('infos') != contracts(command.cfg) or saved.get('asset_sha256') != asset_digest()
                or saved.get('config',{}).get('task') != task or saved['config'].get('mode') != 'train'
                or saved['config'].get('world_loss_recipe') != WORLD_LOSS_RECIPE
                or saved['world_updates'] != receipt['last_metrics']['world_updates']
                or any(counts.values()) or set(world) != set(expected)
                or any(not isinstance(world[key],torch.Tensor) or world[key].shape != value.shape
                    or world[key].dtype != value.dtype or not torch.isfinite(world[key]).all()
                    for key,value in expected.items())):
            raise ValueError('World transfer task/recipe/shape/source/completion mismatch; '
                'Legacy substep world weights cannot initialize control-endpoint dynamics')
        self.world.load_state_dict(world,strict=True)
        return dict(kind='world_transfer_from_completed_v12_final',parent_checkpoint=str(path),
            parent_sha256=source_sha,parent_algorithm_contract=V12_CONTRACT,
            source_world_updates=saved['world_updates'],inherited=['world'],
            inherited_normalizers=['in_norm','delta_norm'],reset=['opt_world'],
            actual_source_exit_record_sha256=hashlib.sha256((trial/'train-exit.json').read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256((source_run/'source_sha256.json').read_bytes()).hexdigest())

    @staticmethod
    def _saved_optimizer_counts(saved):
        counts = {}
        for name in ('world','encoder','actor','critic'):
            states = saved['opt_'+name]['state']
            for value in states.values():
                step = float(value['step'])
                if not torch.isfinite(torch.tensor(step)) or step < 0 or not step.is_integer():
                    raise ValueError('Invalid saved optimizer step: '+name)
            steps = {int(value['step']) for value in states.values()}
            if len(steps)>1:
                raise ValueError('Inconsistent saved optimizer steps: '+name)
            counts[name] = next(iter(steps)) if steps else 0
        return counts

    @staticmethod
    def _phase_for_counts(counts, warmup_updates):
        return ('alternating' if counts['actor'] else 'alternating_ready'
            if counts['world']>=warmup_updates else 'world_warmup')

    def save(self, path, command, *, config, iteration, transitions, world_updates, interrupted=False):
        if self.actor.supermimic_action_clip != ACTION_CLIP:
            raise ValueError('Legacy inference actor cannot be saved as the bounded-action training recipe')
        if config.get('world_loss_recipe') != WORLD_LOSS_RECIPE or config.get('alternating_recipe') != ALTERNATING_RECIPE:
            raise ValueError('Checkpoint world/alternating recipe mismatch')
        counts = self.optimizer_update_counts()
        if counts['world'] != world_updates:
            raise ValueError('World update count differs from actual Adam steps')
        state = dict(algorithm_contract=ALGORITHM_CONTRACT, asset_sha256=asset_digest(), infos=contracts(command.cfg),
            curriculum=command.checkpoint_state(), config=config, iteration=iteration,transitions=transitions,
            world_updates=world_updates,initialization=config['initialization'],
            optimizer_update_counts=counts,training_phase=self._phase_for_counts(counts,config['policy_warmup_updates']),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        for name in ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic'):
            state[name] = getattr(self,name).state_dict()
        if interrupted:
            state['interrupted'] = True  # A partial group is diagnostic only, never a training resume.
        temporary = path.with_suffix('.tmp')
        torch.save(state,temporary)
        temporary.replace(path)

    def load(self, path, command, *, inference=False, expected_config=None):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        allowed = ACTOR_TRANSFER_CONTRACTS if inference else (ALGORITHM_CONTRACT,)
        if (saved.get('algorithm_contract') not in allowed or saved.get('infos') != contracts(command.cfg)
                or saved.get('asset_sha256') != asset_digest()
                or (not inference and saved.get('config',{}).get('world_steps_per_update') != WORLD_STEPS_PER_UPDATE)
                or (not inference and saved.get('config',{}).get('world_loss_recipe') != WORLD_LOSS_RECIPE)
                or (not inference and saved.get('config',{}).get('alternating_recipe') != ALTERNATING_RECIPE)
                or type(saved.get('world_updates')) is not int or saved['world_updates'] < 0
                or (not inference and (saved.get('interrupted',False)
                    or saved['world_updates'] % WORLD_STEPS_PER_UPDATE))):
            raise ValueError('Checkpoint algorithm/task/physics contract mismatch; PPO checkpoints are not accepted')
        names = ('actor',) if inference else ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic')
        if not inference:
            if expected_config is not None and any(saved['config'].get(key)!=value for key,value in expected_config.items()):
                raise ValueError('Resume configuration mismatch before state restoration')
            counts = self._saved_optimizer_counts(saved)
            if (counts != saved.get('optimizer_update_counts') or counts['world'] != saved['world_updates']
                    or saved.get('training_phase') != self._phase_for_counts(counts,saved['config']['policy_warmup_updates'])
                    or counts['actor'] != counts['critic'] or counts['actor'] > counts['encoder']):
                raise ValueError('Saved optimizer counts/phase mismatch')
            if counts['actor'] and counts['world'] <= saved['config']['policy_warmup_updates']:
                raise ValueError('Policy updated before world warmup completed')
        for name in ('actor',) if inference else ('actor','world','critic'):
            expected = getattr(self,name).state_dict();actual = saved[name]
            if (set(actual)!=set(expected) or any(not isinstance(actual[key],torch.Tensor)
                    or actual[key].shape!=value.shape or actual[key].dtype!=value.dtype
                    or not torch.isfinite(actual[key]).all() for key,value in expected.items())):
                raise ValueError('Checkpoint network state mismatch: '+name)
        if not inference:
            for name in ('opt_actor','opt_encoder','opt_world','opt_critic'):
                candidate = saved[name];current = getattr(self,name).state_dict()
                if (len(candidate['param_groups'])!=len(current['param_groups'])
                        or any(len(a['params'])!=len(b['params']) for a,b in zip(candidate['param_groups'],current['param_groups']))):
                    raise ValueError('Checkpoint optimizer groups mismatch: '+name)
                expected_keys = {key for group in candidate['param_groups'] for key in group['params']}
                if candidate['state'] and set(candidate['state']) != expected_keys:
                    raise ValueError('Checkpoint optimizer parameter states mismatch: '+name)
                for group,parameters in zip(candidate['param_groups'],getattr(self,name).param_groups):
                    for key,parameter in zip(group['params'],parameters['params']):
                        state = candidate['state'].get(key,{})
                        if state and not {'step','exp_avg','exp_avg_sq'} <= state.keys():
                            raise ValueError('Incomplete checkpoint Adam state: '+name)
                        if any(not isinstance(state[field],torch.Tensor) or state[field].shape!=parameter.shape
                                or not torch.isfinite(state[field]).all() for field in ('exp_avg','exp_avg_sq') if field in state):
                            raise ValueError('Checkpoint optimizer state mismatch: '+name)
        # Validate every remaining restore operation before changing networks or optimizers.
        curriculum = saved['curriculum']
        validate_task_checkpoint(dict(saved['infos'],standup_curriculum=curriculum),command.cfg)
        try:
            bool(curriculum['unlocked'])
            completed = int(curriculum['completed_episodes'])
            recovered = int(curriculum['recovered_episodes'])
            rate = float(curriculum['last_recovered_rate'])
            if not 0 <= recovered <= completed or not math.isfinite(rate) or not 0. <= rate <= 1.:
                raise ValueError('Invalid height curriculum counters')
        except (KeyError,TypeError,ValueError,RuntimeError,OverflowError) as error:
            raise ValueError('Invalid checkpoint height curriculum') from error
        if not inference:
            try:
                torch.Generator(device='cpu').set_state(saved['torch_rng'])
            except (KeyError,TypeError,ValueError,RuntimeError) as error:
                raise ValueError('Invalid checkpoint CPU RNG state') from error
            cuda_rng = saved.get('cuda_rng')
            if not isinstance(cuda_rng,list) or any(not isinstance(value,torch.Tensor)
                    or value.dtype!=torch.uint8 or value.device.type!='cpu' or value.ndim!=1
                    or not value.numel() for value in cuda_rng):
                raise ValueError('Invalid checkpoint CUDA RNG state')
            if cuda_rng and torch.cuda.is_available():
                expected = torch.cuda.get_rng_state_all()
                if len(cuda_rng)!=len(expected) or any(a.shape!=b.shape for a,b in zip(cuda_rng,expected)):
                    raise ValueError('Checkpoint CUDA RNG devices/shape mismatch')
        for name in names:
            getattr(self,name).load_state_dict(saved[name])
        self.actor.supermimic_action_clip = checkpoint_action_clip(saved['algorithm_contract'])
        command.load_checkpoint_state(saved['curriculum'])
        if not inference:
            torch.set_rng_state(saved['torch_rng'])
            if saved['cuda_rng'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        return saved
