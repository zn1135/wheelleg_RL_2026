#!/usr/bin/env python3
"""Finite V13 gradient, stage, episode-boundary and saved-run checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.supermimic import learner as module
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.state import StateAdapter, SENSORS, NOISE, MEMORY, PARAM_DIM, STATE_DIM, MASSES, SOLREF
from ct_mjlab.supermimic.buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT/'artifacts/supermimic/supermimic-v13-alternating-20261031-r1'
PARENT = ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v7_worldfit4_actortransfer3600/checkpoint_000615.pt'
WORLD = ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v12_motiononly_retrain_r1/final.pt'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def prerequisites():
    assert module.ALGORITHM_CONTRACT.endswith('-v13'), 'Expected a distinct alternating-training V13 contract'
    assert callable(getattr(Learner, 'init_world', None)), 'Missing verified V12 world transfer'
    assert callable(getattr(Learner, 'optimizer_update_counts', None)), 'Missing actual optimizer counters'
    return dict(new_alternating_contract=module.ALGORITHM_CONTRACT)


def fixture(n=20, device='cpu'):
    adapter = StateAdapter().to(device)
    s = torch.zeros(n, STATE_DIM, device=device)
    s[:, :25] = adapter.kin.qpos0.float()
    s[:, 2] = .3
    s[:, 3] = 1.
    p = torch.zeros(n, PARAM_DIM, device=device)
    p[:, :6] = s[:, adapter.motor_q]
    p[:, 6:12], p[:, 12:18], p[:, 18:24] = 10., .5, 1.
    p[:, MASSES] = 1.
    p[:, SOLREF] = p.new_tensor([.02, 1.])
    s[:, SENSORS] = adapter.raw_sensors(s[:, :64], p)[:, None].expand(-1, 9, -1).flatten(1)
    c = torch.zeros(n, 21, device=device)
    c[:, 2], c[:, 6], c[:, 19], c[:, 20] = .3, 1., .04, .01
    c[:, 7:19:2], c[:, 8:19:2] = -torch.inf, torch.inf
    return s, p, c


def command_fixture():
    from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
    state = copy.deepcopy(torch.load(WORLD, map_location='cpu', weights_only=False, mmap=True)['curriculum'])
    return SimpleNamespace(cfg=command_cfg('walk'), checkpoint_state=lambda:copy.deepcopy(state),
        load_checkpoint_state=lambda value:state.update(value))


def initialized(device):
    l = Learner(device, actor_lr=3e-5)
    command = command_fixture()
    actor = l.init_actor(PARENT, command, task='chuanliantui_walk')
    world = l.init_world(WORLD, command, task='chuanliantui_walk')
    return l, command, actor, world


def synthetic_buffer():
    s, p, c = fixture()
    b = ReplayBuffer(24, 20, 'cpu')
    adapter = StateAdapter()
    frame = adapter.actor_frame(s, p, torch.zeros(20, 3), torch.zeros(20, 6))
    history = frame.repeat(1, 5)
    previous = torch.zeros(20, 6)
    for tick in range(16):
        action = torch.full((20, 6), .02+.001*tick)
        path = s[:, None, :64].expand(-1, 5, -1).clone()
        path[:, :, 0] += torch.arange(1, 6)*.0001
        path[:, :, 25] = .01+.001*tick
        ns, frame, reward, failure, features = adapter.advance(s, p, path, torch.zeros(20, 3),
            action, previous, previous*.5, c, s[:, NOISE])
        b.add(history=history, privileged=torch.zeros(20, 71), action=action, next_frame=frame,
            next_privileged=torch.zeros(20, 71), velocity=torch.full((20, 3), .02), reward=reward,
            done=torch.zeros(20, dtype=torch.bool), failure=failure, features=features, context=c,
            previous=previous, previous_previous=previous*.5, state=s, parameters=p, next_state=ns,
            path=path.flatten(-2))
        history, s, previous = torch.cat((history[:, 25:], frame), -1), ns, action
    return b


def snapshot(l):
    return {name:copy.deepcopy(getattr(l, name).state_dict()) for name in
        ('actor', 'world', 'critic', 'opt_actor', 'opt_encoder', 'opt_world', 'opt_critic')}


def cpu_checks(fixture_dir):
    torch.set_num_threads(1)
    torch.manual_seed(61)
    l, command, actor_transfer, world_transfer = initialized('cpu')
    source = torch.load(WORLD, map_location='cpu', weights_only=False, mmap=True)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False, mmap=True)
    assert equal(l.world.state_dict(), source['world']) and equal(l.actor.state_dict(), parent['actor'])
    assert l.optimizer_update_counts() == dict(world=0, encoder=0, actor=0, critic=0)
    s, p, c = fixture(4)
    altered = s.clone()
    altered[:, [9, 14]+list(range(49, 64))+list(range(MEMORY.start+1, MEMORY.stop))] = torch.nan
    assert l.critic.inputs(s, p, c).shape == (4, 324)
    assert equal(l.critic.inputs(s, p, c), l.critic.inputs(altered, p, c))
    assert equal(l.world.inputs(s, p, torch.zeros(4, 6), torch.zeros(4, 6), torch.zeros(4, 6)),
        l.world.inputs(altered, p, torch.zeros(4, 6), torch.zeros(4, 6), torch.zeros(4, 6)))
    # Verify the motion path without any direct PD-torque cost or critic bootstrap.
    physics = torch.load(WORLD.parent/'initial_physics.pt', map_location='cpu', weights_only=False, mmap=True)
    ss, pp = physics['state'][:4].clone(), physics['parameters'][:4].clone()
    aa = torch.full((4, 6), .03, requires_grad=True)
    with module.frozen(l.world):
        state = ss
        for _ in range(3):
            state = l.world.step(state, pp, aa, torch.zeros(4, 3), c, aa*.5, aa*.25, ss[:, NOISE])[0]
        objective = state[:, 25].sum()+state[:, 2].sum()
        gradient = torch.autograd.grad(objective, aa)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-9
        direction = torch.randn_like(aa)
        def value(action):
            st = ss
            for _ in range(3):
                st = l.world.step(st, pp, action, torch.zeros(4, 3), c, action*.5, action*.25, ss[:, NOISE])[0]
            return st[:, 25].sum()+st[:, 2].sum()
        with torch.no_grad():
            numeric = (value(aa+.001*direction)-value(aa-.001*direction))/.002
        analytical = (gradient*direction).sum()
        torch.testing.assert_close(analytical, numeric, atol=2e-3, rtol=.05)
    assert equal(l.world.state_dict(), source['world'])
    b = synthetic_buffer()
    # Terminal transition remains valid; later episode fields cannot influence loss/gradient.
    w = b.sample(4, 4)
    w['done'][:, 1] = True
    w['mask'][:] = torch.tensor([1., 1., 0., 0.])
    poisoned = {k:v.clone() for k, v in w.items()}
    for key, value in poisoned.items():
        if key not in ('mask', 'done', 'failure') and value.is_floating_point():
            value[:, 2:] = torch.nan
    def world_result(window):
        l.opt_world.zero_grad(set_to_none=True)
        loss, _ = l.world_loss(window)
        loss.backward()
        return loss.detach(), [None if v.grad is None else v.grad.detach().clone() for v in l.world.parameters()]
    clean_loss, clean_grad = world_result(w)
    poison_loss, poison_grad = world_result(poisoned)
    assert equal(clean_loss, poison_loss) and equal(clean_grad, poison_grad)
    def imagine_result(window):
        l.opt_actor.zero_grad(set_to_none=True)
        before = snapshot(l)
        with module.frozen(l.world, l.critic, l.actor.encoder):
            result = l.imagination(window)
            result[0].backward()
        assert all(equal(getattr(l, name).state_dict(), before[name]) for name in ('world', 'critic', 'actor'))
        return result, [None if v.grad is None else v.grad.detach().clone() for v in l.actor.mlp.parameters()]
    clean, cg = imagine_result(w)
    poison, pg = imagine_result(poisoned)
    assert equal(clean, poison) and equal(cg, pg)
    assert all(torch.isfinite(g).all() for g in cg if g is not None)
    _, states, parameters, rewards, continuation, mask, contexts = clean
    lengths = mask.sum(0).long()
    with torch.no_grad():
        total = states.new_zeros(mask.shape[1])
        alive = torch.ones_like(total)
        for t in range(len(rewards)):
            total += .99**t*alive*rewards[t]
            alive *= continuation[t]
        for col, length in enumerate(lengths):
            value = l.critic(states[length, col:col+1], parameters[length, col:col+1], contexts[length, col:col+1])
            from ct_mjlab.supermimic.analytic_reward import return_bounds
            lower, upper = return_bounds(contexts[length, col:col+1], .99)
            total[col] += .99**int(length)*continuation[:length, col].prod()*value.clamp(lower, upper)[0]
        torch.testing.assert_close(clean[0], -total.mean(), atol=1e-7, rtol=1e-6)
    value_loss, targets = l.critic_loss(states, parameters, rewards, continuation, mask, contexts)
    assert not targets.requires_grad and torch.isfinite(value_loss)
    with torch.no_grad():
        for col, length in enumerate(lengths):
            values = l.critic(states[:length+1, col], parameters[:length+1, col], contexts[:length+1, col])
            lower, upper = return_bounds(contexts[:length+1, col], .99)
            values = values.clamp(lower, upper)
            target = values[-1]
            for t in reversed(range(int(length))):
                target = rewards[t, col]+.99*continuation[t, col]*(.05*values[t+1]+.95*target)
                torch.testing.assert_close(target, targets[t, col], atol=2e-7, rtol=1e-6)
    # A wholly ineligible task window must not advance actor, value or value statistics.
    ineligible = {k:v.clone() for k, v in w.items()}
    ineligible['context'][:, :, 5] = 1.
    before_empty = copy.deepcopy(l.critic.state_dict())
    empty = l.imagination(ineligible)
    assert not empty[5].any()
    l.critic_loss(*empty[1:])
    assert equal(before_empty, l.critic.state_dict())
    # One real four-world fitting group before and after a four-Adam fixture warmup.
    before = snapshot(l)
    l.update(b, batch=4, window=3, horizon=4, policy=False, encoder=False)
    assert l.optimizer_update_counts() == dict(world=4, encoder=0, actor=0, critic=0)
    assert all(equal(getattr(l, name).state_dict(), before[name]) for name in ('actor', 'critic', 'opt_actor', 'opt_encoder', 'opt_critic'))
    actor_phase = {}
    original_imagination = l.imagination
    original_actor_step = l.opt_actor.step
    def observed_imagination(window):
        actor_phase['world'] = copy.deepcopy(l.world.state_dict())
        actor_phase['critic'] = copy.deepcopy(l.critic.state_dict())
        actor_phase['encoder'] = copy.deepcopy(l.actor.encoder.state_dict())
        return original_imagination(window)
    def observed_actor_step(*args, **kwargs):
        assert equal(l.world.state_dict(), actor_phase['world'])
        assert equal(l.critic.state_dict(), actor_phase['critic'])
        assert equal(l.actor.encoder.state_dict(), actor_phase['encoder'])
        return original_actor_step(*args, **kwargs)
    with patch.object(l, 'imagination', side_effect=observed_imagination), patch.object(l.opt_actor, 'step', side_effect=observed_actor_step):
        l.update(b, batch=4, window=3, horizon=4, policy=True, encoder=True)
    counts = l.optimizer_update_counts()
    assert counts == dict(world=8, encoder=1, actor=1, critic=1)
    assert not equal(l.actor.state_dict(), before['actor']) and not equal(l.critic.state_dict(), before['critic'])
    assert all(torch.isfinite(v).all() for name in ('actor', 'world', 'critic') for v in getattr(l, name).state_dict().values())
    fixture_dir.mkdir(parents=True, exist_ok=False)
    config = dict(source['config'], world_loss_recipe=module.WORLD_LOSS_RECIPE,
        alternating_recipe=module.ALTERNATING_RECIPE, policy_warmup_updates=4,
        initialization=dict(kind='finite_checker_fixture', actor_transfer=actor_transfer, world_transfer=world_transfer))
    checkpoint = fixture_dir/'saved.pt'
    l.save(checkpoint, command, config=config, iteration=2, transitions=320, world_updates=8)
    restored = Learner('cpu', actor_lr=3e-5)
    saved = restored.load(checkpoint, command, expected_config={'policy_warmup_updates':4})
    assert all(equal(getattr(restored, k).state_dict(), snapshot(l)[k]) for k in snapshot(l))
    assert restored.optimizer_update_counts() == counts and restored.training_phase(4) == 'alternating'
    assert torch.equal(torch.get_rng_state(), saved['torch_rng'])
    failures = []
    malformed_curriculum = copy.deepcopy(saved)
    malformed_curriculum['curriculum']['speed_curriculum']['forward']['stage'] = 999999
    malformed_rng = copy.deepcopy(saved)
    malformed_rng['torch_rng'] = torch.zeros(0, dtype=torch.uint8)
    malformed_paths = []
    for name, value in (('bad-curriculum', malformed_curriculum), ('bad-rng', malformed_rng)):
        path = fixture_dir/(name+'.pt')
        torch.save(value, path)
        malformed_paths.append((name, path))
    for name, action in (
        ('changed_resume_schedule', lambda:restored.load(checkpoint, command, expected_config={'policy_warmup_updates':8})),
        ('old_full_resume', lambda:restored.load(WORLD, command)),
        ('unapproved_world_source', lambda:restored.init_world(PARENT, command, task='chuanliantui_walk')),
        *[(name, lambda path=path:restored.load(path, command)) for name, path in malformed_paths]):
        before_bad, rng = snapshot(restored), torch.get_rng_state().clone()
        with patch.object(command, 'load_checkpoint_state') as load_curriculum:
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid restore accepted: '+name)
            assert load_curriculum.call_count == 0
        assert all(equal(getattr(restored, k).state_dict(), v) for k, v in before_bad.items())
        assert torch.equal(rng, torch.get_rng_state())
        failures.append(name)
    return dict(actual_fixture_optimizer_updates=counts, full_615_actor_and_v12_world_inherited=True,
        source_world_updates=source['world_updates'], motion_only_critic_dim=324, deleted_nan_inputs_ignored=True,
        motion_path_gradient_max=float(gradient.abs().max()), finite_difference=float(numeric), analytical=float(analytical),
        no_direct_torque_or_bootstrap_gradient_shortcut=True, terminal_padding_nan_loss_and_gradient_invariant=True,
        actor_phase_world_critic_encoder_parameters_and_statistics_frozen=True,
        soft_return_and_per_tail_lambda_oracles=True, ineligible_critic_parameters_and_statistics_unchanged=True,
        checkpoint_restore_networks_optimizers_counts_rng=True, rejected_without_mutation=failures,
        source_world_transfer=world_transfer, actor_transfer=actor_transfer,
        fixture_scope='Synthetic transitions and actual Adam8/1/1/1 only; not trial learning or behavior.')


def integration(args):
    from ct_mjlab.supermimic import __main__ as entry
    args.task, args.seed, args.num_envs = 'chuanliantui_walk', 42, 20
    env, _ = entry.make_env(args)
    try:
        raw = env.unwrapped
        l = Learner(args.device, actor_lr=3e-5)
        command = raw.command_manager.get_term('standup')
        l.init_actor(PARENT, command, task=args.task)
        l.init_world(WORLD, command, task=args.task)
        before = snapshot(l)
        b = ReplayBuffer(64, 20, args.device)
        obs = env.get_observations()
        adapter = StateAdapter().to(args.device)
        maxima = dict(actor25=0., history125=0., real_reward=0., real_state=0.)
        resets = 0
        for tick in range(32):
            history = obs['history'].clone()
            obs, reward, done = entry.collect(env, l, b, obs)
            data = {k:v[(b.ptr-1)%b.capacity] for k, v in b.data.items()}
            path = data['path'].reshape(20, 5, 64)
            real = adapter.advance(data['state'], data['parameters'], path, obs['actor'][:, 6:9],
                data['action'], data['previous'], data['previous_previous'], data['context'], data['next_state'][:, NOISE])
            motion = l.world.adapter.advance(data['state'], data['parameters'], path, obs['actor'][:, 6:9],
                data['action'], data['previous'], data['previous_previous'], data['context'], data['next_state'][:, NOISE])
            valid = ~done.bool()
            for name, a, target in (('real_reward', real[2], reward), ('real_state', real[0], data['next_state']),
                ('actor25', motion[1][valid], obs['actor'][valid]),
                ('history125', torch.cat((history[:, 25:], motion[1]), -1)[valid], obs['history'][valid])):
                if a.numel():
                    maxima[name] = max(maxima[name], float((a-target).abs().max()))
                    torch.testing.assert_close(a, target, atol=2e-5, rtol=2e-5)
            assert torch.equal(data['done'], done.bool()) and torch.equal(data['failure'], raw.reset_terminated.bool())
            resets += int(done.sum())
        assert all(equal(getattr(l, k).state_dict(), before[k]) for k in before)
        batch = b.sample(8, 4)
        with module.frozen(l.world, l.critic, l.actor.encoder):
            objective = l.imagination(batch)[0]
            objective.backward()
        assert torch.isfinite(objective)
        gradients = [v.grad for v in l.actor.mlp.parameters() if v.grad is not None]
        assert gradients and all(torch.isfinite(v).all() for v in gradients)
        assert l.optimizer_update_counts() == dict(world=0, encoder=0, actor=0, critic=0)
        return dict(real_transitions=640, physics_substeps=3200, actual_resets=resets,
            maximum_errors=maxima, inherited_networks_and_all_optimizers_unchanged=True,
            full_actor_gradient_finite=True, optimizer_updates=0,
            scope='Finite20-environment32-step collection and gradient only, not96-case behavior acceptance.')
    finally:
        env.close()


def audit(run, exit_record):
    from ct_mjlab.supermimic.learner import ALTERNATING_RECIPE, WORLD_LOSS_RECIPE, asset_digest
    initial = torch.load(run/'initial.pt', map_location='cpu', weights_only=False, mmap=True)
    final = torch.load(run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    config = json.loads((run/'config.json').read_text())
    assert config['alternating_recipe'] == ALTERNATING_RECIPE and config['world_loss_recipe'] == WORLD_LOSS_RECIPE
    assert config['policy_warmup_updates'] == 256 and config['seconds'] == 600
    source = ROOT/'mjlab_training/src/ct_mjlab'
    manifest = json.loads((run/'source_sha256.json').read_text())
    assert manifest == {str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    assert manifest == {str(p.relative_to(run/'snapshot/ct_mjlab')):sha(p) for p in (run/'snapshot/ct_mjlab').rglob('*.py')}
    assert all(d['algorithm_contract'] == module.ALGORITHM_CONTRACT and d['asset_sha256'] == asset_digest() for d in (initial, final))
    points = []
    for path in [run/'initial.pt', *[p for p in (run/'prewarm.pt', run/'postwarm.pt') if p.exists()], run/'final.pt']:
        d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        counts = {}
        for name in ('world', 'encoder', 'actor', 'critic'):
            slots = d['opt_'+name]['state']
            steps = {int(v['step']) for v in slots.values()}
            assert len(steps) <= 1
            counts[name] = next(iter(steps)) if steps else 0
        assert d['world_updates'] == counts['world'] and d['optimizer_update_counts'] == counts
        assert counts['world'] % 4 == 0 and counts['actor'] == counts['critic']
        assert all(not v.is_floating_point() or torch.isfinite(v).all() for n in ('actor', 'world', 'critic') for v in d[n].values())
        if path.name == 'prewarm.pt':
            assert counts == dict(world=256, encoder=0, actor=0, critic=0)
            assert equal(d['actor'], initial['actor']) and equal(d['critic'], initial['critic'])
        if path.name == 'postwarm.pt':
            assert counts == dict(world=260, encoder=1, actor=1, critic=1)
        points.append(dict(path=str(path), sha256=sha(path), actual_optimizer_steps=counts,
            phase=d['training_phase'], transitions=d['transitions'], iteration=d['iteration']))
    if config.get('resume'):
        parent = torch.load(config['resume'], map_location='cpu', weights_only=False, mmap=True)
        for name in ('actor', 'world', 'critic', 'opt_actor', 'opt_encoder', 'opt_world', 'opt_critic'):
            assert equal(initial[name], parent[name]), name
        assert initial['world_updates'] == parent['world_updates']
    else:
        parent = torch.load(PARENT, map_location='cpu', weights_only=False, mmap=True)
        world = torch.load(WORLD, map_location='cpu', weights_only=False, mmap=True)
        assert equal(initial['actor'], parent['actor']) and equal(initial['world'], world['world'])
        assert all(not initial[n]['state'] for n in ('opt_world', 'opt_actor', 'opt_encoder', 'opt_critic'))
        assert float(initial['critic']['norm.count']) == 0
        assert (run/'prewarm.pt').exists() and (run/'postwarm.pt').exists()
    receipt = json.loads(exit_record.read_text())
    assert receipt['returncode'] == 0 and receipt['final_checkpoint_exists'] and receipt['final_sha256'] == sha(run/'final.pt')
    assert receipt['last_metrics']['world_updates'] == final['world_updates'] and receipt['last_metrics']['elapsed_s'] >= 600.
    assert points[-1]['actual_optimizer_steps']['actor'] > 0
    return dict(completion_verified=True, source_count=len(manifest), source_sha256=manifest,
        points=points, initialization=initial['initialization'], optimizer_counts_verified=True,
        source_and_snapshots_unchanged=True, no_physics_or_optimizer_updates_during_audit=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--exit-record', type=Path)
    args = p.parse_args()
    result = prerequisites()
    if args.run:
        result.update(audit(args.run.resolve(), args.exit_record.resolve()))
    elif args.device == 'cpu':
        result.update(cpu_checks(args.output.with_name(args.output.stem+'-fixtures')))
    else:
        result.update(integration(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(all_checks_passed=True, checker_sha256=sha(__file__), checks=result), stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
