"""No-learning64-step finite, exact reward isolation and full A migration gates."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import height_knee as h


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, nargs='+', default=[16, 4096])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    assert set(args.num_envs) <= {16, 4096}
    args.output.mkdir(parents=True, exist_ok=False)
    original = h.validate_source(args.source)
    root = Path(__file__).resolve().parents[2]
    frozen = {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest()
        for folder in ('mjlab_training/src/ct_mjlab/experiments', 'mjlab_training/src/ct_mjlab/reference_training')
        for x in (root/folder).glob('*.py') if x.name != 'height_knee.py'}
    report = dict(command=shlex.join([sys.executable, *sys.argv]), started=now(), learning=False,
        source_sha256=h.SOURCE_SHA256, device=args.device, gpu=torch.cuda.get_device_name(args.device),
        versions={n: importlib.metadata.version(n) for n in ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'rsl-rl-lib')},
        steps=64, cases={}, passed=False, exit_code=None, frozen_sources_sha256=frozen,
        module_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest())
    try:
        reference = h.serializable(h.source.make_env_cfg('baseline'))
        for case in h.CASES:
            encoded = h.serializable(h.make_env_cfg(case))
            normalized = json.loads(json.dumps(encoded))
            normalized['rewards'] = reference['rewards']
            assert normalized == reference, 'Unexpected command/physics/observation/action/event/termination changes'
            assert set(encoded['rewards']) == set(reference['rewards']) | {'knee_limit'}
            for name, term in encoded['rewards'].items():
                assert term['weight'] == (-.5 if name == 'knee_limit' else reference['rewards'][name]['weight'])
            for n in args.num_envs:
                torch.manual_seed(42)
                env = RslRlVecEnvWrapper(ManagerBasedRlEnv(h.make_env_cfg(case, n), device=args.device), clip_actions=None)
                try:
                    raw = env.unwrapped
                    state = raw.command_manager.get_term('standup')
                    numerical = 0
                    for _ in range(64):
                        obs, reward, _, _ = env.step(torch.zeros(n, 6, device=args.device))
                        assert obs['actor'].shape == (n, 35) and obs['critic'].shape == (n, 78)
                        assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
                        assert all(torch.isfinite(v).all() for v in raw._huananhu_rewards.values.values())
                        numerical += raw._huananhu_rewards.transition_numerical_failures
                    assert numerical == 0
                    before, velocity = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
                    baseline = {k: v.clone() for k, v in h.height_base.RewardState(raw).compute(raw).items()}
                    variant = h.RewardState(raw, case).compute(raw)
                    difference = {k: float((variant[k]-baseline[k]).abs().max()) for k in baseline}
                    changed = {k for k, v in difference.items() if v != 0.}
                    assert changed <= (set(h.GATED_TERMS) if case == 'knee_gate' else set())
                    assert torch.equal(variant['knee_limit'], h.huananhu_rewards.RewardState(raw).compute(raw)['knee_limit'])
                    # Same real qpos/qvel; alter ONLY the target-height command.
                    command = state.command.clone()
                    height = getattr(state, 'height_above_ground', raw.scene['robot'].data.root_link_pos_w[:, 2])
                    samples = []
                    for error, expected in ((.02, 1.), (.045, .5), (.07, 0.)):
                        state.command[:, 2] = height + error
                        ungated = h.base.RewardState(raw).compute(raw)
                        old = h.height_base.RewardState(raw).compute(raw)
                        new = h.RewardState(raw, case).compute(raw)
                        gates = h.height_gate(height-state.command[:, 2], case)
                        if case == 'knee_gate':
                            assert torch.allclose(gates, torch.full_like(gates, expected), atol=2e-6, rtol=0.)
                        for name in baseline:
                            if name in h.GATED_TERMS:
                                assert torch.equal(new[name], ungated[name]*gates)
                                if case == 'knee_only':
                                    assert torch.equal(new[name], old[name])
                            else:
                                assert torch.equal(new[name], old[name]), name
                        samples.append(dict(error_m=error, gate_min=float(gates.min()), gate_max=float(gates.max()),
                            original_ungated_times_gate_exact=True, all_other_terms_exact=True))
                    state.command.copy_(command)
                    assert torch.equal(before, raw.sim.data.qpos) and torch.equal(velocity, raw.sim.data.qvel)
                    entry = dict(num_envs=n, steps=64, finite=True, all_reward_terms_finite=True,
                        numerical_failures=numerical, configuration_isolation=True,
                        reward_changed=sorted(changed), reward_max_abs_delta=difference,
                        allowed_weighted_reward_changes=['knee_limit']+(list(h.GATED_TERMS) if case == 'knee_gate' else []),
                        knee_weight=-.5, knee_formula_original_exact=True, gate_samples=samples,
                        same_qpos_qvel_for_reward_isolation=True)
                    if n == 4096:
                        runner = h.HeightKneeRunner(env, h.height_base.runner_cfg(42), None, case=case, device=args.device)
                        runner.load(args.source, migration=True)
                        path = args.output/case/'model_resume_initial.pt'
                        path.parent.mkdir(exist_ok=False)
                        runner.save(str(path))
                        saved = torch.load(path, map_location='cpu', weights_only=False)
                        equal = {k: h.source.state_equal(original[k], saved[k]) for k in
                            ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')}
                        assert all(equal.values()) and saved['infos']['completed_updates'] == 2500
                        h.validate_checkpoint(saved, case)
                        runner.load(path)
                        assert runner.actual_updates == 2500 and runner.alg.learning_rate == 1e-5
                        assert all(h.source.state_equal(original[k], getattr(runner.alg, attr).state_dict())
                            for k, attr in (('actor_state_dict', 'actor'), ('critic_state_dict', 'critic'),
                                            ('optimizer_state_dict', 'optimizer')))
                        entry.update(complete_state_equal=equal, initial_updates=2500,
                            initial_learning_rate=runner.alg.learning_rate, own_checkpoint_reload=True,
                            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        del runner, saved
                    report['cases'][f'{case}:{n}'] = entry
                    print('HEIGHT_KNEE_PREFLIGHT ' + json.dumps(entry), flush=True)
                    (args.output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
                finally:
                    env.close()
                del env
                torch.cuda.empty_cache()
        report['passed'] = True
        report['exit_code'] = 0
    except Exception:
        report['exit_code'] = 1
        raise
    finally:
        report['ended'] = now()
        report['frozen_sources_unchanged'] = all(hashlib.sha256((root/k).read_bytes()).hexdigest() == v for k, v in frozen.items())
        (args.output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
