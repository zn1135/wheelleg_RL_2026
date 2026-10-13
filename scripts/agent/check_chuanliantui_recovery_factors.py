"""Native preflight: single-factor isolation, full migration, finite4096-env steps."""
import argparse
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import recovery_factors as factors
from ct_mjlab.experiments import huananhu_v14_recovery as recovery


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    original = factors.validate_source(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(learning=False, source_sha256=factors.SOURCE_SHA256, num_envs=4096,
                  steps=16, cases={}, reward_isolation={})
    reference = factors.serializable(factors.make_env_cfg('recovery_reference'))
    shared_initial = None
    for case in factors.CASES:
        torch.manual_seed(42)
        cfg = factors.make_env_cfg(case)
        encoded = factors.serializable(cfg)
        if case != 'baseline':
            probe = json.loads(json.dumps(encoded))
            probe['rewards'] = reference['rewards']
            assert probe == reference, f'{case}: unexpected non-reward configuration difference'
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'), clip_actions=None)
        try:
            factors.initialize_stage(env.unwrapped, case)
            runner_cfg = factors.source.runner_cfg(42)
            runner = factors.FactorRunner(env, runner_cfg, None, case=case, device='cuda:0')
            runner.load(args.source, migration=True)
            check_path = args.output / f'{case}-initial.pt'
            runner.save(str(check_path))
            loaded = torch.load(check_path, map_location='cpu', weights_only=False)
            equal = {k: factors.state_equal(original[k], loaded[k]) for k in (
                'actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')}
            assert all(equal.values()) and loaded['infos']['completed_updates'] == 2000
            factors.validate_checkpoint(loaded, case)
            runner.load(check_path)
            numerical = 0
            if case != 'baseline':
                initial = env.unwrapped.sim.data.qpos.clone()
                if shared_initial is None:
                    shared_initial = initial
                assert torch.equal(initial, shared_initial), 'Recovery arms start from different qpos'
                state = env.unwrapped.command_manager.get_term('standup')
                assert state.curriculum.stage == int(case == 'skip_righting')
            for _ in range(16):
                obs, rewards, _, _ = env.step(torch.zeros(4096, 6, device=env.device))
                assert obs['actor'].shape == (4096, 35) and obs['critic'].shape == (4096, 78)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rewards).all()
                numerical += int(env.unwrapped.command_manager.get_term('standup').numerical_failure.sum())
            assert numerical == 0
            if case == 'recovery_reference':
                state = env.unwrapped.command_manager.get_term('standup')
                for stage in (0, 1, 2):
                    state.curriculum.stage = stage
                    expected = {k: v.clone() for k, v in recovery.RewardState(env.unwrapped).compute(env.unwrapped).items()}
                    for variant, allowed in (
                        ('keep_height', {'track_height_exp_tight', 'track_height_square'}),
                        ('keep_contact', {'undesired_contact'}), ('bearing_reward', {'wheel_balance'})):
                        actual = factors.RewardState(env.unwrapped, variant).compute(env.unwrapped)
                        changed = {k for k in expected if not torch.equal(expected[k], actual[k])}
                        assert changed <= allowed, f'{variant}: unexpected reward changes {changed}'
                        if ((variant == 'keep_height' and stage == 2)
                                or (variant == 'keep_contact' and stage != 0)
                                or (variant == 'bearing_reward' and stage == 0)):
                            assert not changed
                        report['reward_isolation'][f'{variant}_stage{stage}'] = sorted(changed)
            report['cases'][case] = dict(finite=True, numerical=0, complete_state_equal=equal,
                initial_updates=2000, learning_rate=runner.alg.learning_rate,
                stage=loaded['infos']['recovery_curriculum'], own_checkpoint_reload=True)
            (args.output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
            print('FACTOR_PREFLIGHT ' + json.dumps({case: report['cases'][case]}), flush=True)
            del runner, loaded
        finally:
            env.close()
        del env
        torch.cuda.empty_cache()
    report['passed'] = True
    (args.output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
