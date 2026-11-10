#!/usr/bin/env python3
"""V13 full resumes with actor-count-only or explicitly timed stopping guards.

Production source, clocks, update order and save/restore paths stay unchanged.
The memory-only wrapper adds stop guards and failure diagnostics, never losses.
"""

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts/supermimic/supermimic-v13-continuation-20261101-r1'
MAIN = ROOT/'mjlab_training/src/ct_mjlab/supermimic/__main__.py'
MAIN_SHA = '17051c4a88f66f62be54f17509382588f9620fa6bd959cd0e90a5e14ae76014d'
TARGETS = (700, 1100, 1500, 2200, 5200)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def bounded_source(source):
    outer = 'while time.monotonic() < deadline:'
    inner = 'if time.monotonic() >= deadline:'
    assert source.count(outer) == 1 and source.count(inner) == 2
    below = "learner.optimizer_update_counts()['actor'] < args.stop_actor_updates"
    reached = "learner.optimizer_update_counts()['actor'] >= args.stop_actor_updates"
    revised = source.replace(outer,
        'while time.monotonic() < deadline and '+below+' and not _continuation_hard_stop():')
    revised = revised.replace(inner,
        'if time.monotonic() >= deadline or '+reached+' or _continuation_hard_stop():')
    # Reversibility proves that no calculation or update statement was replaced.
    restored = revised.replace(
        'while time.monotonic() < deadline and '+below+' and not _continuation_hard_stop():', outer)
    restored = restored.replace(
        'if time.monotonic() >= deadline or '+reached+' or _continuation_hard_stop():', inner)
    assert restored == source
    ast.parse(revised)
    return revised


def count_source(source):
    initialization = 'deadline, last_save = started+args.seconds, started'
    outer = 'while time.monotonic() < deadline:'
    inner = 'if time.monotonic() >= deadline:'
    assert source.count(initialization) == 1
    assert source.count(outer) == 1 and source.count(inner) == 2
    below = "learner.optimizer_update_counts()['actor'] < args.stop_actor_updates"
    reached = "learner.optimizer_update_counts()['actor'] >= args.stop_actor_updates"
    assignments = sorted((node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'world_updates'
        and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
        and node.value.value.id == 'counts'), key=lambda node:node.lineno)
    assert assignments
    indentation = ' '*assignments[0].col_offset
    marker = indentation+"world_updates = counts['world']"
    assert source.count(marker) == 1
    checkpoint = marker+'\n'+indentation+(
        '_continuation_save_milestones(learner,command,config,iteration+1,transitions,world_updates)')
    replacements = ((initialization, 'deadline, last_save = None, started'),
        (outer, 'while '+below+':'), (inner, 'if '+reached+':'), (marker, checkpoint))
    revised = source
    for original, replacement in replacements:
        revised = revised.replace(original, replacement)
    restored = revised
    for original, replacement in reversed(replacements):
        restored = restored.replace(replacement, original)
    assert restored == source
    ast.parse(revised)
    return revised


def diagnostic_source(source):
    marker = "                metrics_file.write(json.dumps(metrics)+'\\n'); metrics_file.flush()"
    finalizer = '    finally:\n        if writer is not None:'
    assert source.count(marker) == 1 and source.count(finalizer) == 1
    checked = '                _continuation_check_metrics(metrics)\n'+marker
    handler = ("    except Exception as error:\n"
        "        _continuation_failure_diagnostics(error,locals())\n"
        "        raise\n"+finalizer)
    revised = source.replace(marker, checked).replace(finalizer, handler)
    assert revised.replace(checked, marker).replace(handler, finalizer) == source
    ast.parse(revised)
    return revised


def check_metrics(metrics):
    import math
    if any(isinstance(value, (int, float)) and not math.isfinite(value)
            for value in metrics.values()):
        raise FloatingPointError('Nonfinite production training metric; state preserved')


def failure_diagnostics(error, state):
    """Capture existing tensors before production finally closes the scene."""
    import torch
    directory = state['args'].log_dir
    directory.mkdir(parents=True, exist_ok=True)
    details = dict(exception_type=type(error).__name__, message=str(error),
                   captured_at_utc=utc_now().isoformat())
    numerical = isinstance(error, FloatingPointError) or any(
        word in str(error).lower() for word in ('nonfinite', 'non-finite', 'nan'))
    learner, env, buffer = (state.get(key) for key in ('learner', 'env', 'buffer'))
    raw = dict(exception=details, iteration=state.get('iteration'),
               transitions=state.get('transitions'), metrics=state.get('metrics'))
    if learner is not None:
        raw['learned_state'] = {name:getattr(learner, name).state_dict() for name in
            ('actor', 'world', 'critic', 'opt_actor', 'opt_encoder', 'opt_world', 'opt_critic')}
        if 'config' in state:
            try:
                learner.save(directory/('numerical_failure.pt' if numerical else 'runtime_failure.pt'),
                    state['command'], config=state['config'], iteration=state['iteration'],
                    transitions=state['transitions'],
                    world_updates=learner.optimizer_update_counts()['world'], interrupted=True)
                details['interrupted_checkpoint_saved'] = True
            except Exception as save_error:
                details['checkpoint_save_error'] = repr(save_error)
    if env is not None:
        from ct_mjlab.supermimic.state import capture_state, capture_parameters
        try:
            raw.update(state=capture_state(env.unwrapped), parameters=capture_parameters(env.unwrapped),
                       observations=state.get('obs'))
            data = env.unwrapped.sim.data
            raw['scene'] = {name:getattr(data, name).detach().cpu().clone() for name in
                            ('qpos', 'qvel', 'ctrl', 'qfrc_actuator', 'xfrc_applied')}
        except Exception as capture_error:
            details['scene_capture_error'] = repr(capture_error)
    if buffer is not None and buffer.size:
        count = min(8, buffer.size)
        rows = (buffer.ptr-count+torch.arange(count, device=buffer.device))%buffer.capacity
        raw['replay_tail'] = {key:value[rows].detach().cpu().clone() for key, value in buffer.data.items()}
    torch.save(raw, directory/'failure-state.pt')
    (directory/'failure.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')


def source_proof():
    assert sha(MAIN) == MAIN_SHA, 'Production main changed'
    expected = json.loads((ART/'source-sha256.json').read_text())
    source = ROOT/'mjlab_training/src/ct_mjlab'
    actual = {str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    assert len(actual) == 101 and actual == expected, 'Production source changed'
    return actual


def count_selfcheck(start=700, target=2200, thresholds=(1200,1700), world_start=3056):
    # Exercise the real source transformation using integer counters only.
    assert callable(globals().get('count_source')), 'Count-only source transform is missing'
    source = '''def train(args):
    learner = args.learner
    command, config = None, {}
    started, iteration, transitions = 0., 0, 0
    deadline, last_save = started+args.seconds, started
    while time.monotonic() < deadline:
        for _ in range(8):
            if time.monotonic() >= deadline:
                break
            args.collected += 1
            transitions += 1
        for _ in range(4):
            if time.monotonic() >= deadline:
                break
            learner.update()
            counts = learner.optimizer_update_counts()
            world_updates = counts['world']
        iteration += 1
        args.rows.append(learner.optimizer_update_counts().copy())
        if time.monotonic()-last_save >= 60:
            args.periodic_saves += 1
            last_save = time.monotonic()
    args.final = learner.optimizer_update_counts().copy()
'''
    revised = count_source(source)
    counts = dict(world=world_start, encoder=start, actor=start, critic=start)
    saved = []
    class Counter:
        def optimizer_update_counts(self):
            return counts.copy()
        def update(self):
            counts['world'] += 4
            for key in ('encoder', 'actor', 'critic'):
                counts[key] += 1
    def checkpoint(learner, command, config, iteration, transitions, world_updates):
        for threshold in thresholds:
            if counts['actor'] >= threshold and threshold not in saved:
                assert counts['actor'] == threshold
                saved.append(threshold)
    args = SimpleNamespace(learner=Counter(), stop_actor_updates=target, seconds=None,
        collected=0, rows=[], periodic_saves=0, final=None)
    env = dict(time=SimpleNamespace(monotonic=lambda:10.**30),
        _continuation_save_milestones=checkpoint)
    exec(compile(revised, '<count-only-boundary-fixture>', 'exec'), env)
    env['train'](args)
    assert args.final == dict(world=world_start+4*(target-start), encoder=target, actor=target, critic=target)
    assert max(row['actor'] for row in args.rows) == target and saved == list(thresholds)
    assert args.collected > 0 and args.periodic_saves > 0
    return dict(start_actor=start, additional_actor_updates=target-start, final=counts,
        saved_first_actor_counts=saved, overshoot=0, seconds=None, hard_deadline_utc=None,
        extreme_clock_does_not_stop_count_mode=True, periodic_save_statement_retained=True,
        model_forwards=0, simulator_steps=0, optimizer_steps=0)


def selfcheck():
    production = MAIN.read_text()
    node = next(node for node in ast.parse(production).body
                if isinstance(node, ast.FunctionDef) and node.name == 'train')
    revised = diagnostic_source(bounded_source(ast.get_source_segment(production, node)))
    ast.parse(revised)
    count_revised = diagnostic_source(count_source(ast.get_source_segment(production, node)))
    ast.parse(count_revised)
    assert "if time.monotonic()-last_save >= 60:" in count_revised
    assert "started+args.seconds" not in count_revised
    assert '_continuation_hard_stop' not in count_revised
    # Pure scheduling fixture: these are integers, never models or Adam updates.
    source = '''def train(args):
    learner = args.learner
    deadline = 10000.
    while time.monotonic() < deadline:
        for _ in range(8):
            if time.monotonic() >= deadline:
                break
            args.collected += 1
        for _ in range(4):
            if time.monotonic() >= deadline:
                break
            learner.update()
        args.rows.append(learner.optimizer_update_counts().copy())
    args.final = learner.optimizer_update_counts().copy()
'''
    checks = []
    for start, target in ((368, 700), (700, 1100), (1100, 1500), (368, 370), (700, 2200)):
        counts = dict(world=1728+4*(start-368), encoder=start, actor=start, critic=start)
        class Counter:
            def optimizer_update_counts(self):
                return counts.copy()
            def update(self):
                counts['world'] += 4
                for key in ('encoder', 'actor', 'critic'):
                    counts[key] += 1
        args = SimpleNamespace(learner=Counter(), stop_actor_updates=target,
                               collected=0, rows=[], final=None)
        env = dict(time=SimpleNamespace(monotonic=lambda:0.),
                   _continuation_hard_stop=lambda:False)
        exec(compile(bounded_source(source), '<continuation-boundary-fixture>', 'exec'), env)
        env['train'](args)
        assert args.final['actor'] == target and max(r['actor'] for r in args.rows) == target
        assert args.final['world'] == 1728+4*(target-368)
        checks.append(dict(start_actor=start, target_actor=target, final=args.final,
                           additional_updates=target-start, overshoot=0))
    # The hard stop exits normally without even collection or a synthetic update.
    args = SimpleNamespace(learner=Counter(), stop_actor_updates=1600,
                           collected=0, rows=[], final=None)
    env = dict(time=SimpleNamespace(monotonic=lambda:0.),
               _continuation_hard_stop=lambda:True)
    exec(compile(bounded_source(source), '<continuation-hard-stop-fixture>', 'exec'), env)
    env['train'](args)
    assert args.collected == 0 and not args.rows
    check_metrics(dict(world=1., training_phase='alternating'))
    for value in (float('nan'), float('inf')):
        try:
            check_metrics(dict(world=value))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Nonfinite metric accepted')
    count_checks = count_selfcheck()
    continuation_checks = count_selfcheck(2200,5200,(3200,4200),9056)
    return dict(all_checks_passed=True, scheduling_fixtures=checks, count_only=count_checks,
        count_only_2200_to_5200=continuation_checks,
        hard_stop_before_collection=True, guard_sites=3, only_guard_replacements=True,
        production_source_in_memory_transform_parses=True, nonfinite_metrics_rejected=True,
        model_forwards=0, simulator_steps=0, optimizer_steps=0,
        scope='Synthetic integer counters and a fake fixture clock only; real training uses unchanged actual clocks.')


def run(args):
    import time
    import torch
    from ct_mjlab.supermimic import __main__ as entry

    sources = source_proof()
    budget = json.loads((ART/'budget.json').read_text())
    assert args.stop_actor_updates in TARGETS
    deadline, seconds = None, None
    if args.count_only:
        assert args.seconds is None and args.hard_deadline_utc is None
        assert budget['budget_kind'] == 'actual_actor_updates'
        assert budget['termination_mode'] == 'actor_update_count_only'
        assert budget.get('time_limit') is None and budget.get('user_wall_time_budget_s') is None
        assert Path(budget['source_checkpoint']).resolve() == args.resume.resolve()
        assert sha(args.resume) == budget['source_checkpoint_sha256']
        assert args.stop_actor_updates == budget['maximum_actor_updates']
        assert args.save_actor_updates == budget['save_during_training_actor_updates']
    else:
        assert not args.save_actor_updates
        deadline = datetime.fromisoformat(args.hard_deadline_utc.replace('Z', '+00:00'))
        assert deadline.tzinfo is not None
        total_deadline = datetime.fromisoformat(budget['deadline_utc'])
        last_training_stop = total_deadline-timedelta(seconds=budget['minimum_final_evaluation_reserve_s'])
        assert deadline <= last_training_stop, 'Hard stop consumes the final evaluation reserve'
        assert 0 < args.seconds <= budget['maximum_training_loop_s']
        seconds = min(args.seconds, (deadline-utc_now()).total_seconds()-5.)
        assert seconds > 0, 'No bounded training time remains'
    saved = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
    before = saved['optimizer_update_counts']
    assert before['actor'] < args.stop_actor_updates and before['actor'] == before['critic']
    assert before['world'] >= 256 and saved['training_phase'] == 'alternating'
    if args.count_only:
        assert before == budget['source_optimizer_counts']
        assert before['actor'] == budget['source_actor_updates']
        assert args.stop_actor_updates-before['actor'] == budget['additional_actor_updates']
        assert args.save_actor_updates == sorted(set(args.save_actor_updates))
        assert all(before['actor'] < value < args.stop_actor_updates for value in args.save_actor_updates)
    previous = saved['config']
    keys = ('task', 'num_envs', 'seed', 'device', 'horizon', 'window', 'batch',
            'actor_lr', 'buffer_steps', 'policy_warmup_updates')
    options = {key:previous[key] for key in keys}
    assert options['policy_warmup_updates'] == 256
    options.update(mode='train', seconds=seconds, log_dir=args.log_dir, resume=args.resume,
        init_actor=None, init_world=None, stop_actor_updates=args.stop_actor_updates,
        hard_deadline_utc=deadline.isoformat() if deadline else None)
    if args.count_only:
        options.update(termination_mode='actor_update_count_only',budget_kind='actual_actor_updates',
            count_only=True,save_actor_updates=args.save_actor_updates,
            additional_actor_updates=budget['additional_actor_updates'],time_limit=None)
    function_source = inspect.getsource(entry.train)
    revised = diagnostic_source(count_source(function_source) if args.count_only
        else bounded_source(function_source))
    env = dict(vars(entry))
    if deadline:
        env['_continuation_hard_stop'] = lambda:utc_now() >= deadline
    milestones = []
    def save_milestones(learner, command, config, iteration, transitions, world_updates):
        counts = learner.optimizer_update_counts()
        for threshold in args.save_actor_updates:
            if counts['actor'] >= threshold and not any(
                    item['threshold_actor_updates'] == threshold for item in milestones):
                path = args.log_dir/f'actor_{threshold:06d}.pt'
                assert not path.exists()
                learner.save(path,command,config=config,iteration=iteration,
                    transitions=transitions,world_updates=world_updates)
                milestones.append(dict(threshold_actor_updates=threshold,
                    actual_actor_updates=counts['actor'],path=str(path.resolve()),sha256=sha(path),
                    optimizer_update_counts=counts,iteration=iteration,transitions=transitions))
    env['_continuation_save_milestones'] = save_milestones
    env['_continuation_check_metrics'] = check_metrics
    env['_continuation_failure_diagnostics'] = failure_diagnostics
    exec(compile(revised, str(MAIN), 'exec'), env)
    proof = dict(driver_sha256=sha(__file__), production_main_sha256=MAIN_SHA,
        production_train_sha256=hashlib.sha256(function_source.encode()).hexdigest(),
        bounded_train_sha256=hashlib.sha256(revised.encode()).hexdigest(),
        source_count=len(sources), guard_sites=3, finite_metric_guards=1, exception_diagnostics=1,
        loss_update_collection_and_save_statements_unchanged=True,
        resume=str(args.resume.resolve()), resume_sha256=sha(args.resume),
        initial_optimizer_counts=before, target_actor_updates=args.stop_actor_updates,
        requested_loop_seconds=args.seconds, configured_loop_seconds=seconds,
        hard_deadline_utc=deadline.isoformat() if deadline else None,
        termination_mode='actor_update_count_only' if args.count_only else 'actor_count_or_time',
        budget_kind='actual_actor_updates' if args.count_only else 'wall_time',
        save_actor_updates=args.save_actor_updates,
        resume_semantics='Full production restore including four Adam/norms/counts/RNG; environment and replay rebuilt. No repeated warmup.')
    started = utc_now().isoformat()
    wall_started = time.monotonic()
    env['train'](SimpleNamespace(**options))
    wall_elapsed = time.monotonic()-wall_started
    final_path = args.log_dir/'final.pt'
    final = torch.load(final_path, map_location='cpu', weights_only=False, mmap=True)
    counts = final['optimizer_update_counts']
    assert counts['actor'] <= args.stop_actor_updates, 'Actor target overshot'
    if args.count_only:
        assert counts['actor'] == args.stop_actor_updates
        assert [item['threshold_actor_updates'] for item in milestones] == args.save_actor_updates
    assert counts['actor'] == counts['critic'] and final['world_updates'] == counts['world']
    rows = [json.loads(line) for line in (args.log_dir/'metrics.jsonl').read_text().splitlines()]
    assert rows and all(rows[-1][name+'_updates'] == value for name, value in counts.items())
    assert source_proof() == sources
    loop_elapsed = rows[-1]['elapsed_s']
    reason = ('actor_update_limit' if counts['actor'] == args.stop_actor_updates else
              'hard_deadline' if utc_now() >= deadline else 'training_loop_budget')
    return dict(status='bounded_production_resume_completed', started_at_utc=started,
        finished_at_utc=utc_now().isoformat(), process_training_call_wall_s=wall_elapsed,
        actual_training_loop_s=loop_elapsed, stop_reason=reason,
        target_reached_exactly=counts['actor'] == args.stop_actor_updates,
        final_checkpoint=str(final_path.resolve()), final_sha256=sha(final_path),
        final_optimizer_counts=counts,
        additional_optimizer_counts={key:counts[key]-before[key] for key in counts},
        transitions=final['transitions'], additional_transitions=final['transitions']-saved['transitions'],
        termination_mode=proof['termination_mode'],budget_kind=proof['budget_kind'],
        saved_actor_milestones=milestones,
        last_metrics=rows[-1], production_sources_unchanged=True, proof=proof)


def main():
    global ART
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, default=ART,
        help='Separate budget and source manifest for the authorized continuation')
    p.add_argument('--resume', type=Path)
    p.add_argument('--log-dir', type=Path)
    p.add_argument('--seconds', type=float)
    p.add_argument('--stop-actor-updates', type=int, choices=TARGETS)
    p.add_argument('--hard-deadline-utc')
    p.add_argument('--count-only', action='store_true',
        help='Stop only at actual actor count or numerical/runtime failure, without any time cutoff')
    p.add_argument('--save-actor-updates', nargs='+', type=int, default=[])
    p.add_argument('--selfcheck', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    ART = args.experiment_dir.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.selfcheck:
        required = ('resume','log_dir','stop_actor_updates') if args.count_only else (
            'resume','log_dir','seconds','stop_actor_updates','hard_deadline_utc')
        if any(getattr(args, key) is None for key in required):
            p.error('Missing required training options: '+', '.join(required))
        if args.count_only and (args.seconds is not None or args.hard_deadline_utc is not None):
            p.error('Count-only training forbids seconds and hard-deadline-utc')
    result = selfcheck() if args.selfcheck else run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
