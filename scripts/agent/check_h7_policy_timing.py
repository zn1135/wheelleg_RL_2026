#!/usr/bin/env python3
"""Host check of H7 policy divider and recovery timers; no target connection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "rl_observation.h"
static unsigned tick_ms, releases, last_release;
static void *policy_tick_sem_handle;
static void Latency_Tick(unsigned channel) { assert(channel == 1u); }
static void osSemaphoreRelease(void *handle)
{
    (void)handle;
    assert(tick_ms - last_release == 10u);
    last_release = tick_ms;
    releases++;
}
POLICY_DIVIDER
int main(void)
{
    unsigned pd_steps = 0u;
    rl_recovery_state_t state = {0};
    assert(MACHINE_POLICY_DIV == 10u);
    assert(fabsf(MACHINE_POLICY_DT - .01f) < 1e-7f);
    for (tick_ms = 1u; tick_ms <= 1000u; tick_ms++) {
        Policy_Tick_Div();
        if (tick_ms % MACHINE_RL_CTRL_DIV == 0u) { pd_steps++; }
    }
    assert(releases == 100u);
#if MACHINE_DEFAULT == MACHINE_ID_BIG_WHEELLEG
    assert(pd_steps == 500u);
#else
    assert(pd_steps == 1000u);
#endif
    RL_Recovery_Update(&state, 1u, 1u, 1u, .1f, .27f);
    assert(state.phase == RL_RECOVERY_HOLD && state.hold_ms == 0u);
    for (unsigned i = 1u; i < 200u; i++) {
        RL_Recovery_Update(&state, 1u, 1u, 1u, .1f, .27f);
        assert(state.phase == RL_RECOVERY_HOLD);
    }
    RL_Recovery_Update(&state, 1u, 1u, 1u, .1f, .27f);
    assert(state.phase == RL_RECOVERY_RECOVER && state.hold_ms == 200u);
    for (unsigned i = 1u; i < 500u; i++) {
        RL_Recovery_Update(&state, 1u, 1u, 0u, .27f, .27f);
        assert(state.phase == RL_RECOVERY_RECOVER);
    }
    RL_Recovery_Update(&state, 1u, 1u, 0u, .27f, .27f);
    assert(state.phase == RL_RECOVERY_NORMAL && state.ground_start == 1u);
    RL_Recovery_Update(&state, 0u, 1u, 0u, .27f, .27f);
    assert(state.phase == RL_RECOVERY_IDLE && state.ground_start == 0u);
    printf("{\"machine\":%u,\"policy_calls_per_1000_ticks\":%u,\"pd_divider_count\":%u,\"hold_ms\":200,\"stable_ms\":500}\n", MACHINE_DEFAULT, releases, pd_steps);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h7-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; use a new path')
    h7 = args.h7_root.resolve()
    source = h7 / 'imcalib/task/robot_control.c'
    # 分频函数直接取实际源码，外设/信号量仅用宿主替身计数。
    match = re.search(r'void Policy_Tick_Div\(void\)\s*\{.*?^\}', source.read_text(), re.M | re.S)
    if match is None:
        raise ValueError('Policy_Tick_Div not found')
    directory = args.out.parent / 'policy-timing-host'
    directory.mkdir(parents=True, exist_ok=False)
    harness = directory / 'timing.c'
    harness.write_text(HARNESS.replace('POLICY_DIVIDER', match.group()))
    compiler = os.environ['CC']
    results, commands = [], []
    for machine in (0, 1):
        binary = directory / f'check-{machine}'
        command = [compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                   f'-DMACHINE_DEFAULT={machine}', '-I' + str(h7 / 'imcalib/config'),
                   '-I' + str(h7 / 'imcalib/common'), '-I' + str(h7 / 'imcalib/Algorithm'),
                   str(harness), str(h7 / 'imcalib/Algorithm/rl_observation.c'), '-lm', '-o', str(binary)]
        commands.append(command)
        built = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / f'build-{machine}.log').write_text(built.stdout)
        built.check_returncode()
        checked = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
        results.append(json.loads(checked.stdout))
    paths = [source, h7 / 'imcalib/config/machine_config.h',
             h7 / 'imcalib/Algorithm/rl_observation.c', Path(__file__)]
    report = {'ok': True, 'scope': 'Extracted actual policy divider, actual recovery C and config; IRQ/RTOS stubs, not measured board timing or complete actuation scheduling',
              'results': results, 'commands': commands,
              'sources_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': True, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
