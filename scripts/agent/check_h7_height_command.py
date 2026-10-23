#!/usr/bin/env python3
"""Check real H7 RC/observation C for training-height reachability and mode regressions."""
import argparse
import ast
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
#include <stdio.h>
#include <string.h>
#include "rc_command.h"
#include "rl_observation.h"
#include "wheel_bench.h"
volatile wheel_bench_state_t wheel_bench;
void Rc_Command_Update_Before(rc_command_t *, const dr16_t *, float);
DEADBAND_FUNCTION
static void observation_height(float command_height, float expected)
{
    rc_command_t cmd = {0};
    rl_observation_state_t obs = {0};
    rl_observation_sample_t sample = {0};
    rl_recovery_state_t recovery = {.phase = RL_RECOVERY_NORMAL};
    machine_rl_cfg_t config = machine_table[0].rl;
    cmd.len = command_height;
    sample.gravity[2] = -1.f;
    assert(RL_Observation_Build(&obs, &config, &sample, &cmd, &recovery, 1u));
    assert(fabsf(obs.obs[RL_OBS_CMD_HEIGHT] / config.observation.height_scale - expected) < 1e-7f);
    sample.pos[0] = .4f; sample.gravity[0] = .2f;
    assert(RL_Observation_Build(&obs, &config, &sample, &cmd, &recovery, 1u));
    assert(fabsf(obs.obs[RL_OBS_CMD_HEIGHT] / config.observation.height_scale - expected) < 1e-7f);
}
int main(void)
{
    const uint8_t switches[] = {DR16_SW_UP, DR16_SW_MID, DR16_SW_DOWN};
    const int16_t wheels[] = {-660, -20, 0, 20, 660};
    unsigned regression_cases = 0u;
    for (unsigned l = 0; l < 3; l++) {
        for (unsigned r = 0; r < 3; r++) {
            for (unsigned reserved = 0; reserved < 2; reserved++) {
                for (unsigned online = 0; online < 2; online++) {
                    for (unsigned w = 0; w < 5; w++) {
                        dr16_t raw = {.online = online, .s1 = switches[l], .s2 = switches[r], .wheel = wheels[w]};
                        rc_command_t before = {.len = .29f}, after = before;
                        wheel_bench.reserved = reserved;
                        Rc_Command_Update_Before(&before, &raw, .001f);
                        Rc_Command_Update(&after, &raw, .001f);
                        /* 模式/许可/速度不变；只允许大机RL的高度限幅不同。 */
                        assert(before.vel == after.vel && before.yaw == after.yaw);
                        assert(before.s1 == after.s1 && before.s2 == after.s2 && before.online == after.online);
                        assert(before.mode.strategy == after.mode.strategy && before.mode.engage == after.mode.engage && before.mode.rc_enable == after.mode.rc_enable);
                        if (MACHINE_DEFAULT != 0u || after.mode.strategy != CTRL_STRATEGY_RL || !online) {
                            assert(before.len == after.len);
                        }
                        regression_cases++;
                    }
                }
            }
        }
    }
    wheel_bench.reserved = 0u;
#if MACHINE_DEFAULT == MACHINE_ID_BIG_WHEELLEG
    dr16_t raw = {.online = 1u, .s1 = DR16_SW_UP, .s2 = DR16_SW_MID};
    rc_command_t cmd = {.len = .15f};
    const float min_leg = MACHINE_RL_HEIGHT_MIN - MACHINE_RL_LEG_TO_HEIGHT;
    const float max_leg = MACHINE_RL_HEIGHT_MAX - MACHINE_RL_LEG_TO_HEIGHT;
    const float heights[] = {.2165f, .25f, .27f, .34f, .3664f};
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(fabsf(cmd.len - min_leg) < 1e-7f);
    for (unsigned i = 0; i < 5; i++) {
        cmd.len = heights[i] - MACHINE_RL_LEG_TO_HEIGHT;
        Rc_Command_Update(&cmd, &raw, .001f);
        assert(fabsf(cmd.len + MACHINE_RL_LEG_TO_HEIGHT - heights[i]) < 1e-7f);
        observation_height(cmd.len, heights[i]);
    }
    observation_height(.3f, .33648f);
    cmd.len = max_leg; raw.wheel = 660;
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(cmd.len == max_leg);
    raw.wheel = -660;
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(cmd.len < max_leg && fabsf(cmd.len - (max_leg - .0002f)) < 1e-7f);
    cmd.len = min_leg;
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(cmd.len == min_leg);
    raw.wheel = 0; raw.s1 = DR16_SW_MID; cmd.len = .3664f;
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(fabsf(cmd.len - .31f) < 1e-7f);
    raw.online = 0;
    Rc_Command_Update(&cmd, &raw, .001f);
    assert(cmd.len == machine->rc.len_init);
    observation_height(.15f, .2165f);
    observation_height(.4f, .3664f);
    /* 低位不再被固定0.20m腿投影门槛锁住；倾倒和2000ms失败保护保留。 */
    rl_recovery_state_t recovery = {0};
    RL_Recovery_Update(&recovery, 1u, 1u, 1u, .1f, min_leg);
    for (unsigned i = 0; i < 200; i++) { RL_Recovery_Update(&recovery, 1u, 1u, 1u, .1f, min_leg); }
    assert(recovery.phase == RL_RECOVERY_RECOVER);
    for (unsigned i = 0; i < 500; i++) { RL_Recovery_Update(&recovery, 1u, 1u, 0u, .18f, min_leg); }
    assert(recovery.phase == RL_RECOVERY_NORMAL && recovery.ground_start);
    RL_Recovery_Update(&recovery, 0u, 1u, 0u, .18f, min_leg);
    assert(recovery.phase == RL_RECOVERY_IDLE);
    RL_Recovery_Update(&recovery, 1u, 1u, 1u, .1f, min_leg);
    for (unsigned i = 0; i < 2001; i++) { RL_Recovery_Update(&recovery, 1u, 1u, 1u, .1f, min_leg); }
    assert(recovery.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&recovery, 0u, 1u, 0u, .27f, .27f);
    RL_Recovery_Update(&recovery, 1u, 1u, 0u, .199f, .27f);
    assert(recovery.phase == RL_RECOVERY_HOLD);
#else
    observation_height(.15f, .27f);
    observation_height(.4f, .34f);
#endif
    printf("{\"machine\":%u,\"mode_regression_cases\":%u,\"observation_height_limits\":[%.7g,%.7g]}\n", MACHINE_DEFAULT, regression_cases, MACHINE_RL_HEIGHT_MIN, MACHINE_RL_HEIGHT_MAX);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h7-root', type=Path, required=True)
    parser.add_argument('--before-rc', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; use a new path')
    h7 = args.h7_root.resolve()
    training = ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_mdp.py'
    tree = ast.parse(training.read_text())
    height_range = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HEIGHT_COMMAND_RANGE' for t in node.targets))
    header = (h7 / 'imcalib/config/machine_config.h').read_text().split('#elif')[0]
    for label, value in zip(('MIN', 'MAX'), height_range):
        actual = float(re.search(rf'#define MACHINE_RL_HEIGHT_{label}\s+([\d.]+)f', header).group(1))
        assert actual == value
    # 用当前共享XML和真实轮碰撞网格复核直立目标的换算，不写回机械参数。
    sys_path = str(ROOT / 'mjlab_training/src')
    import sys
    sys.path.insert(0, sys_path)
    sys.path.insert(0, str(ROOT / 'scripts/agent'))
    import mujoco
    import numpy as np
    from scipy.optimize import brentq
    from ct_mjlab.envs.chuanliantui.model import load_model, MODEL_XML
    from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics
    from inspect_multimode_leg_length import standing_endpoint
    model = load_model()
    hip_z = float(np.mean([model.body(name).pos[2] for name in ('lf0', 'rf0')]))
    offset = .06 - hip_z
    assert abs(offset - .03648) < 1e-9
    geometry = []
    data = mujoco.MjData(model)
    for length in (.18, .23, .28, .33):
        knee = brentq(lambda k: _front_kinematics(model, data, k)[0] - length, *model.joint('lf1').range)
        point = standing_endpoint(model, knee)
        height = point['base_reference_height_on_wheel_plane_m']
        error = abs(length + offset - height)
        assert error < 1e-4
        geometry.append({'leg_length_m': length, 'mesh_based_base_height_m': height,
                         'nominal_command_height_m': length + offset, 'absolute_error_m': error})
    driver = h7 / 'imcalib/drivers/dr16.c'
    deadband = re.search(r'int16_t DR16_Deadline\(.*?^\}', driver.read_text(), re.M | re.S).group()
    directory = args.out.parent / 'height-host'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'dr16.h').write_text('#include "rc_types.h"\nint16_t DR16_Deadline(int16_t, uint16_t);\n')
    source = directory / 'height.c'
    source.write_text(HARNESS.replace('DEADBAND_FUNCTION', deadband))
    compiler = os.environ['CC']
    results, commands = [], []
    c_files = ['task/rc_command.c', 'Algorithm/rl_observation.c', 'config/machine_config.c',
               'Algorithm/lqr_gain_table.c', 'Algorithm/lqr_gain_big.c', 'Algorithm/lqr_gain_small.c']
    for machine in (0, 1):
        common = [compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', f'-DMACHINE_DEFAULT={machine}',
                  '-I' + str(directory.resolve())] + ['-I' + str(h7 / 'imcalib' / p) for p in ('config', 'common', 'Algorithm', 'task/inc')]
        before = directory / f'before-{machine}.o'
        command = common + ['-DRc_Command_Update=Rc_Command_Update_Before', '-c', str(args.before_rc), '-o', str(before)]
        subprocess.run(command, check=True)
        binary = directory / f'check-{machine}'
        command = common + [str(source), str(before)] + [str(h7 / 'imcalib' / p) for p in c_files] + ['-lm', '-o', str(binary)]
        commands.append(command)
        built = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / f'build-{machine}.log').write_text(built.stdout)
        built.check_returncode()
        checked = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
        results.append(json.loads(checked.stdout))
    paths = [training, driver, Path(__file__), args.before_rc, MODEL_XML] + [h7 / 'imcalib' / p for p in c_files] + [h7 / 'imcalib/config/machine_config.h']
    report = {'ok': True, 'scope': 'Actual RC/observation/config/gain C; extracted actual deadband and wheel-bench state stub; no board or motor behavior',
              'training_height_range': height_range, 'model_upright_geometry': geometry,
              'model_command_offset_m': offset, 'results': results, 'commands': commands,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': True, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
