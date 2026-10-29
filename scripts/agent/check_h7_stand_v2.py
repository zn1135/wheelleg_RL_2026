"""Check actual H7 stand-v2 observation/history/SDK code on a host; no hardware IO."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import binascii
from types import SimpleNamespace

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui_stand_v2.observations import StandV2ObservationState


HARNESS = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "rl_policy.h"
static uint64_t tick;
uint64_t Mono_Ns_Get(void) { return ++tick; }
static void check_recovery(void)
{
    rl_recovery_state_t state = {0};
    const float loaded[2] = {65.700435f,65.700435f};
    const float empty[2] = {0.f,0.f};
    const float one_wheel[2] = {0.f,131.40087f};
    RL_Recovery_Update(&state,1,1,1,.1f,empty,1.f);
    assert(state.phase == RL_RECOVERY_HOLD && state.hold_ms == 0);
    for (unsigned i=0;i<19;i++) { RL_Recovery_Update(&state,1,1,1,.1f,empty,1.f); }
    assert(state.phase == RL_RECOVERY_HOLD);
    RL_Recovery_Update(&state,1,1,1,.1f,empty,1.f);
    assert(state.phase == RL_RECOVERY_RECOVER && state.hold_ms == 200);
    for (unsigned i=0;i<4000;i++) { RL_Recovery_Update(&state,1,1,1,.1f,empty,1.f); }
    assert(state.phase == RL_RECOVERY_RECOVER && !state.has_stood);
    for (unsigned i=0;i<49;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f); }
    assert(state.phase == RL_RECOVERY_RECOVER && !state.has_stood);
    RL_Recovery_Update(&state,1,1,0,.22f,one_wheel,-1.f);
    assert(state.stable_ms == 0 && state.standing_ms == 0);
    for (unsigned i=0;i<50;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f); }
    assert(state.phase == RL_RECOVERY_NORMAL && state.has_stood && state.tracking_ms == 10);
    for (unsigned i=0;i<49;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f); }
    assert(state.tracking_ms == 500);
    RL_Recovery_Update(&state,1,1,0,.22f,empty,-1.f);
    assert(state.phase == RL_RECOVERY_RECOVER && state.tracking_ms == 0 && state.airborne_ms == 10);
    for (unsigned i=0;i<18;i++) { RL_Recovery_Update(&state,1,1,0,.22f,empty,-1.f); }
    assert(state.phase == RL_RECOVERY_RECOVER && state.airborne_ms == 190);
    RL_Recovery_Update(&state,1,1,0,.22f,empty,-1.f);
    assert(state.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f);
    assert(state.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&state,0,1,0,.22f,loaded,-1.f);
    assert(state.phase == RL_RECOVERY_IDLE && !state.has_stood);
    /* No force source may be interpreted as standing; invalid feedback resets the segment. */
    RL_Recovery_Update(&state,1,0,0,.22f,loaded,-1.f);
    assert(state.phase == RL_RECOVERY_IDLE && !state.support_valid);
    RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f);
    for (unsigned i=0;i<50;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-.8f); }
    assert(state.phase == RL_RECOVERY_NORMAL && !state.has_stood && state.tracking_ms == 0);
    for (unsigned i=0;i<50;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f); }
    assert(state.has_stood);
    for (unsigned i=0;i<100;i++) { RL_Recovery_Update(&state,1,1,1,.19f,loaded,-.5f); }
    assert(state.phase == RL_RECOVERY_RECOVER && state.fallen_ms == 1000);
    RL_Recovery_Update(&state,1,1,1,.19f,loaded,-.5f);
    assert(state.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&state,0,1,0,.22f,loaded,-1.f);
    RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f);
    for (unsigned i=0;i<50;i++) { RL_Recovery_Update(&state,1,1,0,.22f,loaded,-1.f); }
    RL_Recovery_Update(&state,1,1,1,.22f,loaded,.1f);
    assert(state.phase == RL_RECOVERY_FAILED);
    assert(fabsf(RL_Command_Height(.15f)-.22f)<1e-7f);
    assert(fabsf(RL_Command_Height(.35f)-.22f)<1e-7f);
}
int main(int argc, char **argv)
{
    assert(argc == 4 || argc == 5);
    check_recovery();
    FILE *input = fopen(argv[1], "rb"), *output = fopen(argv[2], "wb");
    FILE *network = fopen(argv[3], "wb");
    assert(input && output && network);
    machine_rl_cfg_t config = {0};
    config.observation.gyro_scale = .25f;
    config.observation.height_scale = 5.f;
    config.observation.joint_vel_scale = .05f;
    config.observation.obs_clip = 100.f;
    const float nominal[4] = {-.604720554f,.608324173f,.604720554f,-.608324173f};
    memcpy(config.nominal_pos, nominal, sizeof(nominal));
    rl_observation_state_t state = {0};
    rl_policy_t policy = {0};
    assert(RL_Policy_Init(&policy));
    float row[28], action[6];
    unsigned count = 0;
    if (argc == 5)
    {
        state.valid = 1u;
        while (fread(state.history, sizeof(state.history), 1, input) == 1)
        {
            assert(RL_Policy_Run(&policy, &state, action));
            assert(fwrite(action, sizeof(action), 1, network) == 1);
            assert(fwrite(policy.velocity, sizeof(policy.velocity), 1, network) == 1);
            count++;
        }
        assert(count == 64);
        fclose(input); fclose(output); fclose(network);
        return 0;
    }
    while (fread(row, sizeof(row), 1, input) == 1)
    {
        rl_observation_sample_t sample;
        rl_recovery_state_t recovery = {0};
        rc_command_t command = {0};
        if (row[27] != 0.f) { RL_Observation_Reset(&state); }
        memcpy(sample.gyro, row, 3 * sizeof(float));
        memcpy(sample.gravity, row + 3, 3 * sizeof(float));
        memcpy(sample.pos, row + 6, 4 * sizeof(float));
        memcpy(sample.vel, row + 10, 6 * sizeof(float));
        memcpy(state.last_action, row + 16, 6 * sizeof(float));
        command.vel = row[22]; command.yaw = 1.5f; command.len = .3f;
        recovery.phase = (rl_recovery_phase_t)row[23];
        recovery.has_stood = (uint8_t)row[24]; recovery.tracking_ms = (uint16_t)row[25];
        uint8_t valid = RL_Observation_Build(&state, &config, &sample, &command, &recovery, (uint8_t)row[26]);
        assert(fwrite(state.obs, sizeof(state.obs), 1, output) == 1);
        assert(fwrite(state.history, sizeof(state.history), 1, output) == 1);
        const float flags[2] = {state.command_velocity, (float)valid};
        assert(fwrite(flags, sizeof(flags), 1, output) == 1);
        uint8_t ran = RL_Policy_Run(&policy, &state, action);
        assert(ran == valid);
        assert(fwrite(action, sizeof(action), 1, network) == 1);
        assert(fwrite(policy.velocity, sizeof(policy.velocity), 1, network) == 1);
        count++;
    }
    assert(count == 80);
    /* Invalid graph output is rejected at the SDK boundary, with zero action. */
    state.valid = 1u; state.history[0] = NAN;
    assert(!RL_Policy_Run(&policy, &state, action));
    for (unsigned i = 0; i < 6; i++) { assert(action[i] == 0.f); }
    fclose(input); fclose(output); fclose(network);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h7-root', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--reference-vectors', type=Path, required=True)
    parser.add_argument('--host-runtime', type=Path, required=True,
                        help='ST inspector/workspace directory with include/ and lib/static/')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    args.out.mkdir(parents=True)
    h7, out, runtime = args.h7_root.resolve(), args.out.resolve(), args.host_runtime.resolve()
    (out / 'main.h').write_text('#define __HAL_RCC_CRC_CLK_ENABLE() ((void)0)\n')
    (out / 'mono_ns.h').write_text('#include <stdint.h>\nuint64_t Mono_Ns_Get(void);\n')
    (out / 'check.c').write_text(HARNESS)
    sources = [h7 / 'imcalib/Algorithm/rl_observation.c', h7 / 'imcalib/Algorithm/rl_policy.c']
    sources += [h7 / ('X-CUBE-AI/App/' + n) for n in
                ('networkzn1.c', 'networkzn1_data.c', 'networkzn1_data_params.c')]
    includes = [out, runtime / 'include', h7 / 'X-CUBE-AI/App']
    includes += [h7 / ('imcalib/' + n) for n in ('common', 'config', 'Algorithm')]
    command = [os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra']
    command += ['-I' + str(p) for p in includes]
    command += [str(out / 'check.c')] + [str(p) for p in sources]
    command += ['-Wl,--start-group'] + [str(runtime / ('lib/static/' + n)) for n in
                                      ('libruntime.a', 'libst_cmsis_nn.a', 'libcmsis-nn.a')]
    command += ['-Wl,--end-group', '-lm', '-o', str(out / 'check')]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'build.log').write_text(result.stdout)
    result.check_returncode()
    rng = np.random.default_rng(5999)
    rows = rng.uniform(-3., 3., (80, 28)).astype(np.float32)
    rows[:, 23:28] = [3, 1, 500, 1, 0]
    rows[:, 10:16] *= 500  # Exercise the new scale-before-clip velocity definition.
    rows[0, 27] = 1
    rows[7, 23] = 2       # Recover must immediately clear vx while preserving the FIFO.
    rows[8, 25] = 499     # The last unready policy tick is still zero.
    rows[9, 24] = 0       # NORMAL alone does not establish first-stand success.
    rows[10, 23] = 4      # Failed state rejects inference.
    rows[11, 26] = 0      # Invalid feedback clears history initialization.
    rows[15, 27] = 1      # Explicit exit/re-entry fills all five slots again.
    rows[16:70, 22] = 1.5
    rows[70:, 22] = -1.5
    rows[2, 16:22] = [-300, 300, -.0, 200, -200, 1]
    rows.tofile(out / 'inputs.bin')
    result = subprocess.run([str(out / 'check'), str(out / 'inputs.bin'),
                             str(out / 'observations.bin'), str(out / 'network.bin')],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'run.log').write_text(result.stdout)
    result.check_returncode()
    actual = np.fromfile(out / 'observations.bin', np.float32).reshape(80, 194)
    history, velocity, initialized = np.zeros((5, 32), np.float32), np.float32(0), False
    expected = []
    nominal = np.array([-.604720554, .608324173, .604720554, -.608324173], np.float32)
    for row in rows:
        if row[27]:
            history.fill(0); velocity = np.float32(0); initialized = False
        valid = row[26] and row[23] != 4
        if not valid:
            initialized = False; velocity = np.float32(0)
            expected.append(np.r_[np.zeros(32, np.float32), history.ravel(), velocity, 0])
            continue
        ready = row[23] == 3 and row[24] and row[25] >= 500
        velocity = (np.float32(velocity + np.clip(np.clip(row[22], -.5, .5) - velocity,
                                                 np.float32(-.01), np.float32(.01)))
                    if ready else np.float32(0))
        # Call the actual training actor constructor, with synthetic already sampled sensors.
        state = object.__new__(StandV2ObservationState)
        sensors = torch.from_numpy(np.r_[row[:6], row[6:10] - nominal,
                                        np.zeros(2, np.float32), row[10:16]][None])
        state.delay = SimpleNamespace(read=lambda: sensors)
        state.cfg = SimpleNamespace(noise=False)
        command_state = SimpleNamespace(command=torch.tensor([[velocity, 0., .22]], dtype=torch.float32))
        state.env = SimpleNamespace(device='cpu', command_manager=SimpleNamespace(get_term=lambda name: command_state))
        state.action = SimpleNamespace(raw_action=torch.from_numpy(row[16:22][None]))
        mode = torch.zeros(1, 7); mode[0, 0 if row[23] == 3 else 3] = 1
        state.mode = lambda: mode
        frame = state.actor().numpy()[0]
        history = np.tile(frame, (5, 1)) if not initialized else np.vstack((history[1:], frame))
        initialized = True
        expected.append(np.r_[frame, history.ravel(), velocity, 1])
    error = float(np.max(np.abs(actual - np.asarray(expected, np.float32))))
    if error > 1e-5:
        raise AssertionError(f'Training observation/history disagreement: {error}')
    from onnx.reference import ReferenceEvaluator
    evaluator = ReferenceEvaluator(str(args.onnx))
    network = np.fromfile(out / 'network.bin', np.float32).reshape(80, 9)
    boundary_errors = []
    for i, data in enumerate(actual):
        if data[-1]:
            action, speed = evaluator.run(None, {'observation_history': data[None, 32:192]})
            reference = np.r_[action[0], speed[0]]
            difference = np.abs(network[i] - reference)
            boundary_errors.append(float(difference.max()))
    # 边界值用于精确核对观测/FIFO和有限性；数值交付仍使用导出器的固定参考集与原1e-4门槛。
    vectors = np.load(args.reference_vectors)
    vectors['observation_history'][:64].astype(np.float32).tofile(out / 'reference-inputs.bin')
    result = subprocess.run([str(out / 'check'), str(out / 'reference-inputs.bin'),
                             str(out / 'reference-unused.bin'), str(out / 'reference-network.bin'), 'reference'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'reference-run.log').write_text(result.stdout)
    result.check_returncode()
    network = np.fromfile(out / 'reference-network.bin', np.float32).reshape(64, 9)
    expected_network = np.concatenate((vectors['actions'][:64], vectors['velocity'][:64]), axis=1)
    network_error = float(np.max(np.abs(network - expected_network)))
    if network_error > 1e-4:
        raise AssertionError(f'Actual SDK wrapper disagrees with checkpoint references: {network_error}')
    import check_real2sim_huananhu as mirror_check
    mirror_check.H7 = h7
    packer_directory = out / 'telemetry'
    packer_directory.mkdir()
    raw, telemetry_check = mirror_check.c_packets(packer_directory, usb=True)
    mirror = mirror_check.mirror
    frame = mirror.decode_frame(raw)
    row = mirror.telemetry_row(frame)
    assert row['observation_dim'] == 32 and row['command_leg_length_m'] is None
    assert row['command_body_height_m'] is not None and row['obs_32'] is None
    # Retain old wire layouts; split reads and CRC rejection must still work.
    parser_state = mirror.FrameParser()
    legacy_packet = None
    for version, count in ((1, 48), (2, 160), (3, 160)):
        data = frame[:count].copy(); data[4] = version
        data[-1] = binascii.crc_hqx(data[:-1].astype('<f4').tobytes(), 0xffff)
        packet = data.astype('<f4').tobytes() + mirror.TAIL
        if version == 1:
            legacy_packet = packet
        assert parser_state.feed(packet[:37]) == []
        decoded = parser_state.feed(packet[37:])
        assert len(decoded) == 1 and decoded[0][4] == version
        corrupt = bytearray(packet); corrupt[50] ^= 1
        assert not parser_state.feed(corrupt)
    assert parser_state.versions == {1: 1, 2: 1, 3: 1}
    legacy_directory = out / 'legacy-sessions'
    legacy_directory.mkdir()
    legacy_sessions = mirror_check.check_v2_sessions(legacy_directory, legacy_packet)
    producer_directory = out / 'policy-producer'
    producer_directory.mkdir()
    producer_check = mirror_check.check_policy_producer(producer_directory)
    report = dict(ok=True, samples=len(rows), observation_history_max_abs_diff=error,
                  sdk_checkpoint_reference_max_abs_diff=network_error, reference_samples=64, reference_atol=1e-4,
                  boundary_numeric_diagnostic=dict(max_abs_diff=max(boundary_errors),
                      acceptance='diagnostic only; 1e-4 absolute equivalence is not established for extreme inputs'),
                  telemetry=telemetry_check, wire_versions_accepted=parser_state.versions,
                  legacy_sessions=legacy_sessions,
                  policy_producer=producer_check,
                  recovery='Actual C: 200ms hold, first-stand recovery beyond 2s, dual-load reset, '
                           '500ms normal/first-stand, extra tracking hold, post-stand airborne/inversion/fall locks, exit/invalid reset',
                  scope='Actual observation C + SDK wrapper + generated network on ST host runtime; '
                        'training observation oracle, reset/FIFO/slew/output finite checks; '
                        'synthetic recovery inputs, no board timing or physical acceptance',
                  build_command=command,
                  sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources + [Path(__file__), Path(mirror.__file__), Path(mirror_check.__file__),
                                      h7 / 'imcalib/task/task_actuation.c', h7 / 'imcalib/task/task_policy.c',
                                      h7 / 'imcalib/task/vofa_debug.c', h7 / 'imcalib/task/robot_control.c',
                                      h7 / 'imcalib/Algorithm/rl_observation.h', h7 / 'imcalib/Algorithm/rl_policy.h',
                                      h7 / 'imcalib/common/rl_types.h', h7 / 'imcalib/config/machine_config.c',
                                      h7 / 'imcalib/config/machine_config.h']})
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('build_command', 'sources_sha256')}))


if __name__ == '__main__':
    main()
