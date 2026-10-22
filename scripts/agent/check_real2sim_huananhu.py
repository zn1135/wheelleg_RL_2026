#!/usr/bin/env python3
"""Check actual H7 mirror packing, damaged streams and measured-motor kinematics.

Uses a native C harness for the actual vofa_debug.c (only hardware/RTOS are stubs),
known XML/pose-bank poses, and a pseudo-terminal for the real serial reader. Optional
--build-h7 compiles/links both machine variants from the current eIDE source list,
writing all outputs outside the selected --h7-root. No board is flashed or controlled.
"""
from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H7 = ROOT / "deployment/H7_RL"
sys.path.insert(0, str(ROOT / "sim2sim"))
import real2sim_huananhu as mirror
import h7_serial as link

STUB = r'''
#ifndef MIRROR_STUB
#define MIRROR_STUB
#include <stdint.h>
typedef struct { uint8_t seen; uint64_t published_ns; } feedback_meta_t;
typedef struct { uint8_t seen; uint64_t received_ns; } feedback_sample_t;
typedef struct { float quat[4], gyro_rad_s[3]; uint8_t online; } imu_t;
typedef struct { float pos_continuous_rad[4], vel_rad_s[4], trq_nm[4]; uint64_t parsed_rx_ns[4]; uint8_t online[4]; } dm_t;
typedef struct { float angle_total_rad[2], vel_rad_s[2]; int16_t current_raw[2]; uint64_t parsed_rx_ns[2]; uint8_t online[2]; } dji_t;
typedef struct { struct { uint8_t strategy, rc_enable, engage; } mode; uint8_t online; } rc_t;
typedef struct { struct { uint8_t valid; } output; } leg_t;
typedef struct {
    imu_t imu; struct { dm_t dm; dji_t dji; } motor;
    rc_t rc; leg_t left,right;
    feedback_meta_t imu_meta,motor_meta; uint64_t captured_ns;
    feedback_sample_t imu_sample,rc_sample;
} robot_feedback_t;
typedef struct {
    float obs[39], action[6], command[3];
    uint32_t sequence,sample_tick_ms,published_tick_ms,run_us; uint8_t valid;
} mirror_policy_snapshot_t;
typedef struct {
    float target[6],action[6],pos[4],vel[6],torque[6];
    uint32_t sequence,tick_ms,policy_sequence; uint8_t flags,valid;
} mirror_output_snapshot_t;
extern mirror_policy_snapshot_t mirror_policy_snapshot;
extern mirror_output_snapshot_t mirror_output_snapshot;
typedef struct { struct { uint8_t phase; } recovery; struct { uint8_t ready; } policy; } rl_control_state_t;
typedef struct { uint32_t reset_count; } action_state_t;
extern rl_control_state_t rl_control;
extern action_state_t action_state;
extern struct robot_t { uint8_t motor_enabled,fallen; } robot_state;
extern volatile uint32_t ctrl_fault;
uint32_t HAL_GetTick(void);
uint8_t Robot_Feedback_Read(robot_feedback_t *out);
uint8_t Dm_Is_Enabled(uint8_t i);
uint8_t output_task_lqr_engaged(void);
uint8_t output_task_rl_engaged(void);
uint8_t Vofa_Send(const float *data,uint8_t n);
#define DM_MOTOR_NUM 4u
#define DJI_MOTOR_NUM 2u
#define DM_MOTOR_LEG_F_LFT 0u
#define DM_MOTOR_LEG_B_LFT 1u
#define DM_MOTOR_LEG_F_RGT 2u
#define DM_MOTOR_LEG_B_RGT 3u
#define MACHINE_DEFAULT 0u
#define VOFA_MAX_CH 48u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
#endif
'''


def c_packets(directory: Path, usb=False):
    # Check the actual sender too: hardware stubs must not hide format drift.
    sender = (H7 / "imcalib/bsp/Vofa_send.c").read_text()
    header = (H7 / "imcalib/bsp/Vofa_send.h").read_text()
    channels = int(re.search(r"#define\s+VOFA_MAX_CH\s+(\d+)", header).group(1))
    tail_text = re.search(r"tail\[4\]\s*=\s*\{([^}]+)\}", sender).group(1)
    tail = bytes(int(value.strip(), 0) for value in tail_text.split(","))
    assert channels in (48, 160)
    version = 2 if channels == 160 else 1
    frame_bytes = channels * 4 + len(tail)
    assert frame_bytes == mirror.FRAME_BYTES_BY_VERSION[version]
    assert tail == mirror.TAIL, "actual Vofa_Send tail differs from mirror protocol"
    stub = directory / "stubs"
    stub.mkdir()
    (stub / "robot_state.h").write_text(STUB.replace("#define VOFA_MAX_CH 48u", f"#define VOFA_MAX_CH {channels}u"))
    for name in ("vofa_debug.h", "dm.h", "Vofa_send.h", "dr16.h", "FreeRTOS.h", "task.h"):
        (stub / name).write_text('#include "robot_state.h"\n')
    response = H7 / "imcalib/Algorithm/wheel_response.c"
    if response.exists():
        (stub / "machine_config.h").write_text('#define MACHINE_ID_BIG_WHEELLEG 0u\n#define MACHINE_DEFAULT 0u\n')
    probe = directory / "probe.c"
    probe.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "robot_state.h"
#include "jetson_uart.h"
#include "''' + str(H7 / "imcalib/task/vofa_debug.c") + r'''"
jetson_uart_status_t jetson_uart_status = {0xffffffffu, 0x12345678u, 7u, 0u, 0u};
struct robot_t robot_state = {0,0};
mirror_policy_snapshot_t mirror_policy_snapshot;
mirror_output_snapshot_t mirror_output_snapshot;
rl_control_state_t rl_control;
action_state_t action_state;
volatile uint32_t ctrl_fault = 0;
static robot_feedback_t source;
static uint32_t tick;
static unsigned attempts;
static uint8_t ready = 1;
static FILE *output;
uint32_t HAL_GetTick(void) { return tick; }
uint8_t Dm_Is_Enabled(uint8_t i) { return 0; }
uint8_t output_task_lqr_engaged(void) { return 0; }
uint8_t output_task_rl_engaged(void) { return 0; }
uint8_t Robot_Feedback_Read(robot_feedback_t *out) {
    *out = source;
    out->captured_ns = (uint64_t)tick * 1000000u;
    out->imu_meta.published_ns = out->captured_ns - 2000000u;
    out->motor_meta.published_ns = out->captured_ns - 3000000u;
    for (unsigned i=0;i<4;i++) out->motor.dm.parsed_rx_ns[i] = out->captured_ns > (i+1u)*1000000u ? out->captured_ns - (i+1u)*1000000u : 0;
    for (unsigned i=0;i<2;i++) out->motor.dji.parsed_rx_ns[i] = out->captured_ns > (i+5u)*1000000u ? out->captured_ns - (i+5u)*1000000u : 0;
    out->imu_sample.seen=out->rc_sample.seen=1;
    out->imu_sample.received_ns=out->captured_ns > 7000000u ? out->captured_ns-7000000u : 0;
    out->rc_sample.received_ns=out->captured_ns > 8000000u ? out->captured_ns-8000000u : 0;
    return 1;
}
uint8_t Vofa_Send(const float *data,uint8_t n) {
    const uint8_t tail[4] = {0,0,0x80,0x7f};
    assert(n == VOFA_MAX_CH); attempts++;
    if (!ready) { return 0; }
    assert(fwrite(data,4,n,output)==n);
    assert(fwrite(tail,1,4,output)==4);
    return 1;
}
int main(int argc,char **argv) {
    float halves[2];
    assert(Mirror_Crc16((const uint8_t *)"123456789",9)==0x29b1);
    Mirror_U32(halves,0xffffffffu);
    assert(halves[0]==65535 && halves[1]==65535);
    output=fopen(argv[1],"wb"); assert(output);
    source.imu.online=source.rc.online=1;
    source.imu.quat[0]=1;
    source.imu_meta.seen=source.motor_meta.seen=1;
    source.rc.mode.strategy=0; source.rc.mode.rc_enable=source.rc.mode.engage=1;
    mirror_policy_snapshot.valid=mirror_output_snapshot.valid=1;
    mirror_policy_snapshot.sequence=0x12345u; mirror_policy_snapshot.sample_tick_ms=6;
    mirror_policy_snapshot.published_tick_ms=8; mirror_policy_snapshot.run_us=250;
    mirror_policy_snapshot.command[0]=.2f; mirror_policy_snapshot.command[1]=-.3f; mirror_policy_snapshot.command[2]=.31f;
    mirror_output_snapshot.sequence=0x23456u; mirror_output_snapshot.tick_ms=9;
    mirror_output_snapshot.policy_sequence=0x12345u; mirror_output_snapshot.flags=53u;
    rl_control.recovery.phase=3; rl_control.policy.ready=1; action_state.reset_count=9;
    for (int i=0;i<39;i++) mirror_policy_snapshot.obs[i]=100.f+i;
    for (int i=0;i<6;i++) {
        mirror_policy_snapshot.action[i]=-.1f*i; mirror_output_snapshot.action[i]=-.2f*i;
        mirror_output_snapshot.target[i]=10.f+i; mirror_output_snapshot.vel[i]=4.f+i;
        mirror_output_snapshot.torque[i]=.1f*(i+1);
    }
    for (int i=0;i<4;i++) {
        source.motor.dm.online[i]=1;
        source.motor.dm.pos_continuous_rad[i]=.1f*(i+1);
        source.motor.dm.vel_rad_s[i]=i+2;
        source.motor.dm.trq_nm[i]=.5f*(i+1); mirror_output_snapshot.pos[i]=.3f*i;
    }
    for (int i=0;i<2;i++) {
        source.motor.dji.online[i]=1;
        source.motor.dji.angle_total_rad[i]=3.f+i;
        source.motor.dji.vel_rad_s[i]=5.f+i;
        source.motor.dji.current_raw[i]=i ? -14 : 13;
    }
    tick=1; Vofa_Debug_Send(); tick=9; Vofa_Debug_Send(); assert(attempts==0);
    tick=10; Vofa_Debug_Send(); tick=15; Vofa_Debug_Send(); assert(attempts==1);
    ready=0; tick=20; Vofa_Debug_Send(); assert(attempts==2);
    ready=1; tick=30; Vofa_Debug_Send();
    source.motor.dji.online[0]=0; tick=40; Vofa_Debug_Send();
    source.motor.dm.pos_continuous_rad[0]=INFINITY; tick=50; Vofa_Debug_Send();
    source.imu.online=0; tick=60; Vofa_Debug_Send();
    tick=0xfffffff0u; Vofa_Debug_Send(); tick=4; Vofa_Debug_Send();
    assert(attempts==8 && robot_state.motor_enabled==0 && ctrl_fault==0);
    fclose(output); return 0;
}
''')
    executable, data = directory / "probe", directory / "h7-frames.bin"
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-I", str(stub), "-I", str(H7 / "imcalib/drivers"),
                    "-I", str(H7 / "imcalib/Algorithm"), str(probe), *([str(response)] if response.exists() else []),
                    "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(data)], check=True)
    raw = data.read_bytes()
    frames = mirror.FrameParser().feed(raw)
    assert len(frames) == 7 and frames[0][4] == version
    assert [mirror.read_u32(f[6:8]) for f in frames] == list(range(1, 8))
    assert [mirror.read_u32(f[8:10]) for f in frames] == [10, 30, 40, 50, 60, 0xfffffff0, 4]
    assert frames[0][29] == 127 and np.array_equal(frames[0][30:32], [2, 3])
    assert np.allclose(frames[0][10:14], [.1, .2, .3, .4])
    assert np.array_equal(frames[0][14:22], [2, 3, 4, 5, 3, 4, 5, 6])
    assert np.array_equal(frames[0][22:29], [1, 0, 0, 0, 0, 0, 0])
    assert frames[2][29] == 95 and frames[2][18] == 0
    assert frames[3][29] == 93 and frames[3][10] == 0
    assert frames[4][29] == 92 and np.all(frames[4][22:29] == 0)
    expected_probe = np.zeros(7) if usb else [1, 65535, 65535, 0x5678, 0x1234, 7, 0]
    assert np.array_equal(frames[0][32:39], expected_probe)
    assert np.all(frames[0][39:47] == 0)
    if version == 2:
        assert frames[0][47] == 3
        assert mirror.read_u32(frames[0][48:50]) == 0x12345
        assert mirror.read_u32(frames[0][50:52]) == 6 and mirror.read_u32(frames[0][52:54]) == 8
        assert np.allclose(frames[0][54:57], [.2, -.3, .31])
        assert np.array_equal(frames[0][57:96], np.arange(100, 139))
        assert np.allclose(frames[0][96:102], -.1 * np.arange(6)) and frames[0][102] == 250
        assert mirror.read_u32(frames[0][103:105]) == 0x23456
        assert mirror.read_u32(frames[0][105:107]) == 9 and mirror.read_u32(frames[0][107:109]) == 0x12345
        assert frames[0][109] == 53 and np.array_equal(frames[0][110:116], np.arange(10, 16))
        assert np.allclose(frames[0][116:122], -.2 * np.arange(6))
        assert np.allclose(frames[0][122:126], .3 * np.arange(4))
        assert np.array_equal(frames[0][126:132], np.arange(4, 10))
        assert np.allclose(frames[0][132:138], .1 * np.arange(1, 7))
        assert np.array_equal(frames[0][138:146], np.arange(1, 9))
        assert np.allclose(frames[0][146:150], .5 * np.arange(1, 5))
        assert np.array_equal(frames[0][150:159], [13, -14, 0, 7, 3, 1, 9, 0, int(response.exists())])
    return raw[:frame_bytes], {"actual_c_frames": len(frames), "protocol_version": version,
                      "cadence_busy_wrap_invalid_sources": True,
                      "actual_sender_channel_count_and_tail": True}


def check_wheel_response(directory):
    """Actual wheel C control with pure type headers; no motor drivers."""
    root = H7/'imcalib'
    module = root/'Algorithm/wheel_response.c'
    source = directory/'wheel-response.c'
    source.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "rl_torque.h"
#if HAVE_WHEEL_RESPONSE
#include "wheel_response.h"
#endif
int main(void) {
    machine_rl_cfg_t cfg={0};
    rl_motor_state_t motor={0};
    rl_torque_state_t state;
    torque_output_t out;
    float action[6]={0};
    const float speeds[]={-100.f,-3.f,-.5f,-.01f,0.f,.01f,.5f,3.f,100.f};
    const int signs[6]={1,1,-1,-1,1,-1};
    cfg.torque.pos_scale=.5f; cfg.torque.position_limit=3.14f;
    cfg.torque.wheel_vel_scale=10.f; cfg.torque.wheel_speed_limit=150.f;
    cfg.torque.leg_torque_limit=40.f; cfg.torque.wheel_torque_limit=3.9f;
    for(unsigned i=0;i<6;i++) cfg.motor_sign[i]=signs[i];
    for(unsigned i=0;i<4;i++) { cfg.torque.p_gains[i]=60.f;cfg.torque.d_gains[i]=2.f; }
    cfg.torque.wheel_gain[0]=cfg.torque.wheel_gain[1]=.1f;
    motor.valid=1;
#if HAVE_WHEEL_RESPONSE
    printf("META %u %.9g %.9g %.9g\n",wheel_response_config.mode_id,
        wheel_response_config.friction_nm[0],wheel_response_config.friction_nm[1],wheel_response_config.smoothing_rad_s);
#else
    printf("META 0 0 0 0.5\n");
#endif
    for(unsigned s=0;s<sizeof(speeds)/sizeof(speeds[0]);s++) for(int a=-1;a<=1;a++) {
        motor.vel[4]=speeds[s];motor.vel[5]=-speeds[s];
        action[0]=.2f;action[1]=-.2f;action[3]=-.2f;action[4]=.2f;
        action[2]=action[5]=2.f*(float)a;
        assert(RL_Torque_Compute(&motor,&cfg,action,&state,&out) && out.valid);
        assert(fabsf(out.dm[0]-6.f)<1e-6f && fabsf(out.dm[1]+6.f)<1e-6f);
        assert(fabsf(out.dm[2]-6.f)<1e-6f && fabsf(out.dm[3]+6.f)<1e-6f);
        for(unsigned i=0;i<2;i++) {
            float added=0.f;
#if HAVE_WHEEL_RESPONSE
            added=Wheel_Response_Compensation(i,motor.vel[4+i]);
#endif
            assert(fabsf(state.motor_torque[4+i])<=3.9f && out.dji[i]==signs[4+i]*state.motor_torque[4+i]);
            printf("ROW %u %.9g %.9g %.9g %.9g\n",i,motor.vel[4+i],state.target[4+i],
                added,state.motor_torque[4+i]);
        }
    }
    motor.valid=0;assert(!RL_Torque_Compute(&motor,&cfg,action,&state,&out) && !out.valid);
    motor.valid=1;motor.vel[4]=NAN;assert(!RL_Torque_Compute(&motor,&cfg,action,&state,&out) && !out.valid);
    motor.vel[4]=0;motor.vel[5]=INFINITY;assert(!RL_Torque_Compute(&motor,&cfg,action,&state,&out) && !out.valid);
    memset(&state,0xff,sizeof(state));RL_Torque_Reset(&state);
    rl_torque_state_t zero={0};assert(memcmp(&zero,&state,sizeof(state))==0);
    return 0;
}
''')
    maximum, checked = 0., 0
    for machine in (0, 1):
        binary = directory/f'wheel-response-{machine}'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-DMACHINE_DEFAULT='+str(machine), '-DHAVE_WHEEL_RESPONSE='+str(int(module.exists())),
                        '-I', str(root/'Algorithm'), '-I', str(root/'common'),
                        '-I', str(root/'config'), str(source), *([str(module)] if module.exists() else []), str(root/'Algorithm/rl_torque.c'),
                        '-lm', '-o', str(binary)], check=True)
        lines = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()
        meta = lines[0].split()
        mode = int(meta[1])
        assert mode == (1 if module.exists() and machine == 0 else 0)
        profile = mirror.WHEEL_RESPONSE_CATALOG[str(mode)]
        assert np.array_equal(np.asarray(meta[2:4], dtype=np.float32), np.asarray(profile['friction_nm'], dtype=np.float32))
        assert float(meta[4]) == profile['smoothing_rad_s']
        for line in lines[1:]:
            _, side, velocity, target, added, actual = line.split()
            side, velocity, target, added, actual = int(side), float(velocity), float(target), float(added), float(actual)
            expected_added = profile['friction_nm'][side]*np.tanh(velocity/profile['smoothing_rad_s'])
            expected = np.clip(.1*(target-velocity)+expected_added, -3.9, 3.9)
            maximum = max(maximum, abs(expected_added-added), abs(expected-actual))
            checked += 1
        assert maximum < 2e-6, maximum
    return {'present': module.exists(), 'actual_rl_torque_source': True, 'wheel_cases_checked': checked,
            'max_c_vs_python_torque_difference_nm': maximum, 'small_machine_original_controller': True,
            'leg_mapping_invalid_nan_inf_reset_and_final_clamp_checked': True,
            'catalog_matches_compiled_c_float_coefficients': True,
            'scope': 'native C algorithm only; no CAN, H7 scheduling or physical response verification'}


def check_usb_sender(directory: Path, raw: bytes):
    """Run the actual USB sender with CDC connection/completion/failure stubs."""
    stub = directory / "usb-stub"
    stub.mkdir()
    for name in ("main.h", "usart.h", "machine_config.h"):
        (stub / name).write_text('#include <stdint.h>\n#define MACHINE_VOFA_PORT 10u\n')
    capture = (H7 / 'imcalib/bsp/output_capture.h').exists()
    capture_stub = r'''
#include "''' + str(H7 / 'imcalib/bsp/output_capture.h') + r'''"
uint16_t Output_Capture_Packet(uint8_t *out) { memset(out,0,42); return 42; }
void Output_Capture_Commit(void) {}
''' if capture else ''
    (stub / "usbd_cdc_if.h").write_text('''
#include <stdint.h>
#define USBD_OK 0u
#define USBD_FAIL 2u
uint8_t CDC_Transmit_Ready_HS(void);
uint8_t CDC_Transmit_HS(uint8_t *data, uint16_t size);
''')
    source = directory / "usb-sender.c"
    source.write_text(r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
''' + capture_stub + r'''
#include "''' + str(H7 / "imcalib/bsp/Vofa_send.c") + r'''"
static uint8_t configured, busy, fail_submit;
static uint8_t *pending;
static uint16_t pending_size;
static unsigned submissions;
uint8_t CDC_Transmit_Ready_HS(void) { return configured && !busy; }
uint8_t CDC_Transmit_HS(uint8_t *data, uint16_t size) {
    submissions++;
    if (fail_submit) { return USBD_FAIL; }
    assert(configured && !busy);
    pending = data; pending_size = size; busy = 1;
    return USBD_OK;
}
int main(int argc, char **argv) {
    float values[VOFA_MAX_CH], changed[VOFA_MAX_CH];
    uint8_t original[VOFA_MAX_CH * sizeof(float) + 4];
    FILE *input = fopen(argv[1], "rb"); assert(input);
    assert(fread(original, 1, sizeof(original), input) == sizeof(original)); fclose(input);
    memcpy(values, original, sizeof(values));
    memcpy(changed, values, sizeof(changed)); changed[10] += 1.f;
    assert(!Vofa_Send(values, VOFA_MAX_CH) && submissions == 0); /* 未连接。 */
    configured = 1;
    assert(!Vofa_Send(values, 0) && !Vofa_Send(values, VOFA_MAX_CH + 1) && submissions == 0);
    assert(Vofa_Send(values, VOFA_MAX_CH) && pending_size == sizeof(original) && submissions == 1);
    assert(memcmp(pending, original, sizeof(original)) == 0); /* 实际发送器保持CRC与帧尾。 */
    assert(!Vofa_Send(changed, VOFA_MAX_CH) && memcmp(pending, original, sizeof(original)) == 0 && submissions == 1);
    configured = 0;
    assert(!Vofa_Send(changed, VOFA_MAX_CH) && memcmp(pending, original, sizeof(original)) == 0);
    configured = 1; busy = 0; fail_submit = 1;
    assert(!Vofa_Send(values, VOFA_MAX_CH) && submissions == 2);
    fail_submit = 0;
    assert(Vofa_Send(values, VOFA_MAX_CH) && submissions == 3 && pending_size == sizeof(original));
    assert(memcmp(pending, original, sizeof(original)) == 0); /* 重连/失败后可继续提交。 */
    assert(fwrite(pending, 1, pending_size, stdout) == pending_size);
    return 0;
}
'''.replace('pending_size == sizeof(original)', 'pending_size == sizeof(original)' + (' + 42u' if capture else ''))
       .replace('fwrite(pending, 1, pending_size, stdout) == pending_size', 'fwrite(pending, 1, sizeof(original), stdout) == sizeof(original)'))
    executable, frame = directory / "usb-sender", directory / "usb-input.bin"
    frame.write_bytes(raw)
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-I", str(stub),
                    str(source), "-o", str(executable)], check=True)
    sent = subprocess.check_output([str(executable), str(frame)])
    assert sent == raw and np.array_equal(mirror.decode_frame(sent), mirror.decode_frame(raw))
    return {"actual_c_sender": True, "disconnected_busy_buffer_preserved": True,
            "submit_failure_reconnect": True, "wire_bytes_and_crc_unchanged": True,
            "scope": "actual sender with CDC stubs; USB enumeration/completion need hardware"}


def check_policy_producer(directory):
    """Exercise the actual publication branch while the SDK mutates shared state."""
    stub = directory / "policy-stub"
    stub.mkdir()
    (stub / "policy_stub.h").write_text(r'''
#ifndef POLICY_STUB_H
#define POLICY_STUB_H
#include <stdint.h>
#define RL_ACTION_SIZE 6u
#define RL_OBS_SIZE 39u
#define CTRL_STRATEGY_RL 0u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
typedef struct { float seed; } rl_observation_sample_t;
typedef struct { float obs[39],last_action[6]; rl_observation_sample_t sample; uint8_t valid,sample_valid; } rl_observation_state_t;
typedef struct { uint8_t phase; } rl_recovery_state_t;
typedef struct { uint32_t run_us; } rl_policy_t;
typedef struct { float vel,yaw,len; uint8_t online; struct { uint8_t strategy; } mode; } rc_command_t;
typedef struct { uint8_t configured; } machine_rl_cfg_t;
typedef struct { machine_rl_cfg_t rl; } machine_config_t;
typedef struct { rl_observation_state_t observation; rl_policy_t policy; rl_recovery_state_t recovery; } rl_control_state_t;
typedef struct { float a[6]; uint32_t reset_count,last_ok_tick; uint8_t updated,rl_ready; } action_state_t;
typedef struct { float obs[39],action[6],command[3]; uint32_t sequence,sample_tick_ms,published_tick_ms,run_us; uint8_t valid; } mirror_policy_snapshot_t;
extern machine_config_t *machine;
extern rl_control_state_t rl_control;
extern rc_command_t rc_command;
extern action_state_t action_state;
extern mirror_policy_snapshot_t mirror_policy_snapshot;
extern uint32_t rl_observation_sample_tick_ms;
uint32_t HAL_GetTick(void);
uint64_t Mono_Ns_Get(void);
uint8_t RL_Policy_Init(rl_policy_t *);
uint8_t RL_Policy_Run(rl_policy_t *,const rl_observation_state_t *,float *);
uint8_t RL_Observation_Read_Sample(const rl_observation_state_t *,rl_observation_sample_t *);
void RL_Observation_Reset(rl_observation_state_t *);
uint8_t RL_Observation_Build(rl_observation_state_t *,const machine_rl_cfg_t *,const rl_observation_sample_t *,const rc_command_t *,const rl_recovery_state_t *,uint8_t);
void RL_Observation_Set_Last_Action(rl_observation_state_t *,const float *);
void Action_State_Clear(void);
#endif
''')
    for name in ("robot_control.h", "robot_state.h", "rl_observation.h", "rl_policy.h", "mono_ns.h"):
        (stub / name).write_text('#include "policy_stub.h"\n')
    source = directory / "policy-producer.c"
    source.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "policy_stub.h"
#include "''' + str(H7 / "imcalib/task/task_policy.c") + r'''"
static machine_config_t configuration={{1u}};
machine_config_t *machine=&configuration;
rl_control_state_t rl_control;
rc_command_t rc_command;
action_state_t action_state;
mirror_policy_snapshot_t mirror_policy_snapshot;
uint32_t rl_observation_sample_tick_ms;
static unsigned mode, initialized, sdk_calls, clears;
uint32_t HAL_GetTick(void) { return 7000u; }
uint64_t Mono_Ns_Get(void) { return 123456000u; }
uint8_t RL_Policy_Init(rl_policy_t *p) { initialized++; return 1; }
uint8_t RL_Observation_Read_Sample(const rl_observation_state_t *s,rl_observation_sample_t *out) { *out=s->sample; return s->sample_valid; }
void RL_Observation_Reset(rl_observation_state_t *s) { memset(s,0,sizeof(*s)); }
uint8_t RL_Observation_Build(rl_observation_state_t *s,const machine_rl_cfg_t *c,const rl_observation_sample_t *sample,const rc_command_t *cmd,const rl_recovery_state_t *r,uint8_t valid) {
    for(unsigned i=0;i<39;i++) s->obs[i]=sample->seed+cmd->vel+s->last_action[0]+i;
    return valid;
}
void RL_Observation_Set_Last_Action(rl_observation_state_t *s,const float *a) { memcpy(s->last_action,a,sizeof(s->last_action)); }
void Action_State_Clear(void) { clears++; action_state.reset_count++; action_state.updated=action_state.rl_ready=0; memset(action_state.a,0,sizeof(action_state.a)); }
uint8_t RL_Policy_Run(rl_policy_t *p,const rl_observation_state_t *input,float *a) {
    sdk_calls++; p->run_us=321u;
    for(unsigned i=0;i<6;i++) a[i]=input->obs[i]*.01f+i;
    if(mode==1u) {
        /* Model higher-priority sampling while this inference is in progress. */
        for(unsigned i=0;i<39;i++) rl_control.observation.obs[i]=9000.f+i;
        rl_control.observation.sample.seed=8000.f;
        rl_observation_sample_tick_ms=999u;
        rc_command.vel=9.f; rc_command.yaw=8.f; rc_command.len=7.f;
    }
    if(mode==2u) {
        action_state.reset_count++;
        for(unsigned i=0;i<6;i++) action_state.a[i]=123.f+i;
        action_state.updated=action_state.rl_ready=0;
    }
    return 1;
}
static void reset_case(unsigned selected_mode) {
    mode=selected_mode;
    memset(&rl_control,0,sizeof(rl_control)); memset(&action_state,0,sizeof(action_state));
    memset(&rc_command,0,sizeof(rc_command)); memset(&mirror_policy_snapshot,0,sizeof(mirror_policy_snapshot));
    rc_command.online=1; rc_command.mode.strategy=CTRL_STRATEGY_RL;
    rc_command.vel=.2f; rc_command.yaw=-.3f; rc_command.len=.31f;
    rl_control.observation.sample.seed=10.f; rl_control.observation.sample_valid=1;
    rl_control.observation.last_action[0]=.4f;
    rl_observation_sample_tick_ms=117u; action_state.reset_count=4u;
    mirror_policy_snapshot.sequence=20u; mirror_policy_snapshot.valid=1u;
}
int main(void) {
    ctrl_task_init(); assert(initialized==1u);
    for(unsigned selected_mode=0;selected_mode<2;selected_mode++) {
        reset_case(selected_mode); ctrl_task_body();
        assert(mirror_policy_snapshot.valid && mirror_policy_snapshot.sequence==21u);
        assert(mirror_policy_snapshot.sample_tick_ms==117u && mirror_policy_snapshot.published_tick_ms==123u);
        assert(mirror_policy_snapshot.run_us==321u && action_state.last_ok_tick==7000u);
        assert(action_state.updated && action_state.rl_ready && action_state.reset_count==4u);
        assert(fabsf(mirror_policy_snapshot.command[0]-.2f)<1e-6f);
        assert(fabsf(mirror_policy_snapshot.command[1]+.3f)<1e-6f);
        assert(fabsf(mirror_policy_snapshot.command[2]-.31f)<1e-6f);
        for(unsigned i=0;i<39;i++) assert(fabsf(mirror_policy_snapshot.obs[i]-(10.6f+i))<1e-5f);
        for(unsigned i=0;i<6;i++) {
            float expected=(10.6f+i)*.01f+i;
            assert(fabsf(mirror_policy_snapshot.action[i]-expected)<1e-6f);
            assert(action_state.a[i]==mirror_policy_snapshot.action[i]);
        }
    }
    reset_case(2u);
    mirror_policy_snapshot_t before=mirror_policy_snapshot;
    ctrl_task_body();
    assert(memcmp(&before,&mirror_policy_snapshot,sizeof(before))==0);
    assert(action_state.reset_count==5u && !action_state.updated && !action_state.rl_ready);
    for(unsigned i=0;i<6;i++) assert(action_state.a[i]==123.f+i);
    assert(sdk_calls==3u && clears==0u);
    puts("actual policy producer pairing/reset/timing checks passed");
    return 0;
}
''')
    executable = directory / "policy-producer"
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-I", str(stub),
                    str(source), "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    return {"actual_task_policy_source": True,
            "successful_obs_command_action_pair": True,
            "sdk_shared_state_mutation_keeps_local_pair": True,
            "reset_during_sdk_rejects_old_action_and_diagnostics": True,
            "diagnostic_mono_clock_distinct_from_control_hal_clock": True,
            "scope": "actual publication control flow with SDK/observation/RTOS stubs; no network numeric or hardware validation"}


def check_output_producer(directory):
    """Compile the two actual dispatch functions with driver submission stubs."""
    actual_path = H7 / "imcalib/task/task_actuation.c"
    actual = actual_path.read_text()

    def extract(name):
        declaration = re.search(r"^static void " + name + r"\([^)]*\)\s*\{", actual, re.MULTILINE)
        assert declaration, f"actual {name} definition not found"
        depth = 1
        for end in range(declaration.end(), len(actual)):
            depth += (actual[end] == "{") - (actual[end] == "}")
            if not depth:
                return actual[declaration.start():end + 1] + "\n"
        raise AssertionError(f"unterminated actual {name}")

    functions = extract("Mirror_Output_Publish") + extract("output_dispatch")
    capture = 'Output_Capture_Push' in functions
    capture_stub = r'''
#include "''' + str(H7 / 'imcalib/bsp/output_capture.h') + r'''"
#define MACHINE_DEFAULT 0u
static uint64_t output_dispatch_started_ns;
static void Output_Capture_Push_Check(const output_capture_record_t *record) {
    assert(record->kind==OUTPUT_CAPTURE_CONTROL && record->begin_ns==234567000u);
    assert(memcmp(record->torque_nm,driver_request,sizeof(record->torque_nm))==0);
    assert(record->controller_mode==0);
}
#define Output_Capture_Push Output_Capture_Push_Check
''' if capture else ''
    bench = (H7 / "imcalib/Algorithm/wheel_bench.c").exists()
    bench_stubs = r'''
#define CTRL_STRATEGY_WHEEL_BENCH 5u
#define WHEEL_BENCH_RUN 3u
#define WHEEL_BENCH_GATED 1u
#define WHEEL_BENCH_SEND 5u
static struct { uint8_t reserved,phase,reason; } wheel_bench;
static struct { struct { uint8_t strategy,engage; } mode; } rc_command;
static uint8_t Wheel_Bench_Output_Valid(uint32_t t) { (void)t; return 1; }
static uint32_t HAL_GetTick(void) { return 234u; }
static uint8_t Robot_Control_Enable_Allowed(void) { return 1; }
''' if bench else ""
    source = directory / "output-producer.c"
    source.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "''' + str(H7 / "imcalib/Algorithm/torque_output.h") + r'''"
#define HAL_OK 0u
#define FAULT_NONE 0u
#define CTRL_STRATEGY_RL 0u
#define CTRL_STRATEGY_COMPENSATION 4u
#define RL_RECOVERY_NORMAL 3u
#define RL_RECOVERY_RECOVER 2u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
typedef struct { uint8_t bench; float marker; } compensation_debug_t;
typedef struct { float target[6],action[6],pos[4],vel[6],torque[6]; uint32_t sequence,tick_ms,policy_sequence; uint8_t flags,valid; } mirror_output_snapshot_t;
static mirror_output_snapshot_t mirror_output_snapshot;
static compensation_debug_t pending_compensation,compensation_debug;
static struct { uint8_t drive,wheel_online; uint64_t wheel_rx_ns[2]; float wheel_vel[2]; } control_frame;
static struct { struct { uint8_t phase; } recovery; } rl_control;
static uint8_t ctrl_strategy,torque_output_enabled,output_debug_dm_sent,output_debug_dji_sent;
static uint32_t ctrl_fault,dm_status,dji_status,before_sequence;
static float rl_output_dm_cmd_nm[4],rl_output_wheel_cmd_nm[2],driver_request[6];
static unsigned nonzero_dm,nonzero_dji,zero_dm,zero_dji;
static uint64_t Mono_Ns_Get(void) { return 234567000u; }
static uint8_t Dm_Send_Torque(const float *values) {
    assert(mirror_output_snapshot.sequence==before_sequence);
    memcpy(driver_request,values,4*sizeof(float)); nonzero_dm++; return dm_status;
}
static uint8_t Dji_Send_Wheel_Torque(float left,float right) {
    assert(mirror_output_snapshot.sequence==before_sequence);
    driver_request[4]=left; driver_request[5]=right; nonzero_dji++; return dji_status;
}
static uint8_t Dm_Send_Zero(void) {
    assert(mirror_output_snapshot.sequence==before_sequence);
    memset(driver_request,0,4*sizeof(float)); zero_dm++; return dm_status;
}
static uint8_t Dji_All_Stop(void) {
    assert(mirror_output_snapshot.sequence==before_sequence);
    driver_request[4]=driver_request[5]=0; zero_dji++; return dji_status;
}
''' + bench_stubs + capture_stub + functions + r'''
int main(void) {
    mirror_output_snapshot.sequence=100;
    for(unsigned scenario=0;scenario<9;scenario++) {
        torque_output_t input={0}; mirror_output_snapshot_t diagnostic={0};
        control_frame.drive=torque_output_enabled=1; ctrl_fault=0;
        ctrl_strategy=CTRL_STRATEGY_RL; rl_control.recovery.phase=RL_RECOVERY_NORMAL;
        dm_status=dji_status=HAL_OK;
        output_debug_dm_sent=output_debug_dji_sent=1; /* Old results must not leak. */
        input.valid=1;
        for(unsigned i=0;i<4;i++) input.dm[i]=i+.25f;
        for(unsigned i=0;i<2;i++) input.dji[i]=-1.f-i;
        for(unsigned i=0;i<6;i++) { diagnostic.target[i]=10.f+i; diagnostic.action[i]=-.1f*i; diagnostic.vel[i]=i+.5f; }
        for(unsigned i=0;i<4;i++) diagnostic.pos[i]=i+.125f;
        diagnostic.flags=1; diagnostic.policy_sequence=55;
        if(scenario==0) dji_status=1;
        if(scenario==1) dm_status=1;
        if(scenario==2) torque_output_enabled=0;
        if(scenario==3) ctrl_fault=4;
        if(scenario==4) input.dm[1]=NAN;
        if(scenario==5) control_frame.drive=0;
        if(scenario==6) input.valid=0;
        if(scenario==7) rl_control.recovery.phase=1;
        if(scenario==8) input.dji[1]=INFINITY;
        torque_output_t before=input;
        before_sequence=mirror_output_snapshot.sequence;
        pending_compensation.marker=42.f;
        output_dispatch(&input,&diagnostic);
        unsigned gated=scenario>=2;
        assert(input.valid==before.valid && memcmp(input.dm,before.dm,sizeof(input.dm))==0
               && memcmp(input.dji,before.dji,sizeof(input.dji))==0);
        assert(mirror_output_snapshot.valid && mirror_output_snapshot.sequence==before_sequence+1);
        assert(mirror_output_snapshot.tick_ms==234u && mirror_output_snapshot.policy_sequence==55u);
        unsigned flags=1u|(gated?2u:0u)|(dm_status==HAL_OK?4u:0u)|(dji_status==HAL_OK?8u:0u)
            |(torque_output_enabled?16u:0u)|(control_frame.drive?32u:0u);
        assert(mirror_output_snapshot.flags==flags);
        for(unsigned i=0;i<6;i++) {
            float expected=gated?0.f:(i<4?input.dm[i]:input.dji[i-4]);
            assert(mirror_output_snapshot.torque[i]==expected && driver_request[i]==expected);
            assert(mirror_output_snapshot.target[i]==10.f+i);
            assert(mirror_output_snapshot.action[i]==-.1f*i && mirror_output_snapshot.vel[i]==i+.5f);
            assert((i<4?rl_output_dm_cmd_nm[i]:rl_output_wheel_cmd_nm[i-4])==expected);
        }
        for(unsigned i=0;i<4;i++) assert(mirror_output_snapshot.pos[i]==i+.125f);
        assert(compensation_debug.marker==(gated?0.f:42.f));
    }
    assert(nonzero_dm==2 && nonzero_dji==2 && zero_dm==7 && zero_dji==7);
    puts("actual output dispatch/snapshot/gate checks passed");
    return 0;
}
''')
    executable = directory / "output-producer"
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-I", str(H7 / "imcalib/common"),
                    str(source), "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    return {"actual_dispatch_and_publish_function_sha256": hashlib.sha256(functions.encode()).hexdigest(),
            "task_actuation_source_sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest(),
            "different_hal_results_published_after_driver_callbacks": True,
            "gate_fault_invalid_nan_inf_drive_recovery_zeroed": True,
            "published_sequence_tick_inputs_requests_match_same_dispatch": True,
            "scope": "actual extracted functions with driver/RTOS stubs; submitted requests, not actual motor torque or acknowledgement"}


def check_wheel_bench(directory):
    """Run the actual bench state machine, RC decoder and final output gate with IO stubs."""
    root = H7 / "imcalib"
    module = root / "Algorithm/wheel_bench.c"
    if not module.exists():
        return {"present": False}
    stub = directory / "bench-stubs"
    stub.mkdir()
    (stub / "machine_config.h").write_text('''
#include <stdint.h>
typedef struct { struct { float vel_max,yaw_max,len_rate,len_init; } rc;
    float leg_len_min,leg_len_max; unsigned lqr_configured;
    struct { unsigned configured; } rl; void *spring,*gravity; } bench_machine_t;
extern const bench_machine_t *machine;
''')
    (stub / "dr16.h").write_text('''
#include "rc_types.h"
static inline int16_t DR16_Deadline(int16_t x,uint16_t d) { return x > (int)d || x < -(int)d ? x : 0; }
''')
    (stub / "gas_spring.h").write_text('#define GAS_SPRING_COMP_ENABLE 1\n')
    (stub / "lqr_gain_table.h").write_text('''
typedef struct { float len_min,len_max; } bench_gain_t;
static inline const bench_gain_t *LQR_Gain_Info(void) { static const bench_gain_t g={.1f,.4f}; return &g; }
''')
    actual = (root / "task/task_actuation.c").read_text()
    def extract(name, text=actual):
        start = re.search(r"^(?:static )?(?:void|uint8_t) " + name + r"\([^)]*\)\s*\{", text, re.M)
        assert start
        depth = 1
        for end in range(start.end(), len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if not depth:
                return text[start.start():end + 1] + "\n"
        raise AssertionError(name)
    functions = "".join(extract(n) for n in ("Robot_Control_Enable_Allowed", "Wheel_Bench_Permit", "Mirror_Output_Publish", "output_dispatch"))
    functions += extract("Robot_Enable_Update", (root / "task/task_comm.c").read_text())
    selection = re.search(r"^    strategy = Wheel_Bench_Output_Valid.*;$", actual, re.M).group(0)
    functions += "static ctrl_strategy_t actual_strategy_selection(void) { ctrl_strategy_t strategy; " + selection.strip() + " return strategy; }\n"
    capture_stub = r'''
#include "''' + str(H7 / 'imcalib/bsp/output_capture.h') + r'''"
#define MACHINE_DEFAULT 0u
static uint64_t output_dispatch_started_ns;
void Output_Capture_Push(const output_capture_record_t *record) {
    assert(record->kind==OUTPUT_CAPTURE_CONTROL);
    assert(memcmp(record->torque_nm,driver_request,sizeof(record->torque_nm))==0);
}
''' if 'Output_Capture_Push' in functions else ''
    source = directory / "wheel-bench-check.c"
    source.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "wheel_bench.h"
#include "rc_command.h"
#include "machine_config.h"
static const bench_machine_t config={.rc={1,1,.1f,.27f},.leg_len_min=.1f,.leg_len_max=.4f,.lqr_configured=1,.rl={1},.spring=(void*)1,.gravity=(void*)1};
const bench_machine_t *machine=&config;
#define HAL_OK 0u
#define FAULT_NONE 0u
#define RL_RECOVERY_NORMAL 3u
#define RL_RECOVERY_RECOVER 2u
#define RL_RECOVERY_FAILED 4u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
typedef struct { uint8_t bench; } compensation_debug_t;
typedef struct { float target[6],action[6],pos[4],vel[6],torque[6]; uint32_t sequence,tick_ms,policy_sequence; uint8_t flags,valid; } mirror_output_snapshot_t;
static mirror_output_snapshot_t mirror_output_snapshot;
static compensation_debug_t pending_compensation,compensation_debug;
static struct { uint8_t drive,wheel_online; uint64_t wheel_rx_ns[2]; float wheel_vel[2]; rc_command_t rc; } control_frame;
static struct { struct { uint8_t phase; } recovery; } rl_control;
static struct { uint8_t enabled,recovery_enabled; } standup_control;
static struct { uint8_t fallen,motor_enabled; } robot_state;
static struct { uint8_t valid; struct { uint8_t online[2]; uint64_t parsed_rx_ns[2]; float vel_rad_s[2]; } dji; } motor_state;
static rc_command_t rc_command;
static uint8_t ctrl_strategy,torque_output_enabled=1,output_debug_dm_sent,output_debug_dji_sent;
static uint32_t ctrl_fault,now_ms,dm_status,dji_status,stop_status;
static unsigned remote_online=1,dm_enabled,nonzero_dm,nonzero_dji,zero_dji;
static float rl_output_dm_cmd_nm[4],rl_output_wheel_cmd_nm[2],driver_request[6];
static uint64_t Mono_Ns_Get(void) { return (uint64_t)now_ms*1000000u; }
static uint32_t HAL_GetTick(void) { return now_ms; }
static unsigned DR16_Online(void) { return remote_online; }
static unsigned Dm_Is_Enabled(unsigned i) { (void)i; return dm_enabled; }
static uint8_t Dm_Send_Torque(const float *v) { memcpy(driver_request,v,16); nonzero_dm++; return dm_status; }
static uint8_t Dm_Send_Zero(void) { memset(driver_request,0,16); return dm_status; }
static uint8_t Dji_Send_Wheel_Torque(float l,float r) { driver_request[4]=l; driver_request[5]=r; nonzero_dji++; return dji_status; }
static uint8_t Dji_All_Stop(void) { driver_request[4]=driver_request[5]=0; zero_dji++; return stop_status; }
static unsigned disabled_dm;
static uint8_t Dm_All_Enable(void) { return 0; }
static uint8_t Dm_All_Disable(void) { disabled_dm++; return 0; }
static void Dm_Enable_Watchdog(void) { }
static void Dm_Disable_Watchdog(void) { }
''' + capture_stub + functions + r'''
static torque_output_t step(unsigned left,unsigned right,unsigned permit,float l,float r) {
    dr16_t raw={0}; torque_output_t t; float speed[2]={l,r};
    raw.online=remote_online; raw.s1=left; raw.s2=right;
    Rc_Command_Update(&rc_command,&raw,.001f);
    Wheel_Bench_Update(&rc_command,permit,speed,now_ms++,&t);
    for(unsigned i=0;i<4;i++) assert(t.dm[i]==0);
    assert(!(t.dji[0]!=0 && t.dji[1]!=0));
    assert(fabsf(t.dji[0])<=3.0f && fabsf(t.dji[1])<=3.0f);
    return t;
}
static void reset(void) {
    memset((void*)&wheel_bench,0,sizeof(wheel_bench));
    remote_online=1; now_ms=1000;
}
static void arm(void) {
    for(unsigned i=0;i<=WHEEL_BENCH_ARM_MS;i++) assert(!step(DR16_SW_DOWN,DR16_SW_UP,1,0,0).valid);
    assert(wheel_bench.phase==WHEEL_BENCH_READY && wheel_bench.reserved);
}
static void dispatch(torque_output_t *t) {
    mirror_output_snapshot_t d={0};
    ctrl_strategy=CTRL_STRATEGY_WHEEL_BENCH; control_frame.drive=1;
    motor_state.valid=1;
    for(unsigned i=0;i<2;i++) { motor_state.dji.online[i]=1; motor_state.dji.parsed_rx_ns[i]=Mono_Ns_Get(); motor_state.dji.vel_rad_s[i]=0; }
    output_dispatch(t,&d);
}
int main(void) {
    torque_output_t t;
    reset();
    /* All switch pairs: RL restored, LQR/compensation and bench priority unchanged. */
    for(unsigned l=1;l<=3;l++) for(unsigned r=1;r<=3;r++) {
        step(l,r,0,0,0); wheel_bench.reserved=0; wheel_bench.phase=0;
        if(l==DR16_SW_UP && r==DR16_SW_UP) assert(rc_command.mode.strategy==CTRL_STRATEGY_WHEEL_BENCH);
        if(l==DR16_SW_UP && r==DR16_SW_MID) assert(rc_command.mode.strategy==CTRL_STRATEGY_RL && rc_command.mode.engage && rc_command.mode.rc_enable);
        if(l==DR16_SW_UP && r==DR16_SW_DOWN) assert(rc_command.mode.strategy==CTRL_STRATEGY_RL && !rc_command.mode.engage);
        if(l==DR16_SW_MID && r==DR16_SW_MID) assert(rc_command.mode.strategy==CTRL_STRATEGY_LQR && rc_command.mode.engage && rc_command.mode.rc_enable);
        if(l==DR16_SW_MID && r==DR16_SW_UP) assert(rc_command.mode.strategy==CTRL_STRATEGY_COMPENSATION);
    }
    reset();
    for(unsigned i=0;i<1500;i++) assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid); /* Boot never fires. */
    for(unsigned pulse=0;pulse<4;pulse++) {
        arm();
        step(DR16_SW_MID,DR16_SW_UP,1,0,0);
        assert(!rc_command.mode.rc_enable && !Robot_Control_Enable_Allowed());
        t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
        assert(t.valid && t.dji[pulse/2]==(pulse%2?-3.0f:3.0f));
        assert(!Robot_Control_Enable_Allowed());
        for(unsigned i=1;i<300;i++) assert(step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid);
        for(unsigned i=0;i<500;i++) assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid);
        assert(wheel_bench.phase==WHEEL_BENCH_DONE);
    }
    /* Running pulses ignore permit/speed/NaN/interval/offline/switch/release changes. */
    for(unsigned scenario=0;scenario<9;scenario++) {
        reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0); assert(t.valid);
        if(scenario==0) t=step(DR16_SW_UP,DR16_SW_UP,0,0,0);
        if(scenario==1) t=step(DR16_SW_UP,DR16_SW_UP,1,300,0);
        if(scenario==2) t=step(DR16_SW_UP,DR16_SW_UP,1,0,-300);
        if(scenario==3) t=step(DR16_SW_UP,DR16_SW_UP,1,NAN,INFINITY);
        if(scenario==4) { now_ms+=6; t=step(DR16_SW_UP,DR16_SW_UP,1,0,0); }
        if(scenario==5) { remote_online=0; t=step(DR16_SW_UP,DR16_SW_UP,1,0,0); }
        if(scenario==6) t=step(DR16_SW_MID,DR16_SW_UP,1,0,0);
        if(scenario==7) t=step(DR16_SW_DOWN,DR16_SW_DOWN,0,0,0);
        if(scenario==8) { now_ms=wheel_bench.start_ms+299; t=step(DR16_SW_UP,DR16_SW_UP,0,300,0); }
        assert(t.valid && wheel_bench.phase==WHEEL_BENCH_RUN && wheel_bench.reserved);
        assert(t.dji[0]==3.0f && t.dji[1]==0);
        now_ms=wheel_bench.start_ms+300;
        assert(!step(DR16_SW_UP,DR16_SW_UP,0,300,0).valid);
        assert(wheel_bench.phase==WHEEL_BENCH_DONE);
        for(unsigned i=0;i<500;i++) assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid);
    }
    reset(); now_ms=UINT32_MAX-350; arm();
    assert(step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid);
    for(unsigned i=1;i<300;i++) assert(step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid);
    assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid); /* uint32 wrap */
    reset(); arm(); now_ms+=6;
    assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid); /* Stalled READY cannot fire. */
    reset(); arm(); step(DR16_SW_DOWN,DR16_SW_DOWN,1,0,0);
    assert(!step(DR16_SW_UP,DR16_SW_UP,1,0,0).valid); /* Even a short exit cancels READY. */
    reset();
    for(unsigned i=0;i<400;i++) assert(!step(DR16_SW_DOWN,DR16_SW_UP,1,.3f,0).valid);
    assert(wheel_bench.phase!=WHEEL_BENCH_READY);
    arm(); step(DR16_SW_MID,DR16_SW_MID,1,0,0);
    assert(!rc_command.mode.rc_enable); /* Reserved traversal is inhibited. */
    for(unsigned i=0;i<=300;i++) step(DR16_SW_DOWN,DR16_SW_DOWN,1,0,0);
    assert(!wheel_bench.reserved);
    step(DR16_SW_MID,DR16_SW_MID,1,0,0);
    assert(rc_command.mode.rc_enable && rc_command.mode.engage); /* Normal LQR restored. */
    step(DR16_SW_UP,DR16_SW_MID,1,0,0);
    assert(rc_command.mode.strategy==CTRL_STRATEGY_RL && rc_command.mode.rc_enable && rc_command.mode.engage);

    reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
    dispatch(&t);
    assert(!nonzero_dm && nonzero_dji==1 && driver_request[4]==3.0f);
    assert(!(mirror_output_snapshot.flags&2) && mirror_output_snapshot.torque[4]==3.0f);
    motor_state.dji.vel_rad_s[1]=29.99f; assert(Wheel_Bench_Permit());
    motor_state.dji.vel_rad_s[1]=30.f; assert(!Wheel_Bench_Permit());
    for(unsigned scenario=0;scenario<13;scenario++) {
        reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
        dm_status=dji_status=0; ctrl_fault=0; dm_enabled=0; robot_state.fallen=robot_state.motor_enabled=0;
        dispatch(&t); unsigned before=nonzero_dji;
        if(scenario==0) motor_state.dji.parsed_rx_ns[0]-=10000000u;
        if(scenario==1) motor_state.dji.vel_rad_s[1]=300;
        if(scenario==2) dm_enabled=1;
        if(scenario==3) remote_online=0;
        if(scenario==4) ctrl_fault=1;
        if(scenario==5) rc_command.mode.strategy=CTRL_STRATEGY_LQR;
        if(scenario==6) now_ms=wheel_bench.start_ms+300;
        if(scenario==7) torque_output_enabled=0;
        if(scenario==8) robot_state.motor_enabled=1;
        if(scenario==9) robot_state.fallen=1;
        if(scenario==10) motor_state.valid=0;
        if(scenario==11) motor_state.dji.vel_rad_s[1]=NAN;
        if(scenario==12) control_frame.drive=0;
        mirror_output_snapshot_t d={0}; output_dispatch(&t,&d);
        if(scenario==6) assert(nonzero_dji==before && driver_request[4]==0 && (d.flags&2));
        else assert(nonzero_dji==before+1 && driver_request[4]==3.0f && !(d.flags&2));
        torque_output_enabled=1;
    }
    ctrl_fault=dm_enabled=0; robot_state.fallen=robot_state.motor_enabled=0;
    for(unsigned scenario=0;scenario<2;scenario++) {
        reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
        dm_status=scenario; dji_status=!scenario; stop_status=scenario;
        unsigned before=zero_dji; dispatch(&t);
        assert(wheel_bench.phase==WHEEL_BENCH_RUN && zero_dji==before && driver_request[4]==3.0f);
        assert(!(mirror_output_snapshot.flags&2));
        assert(!!(mirror_output_snapshot.flags&8)==!dji_status);
    }
    dm_status=dji_status=stop_status=0;
    /* Lost RC cannot route the latched pulse into another controller. */
    reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
    memset(&control_frame.rc,0,sizeof(control_frame.rc));
    assert(actual_strategy_selection()==CTRL_STRATEGY_WHEEL_BENCH);
    now_ms=wheel_bench.start_ms+299;
    assert(actual_strategy_selection()==CTRL_STRATEGY_WHEEL_BENCH);
    now_ms=wheel_bench.start_ms+300;
    assert(actual_strategy_selection()==CTRL_STRATEGY_DISABLE);
    /* Communication-task disable does not inject a wheel zero before expiry. */
    reset(); arm(); t=step(DR16_SW_UP,DR16_SW_UP,1,0,0);
    robot_state.motor_enabled=1;
    unsigned stops=zero_dji,disabled=disabled_dm;
    Robot_Enable_Update();
    assert(!robot_state.motor_enabled && zero_dji==stops && disabled_dm==disabled+1);
    now_ms=wheel_bench.start_ms+300;robot_state.motor_enabled=1;
    Robot_Enable_Update();assert(zero_dji==stops+1);
    assert(!nonzero_dm);
    puts("wheel bench: pretrigger guards, latched 300ms despite abort inputs, expiry, wrap, dispatch, selection and comm stop arbitration passed");
}
''')
    executable = directory / "wheel-bench-check"
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-I", str(stub), "-I", str(root / "common"), "-I", str(root / "Algorithm"),
                    "-I", str(root / "task/inc"), str(source), str(module), str(root / "task/rc_command.c"),
                    "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    return {"present": True, "source_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "rc_decoder_sha256": hashlib.sha256((root / "task/rc_command.c").read_bytes()).hexdigest(),
            "actual_gate_functions_sha256": hashlib.sha256(functions.encode()).hexdigest(),
            "switch_map_lqr_preserved_rl_engage_restored_bench_priority": True,
            "pretrigger_guards_and_300ms_only_run_termination_checked": True,
            "run_ignores_speed_fault_stale_switch_gap_send_failure_checked": True,
            "selection_and_communication_stop_arbitration_checked": True,
            "scope": "actual C state/RC/permit/dispatch with driver and RTOS substitutes; no board movement"}


def encode(frame):
    payload = np.asarray(frame[:-1], dtype="<f4").tobytes()
    return payload + struct.pack("<f", binascii.crc_hqx(payload, 0xffff)) + mirror.TAIL


def check_stream(raw):
    frame = mirror.decode_frame(raw)
    damaged = bytearray(raw)
    damaged[40:44] = struct.pack("<f", .11)  # finite/plausible, but a bad CRC
    false_tail = frame.copy()
    false_tail[10] = np.inf
    false_tail[29] = int(false_tail[29]) & ~2
    old = frame.copy()
    old[3] = 0
    wrong_machine = frame.copy()
    wrong_machine[5] = 1
    stream = b"noise" + bytes(damaged) + encode(old) + encode(false_tail) + encode(wrong_machine) + raw
    parser = mirror.FrameParser()
    received = []
    for i in range(0, len(stream), 7):
        received += parser.feed(stream[i:i+7])
    assert len(received) == 2 and np.isinf(received[0][10])
    assert np.array_equal(received[1], frame)
    parser.feed(b"x" * 10000)
    assert len(parser.buffer) <= mirror.MAX_FRAME_BYTES - 1
    clock = mirror.FrameClock()
    def f(seq, tick):
        v = frame.copy()
        v[6:10] = (seq & 65535, seq >> 16, tick & 65535, tick >> 16)
        return v
    assert clock.accept(f(0xfffffffe, 0xfffffff0))
    assert clock.accept(f(0xffffffff, 0xfffffff9))
    assert clock.accept(f(1, 4)) and clock.missing == 1
    assert not clock.accept(f(1, 4)) and clock.duplicates == 1
    assert not clock.accept(f(0, 5)) and clock.out_of_order == 1
    assert clock.accept(f(1, 0)) and clock.restarts == 1
    clock = mirror.FrameClock()
    assert clock.accept(f(100000, 0x90000000))
    assert clock.accept(f(1, 10)) and clock.restarts == 1
    return {"damaged_and_legacy_rejected": True, "false_tail_resynchronized": True,
            "fragmentation_sequence_wrap_restart": True}


def check_v2_sessions(directory, raw):
    """Check both wire versions and preserve missing versus genuine zero data."""
    original = mirror.decode_frame(raw)
    legacy = np.zeros(48, dtype=np.float32)
    legacy[:47] = original[:47]
    legacy[4], legacy[6:10], legacy[32:47] = 1, [0, 0, 0, 0], 0
    if len(original) == 160:
        full = original.copy()  # The USB path exercises real C diagnostic bytes.
    else:
        full = np.zeros(160, dtype=np.float32)
        full[:47] = legacy[:47]
        full[4], full[47] = 2, 3
        full[48:54] = [0x2345, 1, 6, 0, 8, 0]
        full[54:57] = [.2, -.3, .31]
        full[57:96] = np.arange(100, 139)
        full[96:102] = -.1 * np.arange(6)
        full[102:110] = [250, 0x3456, 2, 9, 0, 0x2345, 1, 53]
        full[110:116] = np.arange(10, 16)
        full[116:122] = -.2 * np.arange(6)
        full[122:126], full[126:132] = .3 * np.arange(4), np.arange(4, 10)
        full[132:138], full[138:146] = .1 * np.arange(1, 7), np.arange(1, 9)
        full[146:159] = [0.5, 1, 1.5, 2, 13, -14, 0, 7, 3, 1, 9, 0, 0]
    full[6:10] = [1, 0, 10, 0]
    full[32:47] = 0
    missing = full.copy()
    missing[6:10], missing[47], missing[31], missing[138:144] = [2, 0, 20, 0], 0, 201, 201
    zero = full.copy()
    zero[6:10], zero[47], zero[109] = [3, 0, 30, 0], 2, 14
    zero[132:138], zero[138:146] = 0, -1
    policy_only = full.copy()
    policy_only[6:10], policy_only[47] = [4, 0, 40, 0], 1
    legacy_last = legacy.copy()
    legacy_last[6:10] = [5, 0, 50, 0]
    nonfinite = full.copy()
    nonfinite[6:10], nonfinite[57], nonfinite[96] = [6, 0, 60, 0], np.nan, np.inf
    expected = [legacy, full, missing, zero, policy_only, legacy_last, nonfinite]
    damaged_v1 = bytearray(encode(legacy))
    damaged_v1[40:44] = struct.pack("<f", .123)
    damaged_v2 = bytearray(encode(full))
    damaged_v2[57 * 4:58 * 4] = struct.pack("<f", 999.)
    stream = b"noise" + bytes(damaged_v1) + encode(legacy) + bytes(damaged_v2)
    stream += b"".join(encode(frame) for frame in expected[1:])
    parser = mirror.FrameParser()
    received = []
    for start in range(0, len(stream), 11):
        received.extend(parser.feed(stream[start:start + 11]))
    assert len(received) == 7 and parser.versions == {1: 2, 2: 5} and parser.last_version == 2
    for frame, wanted in zip(received, expected):
        assert np.array_equal(frame[:-1], wanted[:-1], equal_nan=True)
    assert parser.rejected >= 2
    # Retaining a longer v2 prefix must not recount the same damaged v1 tail.
    once = mirror.FrameParser()
    assert not once.feed(bytes(damaged_v1)) and once.rejected == 1
    assert not once.feed(b"tail-free-noise") and once.rejected == 1
    assert len(once.feed(encode(full)[:-2])) == 0
    assert len(once.feed(encode(full)[-2:])) == 1 and once.rejected == 1
    unsupported = mirror.FrameParser()
    unsupported.feed(encode(legacy))
    assert not mirror.wire_info(unsupported)["control_telemetry_supported"]
    assert "UNSUPPORTED" in mirror.wire_info(unsupported)["control_telemetry_notice"]
    for index, value in ((4, 1), (47, 4), (109, 64), (158, -.5)):
        bad = full.copy()
        bad[index] = value
        try:
            mirror.decode_frame(encode(bad))
        except ValueError:
            pass
        else:
            raise AssertionError(("bad v2 header/mask accepted", index, value))

    recorder = mirror.SessionRecorder(directory / "mixed-wire-session", machine_config=H7 / "imcalib/config/machine_config.c")
    recorder.record("connected", elapsed_ns=0, state="CONNECTED")
    chunks = [stream[start:start + 257] for start in range(0, len(stream), 257)]
    for i, chunk in enumerate(chunks, 1):
        recorder.record("data", elapsed_ns=i * 1_000_000, data_hex=chunk.hex())
    recorder.finish(elapsed_ns=(len(chunks) + 10) * 1_000_000)
    assert recorder.error is None
    mirror.load_session(recorder.directory)
    saved_bytes = b"".join(bytes.fromhex(e["data_hex"]) for e in mirror.iter_session_events(recorder.directory) if e["type"] == "data")
    assert saved_bytes == stream
    replay = mirror.ReplayReceiver(recorder.directory)
    replay.start()
    replay.join(timeout=2.)
    assert not replay.is_alive() and replay.error is None and replay.parser.versions == {1: 2, 2: 5}
    assert np.array_equal(replay.snapshot()[0][:-1], nonfinite[:-1], equal_nan=True)
    report = mirror.analyze_session(recorder.directory)
    with (recorder.directory / "analysis/trajectory.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 7
    extra_columns = list(mirror.telemetry_row(legacy))
    for row in (rows[0], rows[5]):
        assert all(row[name] == "" for name in extra_columns), "v1 missing telemetry became a zero measurement"
    paired = rows[1]
    assert paired["policy_pair_valid"] == paired["output_snapshot_valid"] == paired["rl_pd_valid"] == "1"
    assert int(paired["policy_sequence"]) == 0x12345 and int(paired["output_used_policy_sequence"]) == 0x12345
    assert int(paired["policy_sample_tick_ms"]) == 6 and int(paired["action_publish_tick_ms"]) == 8
    assert np.isclose(float(paired["command_leg_length_m"]), .31)
    assert int(paired['wheel_response_mode']) == full[158]
    profile = mirror.WHEEL_RESPONSE_CATALOG[str(int(full[158]))]
    assert float(paired['wheel_compensation_l_nm']) == profile['friction_nm'][0]
    assert float(paired['wheel_compensation_r_nm']) == profile['friction_nm'][1]
    assert recorder.metadata['wheel_response_catalog'] == mirror.WHEEL_RESPONSE_CATALOG
    for i in range(39):
        assert float(paired[f"obs_{i:02d}"]) == 100 + i
    action_order = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
    for i, name in enumerate(action_order):
        assert np.isclose(float(paired[f"action_{name}"]), -.1 * i)
        assert np.isclose(float(paired[f"pd_used_action_{name}"]), -.2 * i)
    for i, name in enumerate(mirror.MOTORS):
        unit = "rad" if i < 4 else "rad_s"
        assert float(paired[f"pd_target_{name}_{unit}"]) == 10 + i
        assert float(paired[f"pd_vel_{name}_rad_s"]) == 4 + i
        assert np.isclose(float(paired[f"gated_torque_{name}_firmware_nm"]), .1 * (i + 1))
        assert float(paired[f"motor_age_{name}_ms"]) == 1 + i
        if i < 4:
            assert np.isclose(float(paired[f"pd_pos_{name}_rad"]), .3 * i)
            assert np.isclose(float(paired[f"feedback_torque_{name}_firmware_nm"]), .5 * (i + 1))
    assert float(paired["wheel_current_lfwheel_raw"]) == 13 and float(paired["wheel_current_rfwheel_raw"]) == -14
    assert paired["imu_sample_age_ms"] == "7.0" and paired["rc_sample_age_ms"] == "8.0"
    for name in ("command_vx_m_s", "obs_00", "action_lf0", "pd_target_lf0_rad", "gated_torque_lf0_firmware_nm"):
        assert rows[2][name] == "", (name, rows[2][name])
    assert rows[2]["policy_pair_valid"] == rows[2]["output_snapshot_valid"] == "0"
    assert rows[2]["policy_sequence"] == paired["policy_sequence"]
    for name in ("feedback_torque_lf0_firmware_nm", "wheel_current_lfwheel_raw"):
        assert rows[2][name] == rows[3][name] == ""
    assert rows[2]["motor_age_lf0_ms"] == "201.0"
    assert rows[3]["motor_age_lf0_ms"] == rows[3]["imu_sample_age_ms"] == ""
    assert rows[3]["output_zero"] == rows[3]["output_dm_hal_success"] == rows[3]["output_dji_hal_success"] == "1"
    assert rows[3]["output_enabled"] == rows[3]["drive_permitted"] == rows[3]["rl_pd_valid"] == "0"
    assert rows[3]["pd_target_lf0_rad"] == "" and float(rows[3]["gated_torque_lf0_firmware_nm"]) == 0
    assert rows[4]["obs_00"] == "100.0" and rows[4]["gated_torque_lf0_firmware_nm"] == ""
    assert rows[6]["obs_00"] == rows[6]["action_lf0"] == "" and rows[6]["obs_01"] == "101.0"
    assert report["channels"]["obs_00"]["count"] == 2 and report["channels"]["obs_01"]["count"] == 3
    assert report["channels"]["gated_torque_lf0_firmware_nm"]["count"] == 3
    telemetry = report["control_telemetry"]
    assert telemetry["supported"] and telemetry["accepted_wire_versions"] == {1: 2, 2: 5}
    assert (telemetry["v1_unsupported_frames"], telemetry["policy_pair_frames"], telemetry["output_snapshot_frames"], telemetry["rl_pd_frames"]) == (2, 3, 3, 2)
    assert telemetry["unique_policy_sequences"] == telemetry["unique_output_sequences"] == 1
    assert telemetry["output_sample_counts"]["zero_output"] == 1
    assert report["pd_tracking_errors"]["motors"]["lf0"]["count"] == 2
    assert np.isclose(report["pd_tracking_errors"]["motors"]["lf0"]["mean"], 10.)
    for name in ("signals.png", "telemetry.png", "observations.png"):
        assert (recorder.directory / "analysis" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    return {"v1_v2_mixed_fragmented_crc_false_tail_resynchronization": True,
            "damaged_v1_candidate_not_recounted_by_future_noise": True,
            "real_c_extension_bytes_record_replay_csv": len(original) == 160,
            "v1_and_invalid_groups_missing_not_zero": True,
            "paired_commands_observations_actions_pd_inputs_torque_order": True,
            "zero_gate_is_real_zero_and_hal_submission_flags_separate": True,
            "unknown_and_stale_source_ages_feedback_not_zero": True,
            "repeated_snapshot_sequences_counted_once": True,
            "control_tracking_summary_and_pngs": True,
            "scope": "software capture/parse/export only; final torque is drive-entry request, not motor acknowledgement or measured torque"}


def frame_for_pose(pose, qpos):
    frame = np.zeros(48, dtype=np.float32)
    frame[22:26] = qpos[3:7]
    native = np.concatenate([qpos[q] for q in pose.drive_q])
    frame[10:14] = native / pose.sign[:4] + pose.zero
    frame[18:20] = qpos[pose.wheel_q] / pose.sign[4:]
    frame[29] = 127
    return frame


def check_geometry():
    pose = mirror.MirrorPose(machine_config=H7 / "imcalib/config/machine_config.c")
    assert (np.linalg.norm(pose.data.light_xdir, axis=1) > .9).all()
    fixed_translation = pose.data.qpos[:3].copy()
    poses = np.load(mirror.MODEL.with_name("huananhu_pose_bank.npz"), allow_pickle=False)
    joints = [("lf0", "lf1", "lfwheel", "left_spring1_joint", "left_spring2_joint", "lf00", "lf01", "lf02", "lf03"),
              ("rf0", "rf1", "rfwheel", "right_spring1_joint", "right_spring2_joint", "rf00", "rf01", "rf02", "rf03")]
    count, maximum, wheel_error, times, cold_error = 0, 0., 0., [], 0.
    for mode in ("normal", "positive", "negative"):
        for side, name in enumerate(("left", "right")):
            for index in np.linspace(0, 1023, 16, dtype=int):
                pose.data.qpos[:] = pose.model.qpos0
                addresses = [pose.model.joint(n).qposadr[0] for n in joints[side]]
                pose.data.qpos[addresses] = poses[f"{mode}_{name}_pos"][index]
                mujoco.mj_kinematics(pose.model, pose.data)
                expected_wheel = pose.data.body(mirror.WHEELS[side]).xpos.copy()
                measured = pose.data.qpos[pose.drive_q[side]].copy()
                expected_qpos = pose.data.qpos.copy()
                # First telemetry may arrive while folded: test starting from XML,
                # without supplying the hidden passive angles to the solver.
                pose.data.qpos[:] = pose.model.qpos0
                assert pose._solve_leg(side, measured), ("cold", mode, side, index)
                cold_error = max(cold_error, float(np.linalg.norm(pose.data.body(mirror.WHEELS[side]).xpos - expected_wheel)))
                pose.data.qpos[:] = expected_qpos
                pose.data.qpos[pose.passive_q[side]] += .01
                started = time.perf_counter()
                assert pose._solve_leg(side, measured), (mode, side, index, pose.residual_m)
                times.append(time.perf_counter() - started)
                assert np.array_equal(pose.data.qpos[pose.drive_q[side]], measured)
                maximum = max(maximum, pose.residual_m[side])
                wheel_error = max(wheel_error, float(np.linalg.norm(pose.data.body(mirror.WHEELS[side]).xpos - expected_wheel)))
                count += 1
    assert wheel_error < .001 and cold_error < .001, (wheel_error, cold_error)
    pose.data.qpos[:] = pose.model.qpos0
    frame = frame_for_pose(pose, pose.data.qpos)
    frame[22:26] = [np.cos(.2), np.sin(.2), 0, 0]
    frame[18:20] = [1., -2.]
    pose.apply(frame, 1.)
    assert pose.labels(1.) == ["OK"] * 5
    assert np.array_equal(pose.data.qpos[:3], fixed_translation)
    assert np.allclose(pose.data.qpos[3:7], frame[22:26], atol=1e-7)
    assert np.allclose(pose.data.qpos[pose.wheel_q], pose.sign[4:] * frame[18:20])
    previous = pose.data.qpos.copy()
    frame[29] = 127 & ~(1 << 5)
    frame[18:20] = [9., -3.]
    pose.apply(frame, 1.01)
    assert pose.status[3] == "INVALID WHEEL" and pose.status[4] == "OK"
    assert pose.data.qpos[pose.wheel_q[0]] == previous[pose.wheel_q[0]]
    assert pose.data.qpos[pose.wheel_q[1]] != previous[pose.wheel_q[1]]
    frame[22:26] = 0
    pose.apply(frame, 1.02)
    assert pose.status[0] == "INVALID IMU" and np.array_equal(pose.data.qpos[3:7], previous[3:7])
    before = pose.data.qpos.copy()
    frame[10:12] = [np.nan, np.inf]
    pose.apply(frame, 1.03)
    assert pose.status[1] == "INVALID FEEDBACK"
    assert np.array_equal(pose.data.qpos[pose.drive_q[0]], before[pose.drive_q[0]])
    frame = frame_for_pose(pose, pose.data.qpos.copy())
    frame[31] = 201
    frame[10:14] += .5
    frame[18:20] += 3
    saved_motor = pose.data.qpos[7:].copy()
    pose.apply(frame, 1.04)
    assert pose.status[1:] == ["STALE MOTOR SOURCE"] * 4
    assert np.array_equal(pose.data.qpos[7:], saved_motor)
    frame = frame_for_pose(pose, pose.data.qpos.copy())
    frame[30] = 201
    frame[22:26] = [1, 0, 0, 0]
    saved_quat = pose.data.qpos[3:7].copy()
    pose.apply(frame, 1.05)
    assert pose.status[0] == "STALE IMU SOURCE" and np.array_equal(pose.data.qpos[3:7], saved_quat)
    assert pose.stale(1.251) and pose.labels(1.251) == ["STALE"] * 5
    before = pose.data.qpos.copy()
    original_solver = mirror.least_squares
    try:
        mirror.least_squares = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad initial pose"))
        assert not pose._solve_leg(0, [1., 1.])
        assert np.array_equal(pose.data.qpos[pose.drive_q[0]], before[pose.drive_q[0]])
        from types import SimpleNamespace
        mirror.least_squares = lambda residual, initial, **kw: SimpleNamespace(x=initial)
        side_indices = np.concatenate((pose.drive_q[0], pose.passive_q[0], pose.spring_q[0]))
        saved = pose.data.qpos[side_indices].copy()
        assert not pose._solve_leg(0, saved[:2] + [.5, -.3])
        assert np.array_equal(pose.data.qpos[side_indices], saved)
    finally:
        mirror.least_squares = original_solver
    # Closed out-of-range solutions must stay visibly warned, never silently become OK.
    pose.data.qpos[:] = pose.model.qpos0
    pose.data.qpos[pose.passive_q[0][0]] = -3.
    assert pose._solve_leg(0, pose.model.qpos0[pose.drive_q[0]])
    assert pose.status[1] == "MODEL RANGE WARNING" and pose.residual_m[0] < .001
    # Actual received prone feedback: display both legs, without changing training limits.
    pose.data.qpos[:] = pose.model.qpos0
    limits = pose.model.jnt_range.copy()
    frame = frame_for_pose(pose, pose.model.qpos0)
    frame[10:14] = [.407824188, 2.022188663, .429779440, 1.983071923]
    pose.apply(frame, 2.)
    assert pose.status == ["OK", "MODEL RANGE WARNING", "MODEL RANGE WARNING", "OK", "OK"]
    for side in (0, 1):
        assert np.allclose(pose.data.qpos[pose.drive_q[side]],
                           pose.sign[side * 2:side * 2 + 2] * frame[10 + side * 2:12 + side * 2])
        assert pose.residual_m[side] < .001 and pose.seen[side + 1]
    pose.paint_status(2.)
    leg_geoms = np.flatnonzero(pose.model.geom_bodyid == pose.model.body("lf0").id)
    assert len(leg_geoms) and np.allclose(pose.model.geom_rgba[leg_geoms], [1., .65, .10, 1.])
    pose.paint_status(2.201)
    assert np.allclose(pose.model.geom_rgba[leg_geoms], [.85, .20, .16, 1.])
    pose.apply(frame_for_pose(pose, pose.model.qpos0), 2.3)
    assert pose.status == ["OK"] * 5
    pose.paint_status(2.3)
    assert np.array_equal(pose.model.geom_rgba, pose.original_rgba)
    assert np.array_equal(pose.model.jnt_range, limits)
    return {"known_model_leg_poses": count, "max_closure_m": maximum,
            "max_reconstructed_wheel_error_m": wheel_error,
            "cold_start_poses": count, "max_cold_start_wheel_error_m": cold_error,
            "solve_p95_ms": float(np.percentile(times, 95) * 1000),
            "individual_source_failure_and_200ms_stale": True,
            "publication_stop_with_live_uart": True,
            "closed_out_of_range_pose_displayed_with_warning": True,
            "actual_prone_feedback_color_stale_recovery_limits_unchanged": True,
            "scope": "model kinematics; assembly branch, physical signs and pose require bench confirmation"}


def check_serial(raw, directory):
    master, slave = pty.openpty()
    recorder = mirror.SessionRecorder(directory / "pty-recording", machine_config=H7 / "imcalib/config/machine_config.c")
    receiver = mirror.SerialReceiver(os.ttyname(slave), 1152000, recorder=recorder)
    receiver.start()
    try:
        deadline = time.monotonic() + 3
        while receiver.state != "CONNECTED" and time.monotonic() < deadline:
            time.sleep(.01)
        assert receiver.state == "CONNECTED", receiver.state
        os.write(master, raw[:37])
        time.sleep(.04)
        remaining = bytearray(raw[37:])
        for sequence in range(2, 21):
            frame = mirror.decode_frame(raw)
            frame[6:10] = (sequence, 0, sequence * 10, 0)
            remaining.extend(encode(frame))
        os.write(master, remaining)
        while time.monotonic() < deadline:
            if receiver.snapshot() is not None and mirror.read_u32(receiver.snapshot()[0][6:8]) == 20:
                break
            time.sleep(.01)
        assert receiver.snapshot() is not None
        assert mirror.read_u32(receiver.snapshot()[0][6:8]) == 20
        os.set_blocking(master, False)
        try:
            assert not os.read(master, 1024), "mirror wrote to H7"
        except BlockingIOError:
            pass
        os.close(master)
        master = None
        deadline = time.monotonic() + 1
        while not receiver.state.startswith("DISCONNECTED") and time.monotonic() < deadline:
            time.sleep(.01)
        assert receiver.state.startswith("DISCONNECTED"), receiver.state
    finally:
        receiver.stop_event.set()
        receiver.join(timeout=1.)
        recorder.finish()
        os.close(slave)
        if master is not None:
            os.close(master)
    assert not receiver.is_alive() and recorder.error is None
    mirror.load_session(recorder.directory)
    events = list(mirror.iter_session_events(recorder.directory))
    recorded = b"".join(bytes.fromhex(event["data_hex"]) for event in events if event["type"] == "data")
    assert recorded == raw[:37] + remaining
    assert len(mirror.FrameParser().feed(recorded)) == 20
    assert any(event["type"] == "disconnected" for event in events)
    return {"actual_serial_reader_fragmentation_disconnect": True, "no_outbound_bytes": True,
            "all_20_frames_recorded_independent_of_latest_display": True}


def check_sessions(directory, raw):
    """Exercise persisted raw chunks, replay timing and offline data validity."""
    config = H7 / "imcalib/config/machine_config.c"
    pose = mirror.MirrorPose(machine_config=config)
    base = mirror.decode_frame(raw)
    geometry = frame_for_pose(pose, pose.model.qpos0)
    base[10:14], base[18:20], base[22:26] = geometry[10:14], geometry[18:20], geometry[22:26]
    base[30:32] = (2, 3)

    def packet(sequence, tick, invalid=False, old=False):
        frame = base.copy()
        frame[6:10] = (sequence & 65535, sequence >> 16, tick & 65535, tick >> 16)
        frame[18:20] += np.array([.1, -.2]) * (sequence & 3)
        tilt = .03 * (sequence & 3)
        frame[22:26] = (np.cos(tilt), np.sin(tilt), 0, 0)
        if invalid:
            frame[29] = 0
            frame[10:29] = 0
        if old:
            frame[30:32] = (201, 201)
            frame[10:29] = 0
            frame[22] = 1
        return encode(frame)

    a, b, c = packet(0xfffffffe, 0xfffffff0), packet(0xffffffff, 0xfffffff9), packet(0, 4)
    d, e, f = packet(2, 24), packet(3, 34), packet(4, 44)
    damaged = bytearray(b)
    damaged[40:44] = struct.pack("<f", .123)  # Plausible payload: CRC must reject it.
    events = [("connected", 0, {"state": "CONNECTED"}),
              ("data", 1_000_000, {"data_hex": (b"noise" + a[:37]).hex()}),
              ("data", 2_000_000, {"data_hex": (a[37:] + damaged + b + c).hex()}),
              ("data", 10_000_000, {"data_hex": d.hex()}),
              ("data", 20_000_000, {"data_hex": d.hex()}),
              ("data", 30_000_000, {"data_hex": e[:80].hex()}),
              ("disconnected", 40_000_000, {"state": "DISCONNECTED: test unplug"}),
              ("connected", 41_000_000, {"state": "CONNECTED"}),
              ("data", 42_000_000, {"data_hex": e[80:].hex()}),
              ("data", 50_000_000, {"data_hex": f.hex()}),
              ("data", 330_000_000, {"data_hex": packet(1, 0).hex()}),
              ("data", 340_000_000, {"data_hex": packet(2, 10, invalid=True).hex()}),
              ("data", 350_000_000, {"data_hex": packet(3, 20, old=True).hex()})]
    recorder = mirror.SessionRecorder(directory / "recording", machine_config=config)
    for kind, when, values in events:
        recorder.record(kind, elapsed_ns=when, **values)
    recorder.finish(elapsed_ns=600_000_000)
    assert recorder.error is None
    meta, model_path, config_path = mirror.load_session(recorder.directory)
    assert meta["complete"] and model_path.resolve() == mirror.MODEL.resolve() and config_path.resolve() == config.resolve()
    saved = list(mirror.iter_session_events(recorder.directory))
    assert [(event["type"], event["t_ns"]) for event in saved[:-1]] == [(kind, when) for kind, when, _ in events]
    assert saved[-1]["type"] == "stop" and saved[-1]["t_ns"] == 600_000_000
    assert [event["data_hex"] for event in saved if event["type"] == "data"] == [values["data_hex"] for kind, _, values in events if kind == "data"]

    receiver = mirror.ReplayReceiver(recorder.directory)
    receiver.start()
    gap_stale = False
    previous = None
    started = time.monotonic()
    try:
        while not receiver.finished_event.is_set() and time.monotonic() - started < 3:
            snapshot = receiver.snapshot()
            if snapshot is not None and snapshot[1] != previous:
                pose.apply(*snapshot)
                previous = snapshot[1]
            now = receiver.now()
            if .26 < now < .32:
                assert snapshot is not None and np.isclose(snapshot[1], .05), snapshot
                frozen = pose.data.qpos.copy()
                assert pose.stale(now) and pose.labels(now) == ["STALE"] * 5
                assert np.array_equal(pose.data.qpos, frozen)
                gap_stale = True
            time.sleep(.002)
    finally:
        receiver.stop_event.set()
        receiver.join(timeout=1.)
    assert not receiver.is_alive() and receiver.error is None and gap_stale
    assert time.monotonic() - started >= .55, "replay did not preserve original duration"
    assert receiver.parser.accepted == 9 and receiver.parser.rejected >= 1
    assert (receiver.clock.missing, receiver.clock.duplicates, receiver.clock.restarts) == (2, 1, 1)
    assert mirror.read_u32(receiver.snapshot()[0][6:8]) == 3

    # Replay must finish without importing pyserial or opening a serial device.
    replay_report = directory / "replay-cli.json"
    # run_path needs the same module search path as direct script execution.
    result = subprocess.run([sys.executable, "-c", "import sys,runpy; from pathlib import Path; "
                             "sys.path.insert(0,str(Path(sys.argv[1]).parent)); sys.modules['serial']=None; "
                             "runpy.run_path(sys.argv.pop(1),run_name='__main__')",
                             str(ROOT / "sim2sim/real2sim_huananhu.py"), "--replay", str(recorder.directory),
                             "--headless", "--report", str(replay_report)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0 and replay_report.is_file(), result.stdout
    replayed = json.loads(replay_report.read_text())
    assert replayed["accepted_frames"] == 9 and replayed["duplicates"] == 1
    assert replayed["missing_submitted_frames"] == 2 and replayed["restarts"] == 1
    assert replayed["sources"] == ["STALE"] * 5
    early_report = directory / "replay-early-stop.json"
    early = subprocess.run(result.args[:-1] + [str(early_report), "--seconds", ".30"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    assert early.returncode == 0 and early_report.is_file(), early.stdout
    stopped = json.loads(early_report.read_text())
    assert stopped["accepted_frames"] == 6 and stopped["sources"] == ["STALE"] * 5
    report = mirror.analyze_session(recorder.directory)
    assert report["recording_complete"] and report["accepted_frames"] == 9 and report["applied_frames"] == 8
    assert (report["missing_submitted_frames"], report["duplicates"], report["restarts"]) == (2, 1, 1)
    assert report["connections"] == 2 and report["disconnections"] == 1
    analysis = recorder.directory / "analysis"
    with (analysis / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [int(row["sequence"]) for row in rows] == [0xfffffffe, 0xffffffff, 0, 2, 4, 1, 2, 3]
    for row in rows[-2:]:
        for column in ("lf0_raw_q_rad", "lf0_q_rad", "lf0_qd_rad_s", "lfwheel_q_rad", "quat_w", "roll_rad", "gyro_x_rad_s", "left_leg_length_m", "left_knee_rad", "left_closure_m"):
            assert row[column] == "", (column, row[column])
    assert np.isclose(float(rows[0]["lf0_q_rad"]), pose.sign[0] * (base[10] - pose.zero[0]))
    assert np.isclose(float(rows[0]["lfwheel_q_rad"]), pose.sign[4] * (base[18] + .2))
    assert np.isclose(float(rows[5]["lfwheel_q_rad"]), pose.sign[4] * (base[18] + .1))
    assert report["channels"]["lf0_q_rad"]["count"] == 6
    assert report["channels"]["left_closure_m"]["count"] == 6
    assert all(report["range_warnings"][side]["closed_frames"] == 6
               and report["range_warnings"][side]["warning_fraction"] == 0 for side in ("left", "right"))
    assert json.loads((analysis / "report.json").read_text())["applied_frames"] == 8
    assert (analysis / "signals.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def rejected(operation):
        try:
            operation()
        except (ValueError, OSError, RuntimeError):
            return
        raise AssertionError("malformed/incompatible recording was accepted")

    rejected(lambda: mirror.analyze_session(recorder.directory))
    changed_config = directory / "changed-machine-config.c"
    changed_config.write_text(config.read_text() + "\n/* changed recording configuration */\n")
    rejected(lambda: mirror.load_session(recorder.directory, machine_config=changed_config))
    changed_model = directory / "changed-model.xml"
    changed_model.write_text(mirror.MODEL.read_text() + "\n<!-- changed recording model -->\n")
    rejected(lambda: mirror.load_session(recorder.directory, model_path=changed_model))

    for mode in ("checksum", "truncated", "missing-stop", "backwards-time", "negative-time", "invalid-hex", "incomplete", "event-count", "duration"):
        target = directory / mode
        shutil.copytree(recorder.directory, target, ignore=shutil.ignore_patterns("analysis"))
        stream_path, meta_path = target / "stream.jsonl", target / "meta.json"
        content = stream_path.read_bytes()
        manifest = json.loads(meta_path.read_text())
        if mode == "incomplete":
            manifest["complete"] = False
        elif mode == "event-count":
            manifest["event_count"] += 1
        elif mode == "duration":
            manifest["duration_ns"] += 1
        elif mode == "checksum":
            stream_path.write_bytes(content + b" ")
        else:
            if mode == "truncated":
                content = content[:-5]
            else:
                timeline = [json.loads(line) for line in content.splitlines()]
                if mode == "missing-stop":
                    timeline.pop()
                elif mode == "backwards-time":
                    timeline[-2]["t_ns"] = 0
                elif mode == "negative-time":
                    timeline[0]["t_ns"] = -1
                else:
                    timeline[1]["data_hex"] = "zz"
                content = b"".join((json.dumps(event) + "\n").encode() for event in timeline)
            stream_path.write_bytes(content)
            manifest["stream_sha256"] = hashlib.sha256(content).hexdigest()
        meta_path.write_text(json.dumps(manifest))

        def load_and_iterate():
            mirror.load_session(target)
            list(mirror.iter_session_events(target))
        rejected(load_and_iterate)

    empty = mirror.SessionRecorder(directory / "no-frames", machine_config=config)
    empty.record("connected", elapsed_ns=0, state="CONNECTED")
    empty.finish(elapsed_ns=250_000_000)
    empty_report = mirror.analyze_session(empty.directory)
    assert empty_report["accepted_frames"] == empty_report["applied_frames"] == 0
    assert empty_report["no_data_before_first_frame_s"] == .25 and not empty_report["channels"]
    with (empty.directory / "analysis/trajectory.csv").open(newline="") as stream:
        no_rows = csv.DictReader(stream)
        assert "lf0_q_rad" in no_rows.fieldnames and not list(no_rows)

    class BrokenRecorder(mirror.SessionRecorder):
        def _write_event(self, event):
            raise OSError("injected disk write failure")

    broken = BrokenRecorder(directory / "write-failed", machine_config=config)
    broken.record("connected", elapsed_ns=0, state="CONNECTED")
    try:
        broken.finish()
    except (OSError, RuntimeError):
        pass
    assert broken.error and "injected disk write failure" in broken.error
    assert not json.loads((broken.directory / "meta.json").read_text())["complete"]
    rejected(lambda: mirror.load_session(broken.directory))
    gate, entered = threading.Event(), threading.Event()

    class SlowRecorder(mirror.SessionRecorder):
        def _write_event(self, event):
            entered.set()
            if not gate.wait(timeout=3.):
                raise OSError("test writer gate timeout")
            return super()._write_event(event)

    saturated = SlowRecorder(directory / "queue-full", machine_config=config)
    saturated.pending.maxsize = 1
    try:
        saturated.record("connected", elapsed_ns=0, state="CONNECTED")
        assert entered.wait(timeout=1.)
        saturated.record("data", elapsed_ns=1, data_hex="00")
        saturated.record("data", elapsed_ns=2, data_hex="01")
        assert saturated.error, "full writer queue silently discarded serial bytes"
    finally:
        gate.set()
        try:
            saturated.finish()
        except (OSError, RuntimeError):
            pass
    assert not json.loads((saturated.directory / "meta.json").read_text())["complete"]
    rejected(lambda: mirror.load_session(saturated.directory))
    return {"raw_chunks_preserved_fragmentation_bad_crc": True,
            "reconnect_discards_partial_frame": True,
            "replay_original_timing_200ms_stale_no_serial": True,
            "sequence_tick_wrap_gaps_duplicates_restart": True,
            "csv_png_exclude_invalid_and_stale_source_values": True,
            "no_received_frames_produces_empty_csv_schema": True,
            "model_config_checksum_truncation_eof_time_validation": True,
            "disk_write_failure_marks_incomplete": True,
            "writer_queue_full_marks_incomplete": True,
            "scope": "synthetic recording and PTY; no physical movement acceptance"}


def check_viewer_timing(directory, raw):
    """Viewer startup/shutdown must not consume sampling or age the final report."""
    from contextlib import nullcontext, redirect_stdout
    import io
    from types import ModuleType, SimpleNamespace

    config = H7 / "imcalib/config/machine_config.c"
    pose = mirror.MirrorPose(machine_config=config)
    frame = mirror.decode_frame(raw)
    geometry = frame_for_pose(pose, pose.model.qpos0)
    frame[10:14], frame[18:20], frame[22:26] = geometry[10:14], geometry[18:20], geometry[22:26]
    frame[29:32] = (127, 0, 0)

    class Viewer:
        def __init__(self):
            self.cam = SimpleNamespace(lookat=np.zeros(3))
            self.opt = SimpleNamespace(geomgroup=np.zeros(6))
            self.closed = False
            self.text_updates = []
            self.close_requested = threading.Event()
            self.renderer_done = threading.Event()
            self.renderer = None

        def lock(self):
            return nullcontext()

        def is_running(self):
            return not self.closed

        def set_texts(self, texts):
            self.text_updates.append(list(texts))

        def sync(self):
            pass

        def close(self):
            time.sleep(.25)
            self.closed = True
            self.close_requested.set()

    viewer = Viewer()
    viewer_module = ModuleType("mujoco.viewer")

    def launch(*args, **kwargs):
        time.sleep(.25)

        def renderer():
            viewer.close_requested.wait()
            time.sleep(.05)  # close() returns before native-window destruction ends.
            viewer.renderer_done.set()

        viewer.renderer = threading.Thread(target=renderer, daemon=True)
        viewer.renderer.start()
        return viewer

    viewer_module.launch_passive = launch
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    stopped, errors = threading.Event(), []

    def board():
        sequence, pending = 0, b""
        try:
            while not stopped.is_set():
                if not pending:
                    sequence += 1
                    values = frame.copy()
                    values[6:10] = (sequence & 65535, sequence >> 16, sequence & 65535, sequence >> 16)
                    pending = encode(values)
                try:
                    sent = os.write(master, pending)
                    pending = pending[sent:]
                except BlockingIOError:
                    pass
                stopped.wait(.001)
        except BaseException as exc:
            errors.append(repr(exc))

    session, report_path = directory / "viewer-timing", directory / "viewer-timing.json"
    original_argv = sys.argv
    original_module = sys.modules.get("mujoco.viewer")
    original_viewer = getattr(mujoco, "viewer", None)
    worker = threading.Thread(target=board, daemon=True)
    terminal_output = io.StringIO()
    try:
        # Intercept the module itself so the check cannot open a real window.
        sys.modules["mujoco.viewer"] = viewer_module
        mujoco.viewer = viewer_module
        sys.argv = [str(ROOT / "sim2sim/real2sim_huananhu.py"), "--port", os.ttyname(slave),
                    "--machine-config", str(config), "--record", str(session),
                    "--seconds", ".15", "--report", str(report_path)]
        worker.start()
        with redirect_stdout(terminal_output):
            assert mirror.main() == 0
            assert viewer.renderer_done.is_set() and not viewer.renderer.is_alive(), "main returned before renderer destruction"
    finally:
        stopped.set()
        worker.join(timeout=1.)
        viewer.close_requested.set()
        if viewer.renderer is not None:
            viewer.renderer.join(timeout=1.)
        sys.argv = original_argv
        if original_module is None:
            sys.modules.pop("mujoco.viewer", None)
        else:
            sys.modules["mujoco.viewer"] = original_module
        if original_viewer is None:
            del mujoco.viewer
        else:
            mujoco.viewer = original_viewer
        os.close(master)
        os.close(slave)
    assert not worker.is_alive() and not errors, errors
    report = json.loads(report_path.read_text())
    assert viewer.closed and report["recording_complete"] and report["accepted_frames"] > 5
    assert report["sources"] == ["OK"] * 5, report
    panels = [text[2] for update in viewer.text_updates for text in update
              if text[1] == mujoco.mjtGridPos.mjGRID_TOPRIGHT.value]
    assert panels and all("REC RECORDING |" in panel and " | left " in panel
                          and f"SAVE {session}" in panel and "Pending writes " in panel for panel in panels)
    assert any(float(re.search(r"REC RECORDING \| ([\d.]+)s", panel).group(1)) > 0 for panel in panels)
    assert any(int(re.search(r"RX frames (\d+)", panel).group(1)) > 0 for panel in panels)
    assert any(float(re.search(r"stream ([\d.]+) MiB", panel).group(1)) > 0 for panel in panels)
    messages = [json.loads(line) for line in terminal_output.getvalue().splitlines() if line.strip()]
    progress = [message["recording"] for message in messages if "recording" in message]
    assert progress and progress[0]["status"] == "RECORDING" and progress[-1]["status"] == "SAVED"
    saved = progress[-1]
    assert saved["received_frames"] == report["accepted_frames"] and saved["received_frames"] > 5
    assert saved["written_bytes"] == (session / "stream.jsonl").stat().st_size > 0
    assert saved["pending_events"] == 0 and saved["error"] is None
    assert saved["directory"] == str(session.resolve()) and saved["remaining_s"] == 0
    meta, _, _ = mirror.load_session(session)
    assert meta["format"] == "h7-mirror-session" and meta["version"] == 1 and meta["complete"]
    data = [event for event in mirror.iter_session_events(session) if event["type"] == "data"]
    active_seconds = (meta["duration_ns"] - data[0]["t_ns"]) / 1e9
    assert data[0]["t_ns"] >= 240_000_000, "fake viewer startup did not delay USB reading"
    assert active_seconds >= .15, ("viewer startup consumed --seconds budget", active_seconds)
    return {"delayed_viewer_start_does_not_consume_sampling_budget": True,
            "delayed_viewer_close_does_not_mark_fresh_final_sources_stale": True,
            "renderer_destruction_finishes_before_main_returns": True,
            "recording_hud_elapsed_frames_bytes_directory": True,
            "terminal_recording_and_saved_with_exact_stream_bytes": True,
            "requested_sampling_s": .15, "first_data_to_stop_s": active_seconds,
            "scope": "fake viewer and PTY only; no real graphics or H7 hardware"}


def check_probe(directory, raw):
    """Exercise actual H7 RX code, then the pyserial checker through a PTY."""
    stub = directory / "probe-stub"
    stub.mkdir()
    (stub / "Vofa_send.h").write_text(r'''
#include <stdint.h>
typedef struct { uint16_t counter; unsigned interrupts, state, locked; } DMA_HandleTypeDef;
typedef struct { unsigned RxState; DMA_HandleTypeDef *hdmarx; } UART_HandleTypeDef;
extern UART_HandleTypeDef uart;
#define VOFA_UART (&uart)
#define HAL_UART_STATE_BUSY_RX 1u
#define HAL_OK 0u
#define DMA_IT_HT 1u
#define DMA_IT_TC 2u
#define HAL_DMA_STATE_READY 0u
#define HAL_DMA_STATE_BUSY 1u
#define HAL_DMA_STATE_ABORT 2u
#define __HAL_DMA_DISABLE_IT(dma,bit) ((dma)->interrupts &= ~(bit))
#define __HAL_DMA_GET_COUNTER(dma) ((dma)->counter)
unsigned HAL_UART_Receive_DMA(UART_HandleTypeDef *, uint8_t *, uint16_t);
''')
    (stub / "dma_cache.h").write_text(r'''
#define DMA_CACHE_LINE_SIZE 32u
#define Dma_Cache_Invalidate_Rx(data,size) ((void)0)
''')
    packet = link.encode_probe(0xffffffff, 0x12345678)
    source = directory / "uart-probe.c"
    source.write_text(r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "''' + str(H7 / "imcalib/drivers/jetson_uart.c") + r'''"
static DMA_HandleTypeDef dma;
UART_HandleTypeDef uart = {0u, &dma};
static uint8_t *destination;
static uint16_t cursor;
static uint8_t fail_start;
unsigned HAL_UART_Receive_DMA(UART_HandleTypeDef *handle, uint8_t *data, uint16_t size) {
    assert(size == 256 && (uintptr_t)data % 32 == 0);
    if (fail_start || dma.state != HAL_DMA_STATE_READY || dma.locked) { return 1u; }
    dma.state = HAL_DMA_STATE_BUSY; dma.locked = 1u;
    dma.interrupts = DMA_IT_HT | DMA_IT_TC;
    destination = data; cursor = 0; dma.counter = size; handle->RxState = 1u;
    return 0u;
}
/* Model HAL UART blocking-error abort and DMA TC completion, including its lock.
 * See stm32h7xx_hal_uart.c / stm32h7xx_hal_dma.c; this is not a hardware IRQ test. */
static void rx_error(void) {
    assert(dma.state == HAL_DMA_STATE_BUSY && dma.locked);
    uart.RxState = 0u;
    dma.state = HAL_DMA_STATE_ABORT;
}
static void dma_abort_irq(void) {
    if ((dma.interrupts & DMA_IT_TC) && dma.state == HAL_DMA_STATE_ABORT) {
        dma.interrupts = 0u;
        dma.state = HAL_DMA_STATE_READY;
        dma.locked = 0u;
    }
}
static void push(const uint8_t *data, unsigned size) {
    for (unsigned i=0; i<size; i++) {
        destination[cursor] = data[i]; cursor = (cursor + 1u) % 256u;
    }
    dma.counter = 256u - cursor;
    Jetson_Uart_Process();
}
int main(int argc, char **argv) {
    const uint8_t good[14] = {''' + ",".join(str(v) for v in packet) + r'''};
    uint8_t damaged[14], noise[250];
    unsigned accepted;
    int byte;
    Jetson_Uart_Init();
    if (argc == 2 && strcmp(argv[1], "--serve") == 0) {
        while ((byte = getchar()) != EOF) {
            uint8_t value = byte;
            push(&value, 1);
            assert(fwrite(&jetson_uart_status.sequence, 4, 1, stdout) == 1);
            assert(fwrite(&jetson_uart_status.token, 4, 1, stdout) == 1);
            assert(fwrite(&jetson_uart_status.received, 4, 1, stdout) == 1);
            fflush(stdout);
        }
        return 0;
    }
    assert(Jetson_Crc16((const uint8_t *)"123456789",9) == 0x29b1);
    for (unsigned split=1; split<14; split++) {
        accepted = jetson_uart_status.received;
        push(good, split); assert(jetson_uart_status.received == accepted);
        push(good+split, 14-split); assert(jetson_uart_status.received == accepted+1);
        assert(jetson_uart_status.sequence == 0xffffffffu);
        assert(jetson_uart_status.token == 0x12345678u);
    }
    /* 跨DMA环尾、噪声、坏CRC、连续粘包和计数回绕。 */
    memset(noise, 0x55, sizeof(noise)); push(noise, sizeof(noise));
    memcpy(damaged, good, 14); damaged[8] ^= 1;
    accepted = jetson_uart_status.received;
    push(damaged, 14); assert(jetson_uart_status.received == accepted);
    assert(jetson_uart_status.crc_errors == 1);
    push(good, 14); push(good, 14); assert(jetson_uart_status.received == accepted+2);
    jetson_uart_status.received = 0xffffffffu;
    push(good,14); assert(jetson_uart_status.received == 0);
    /* UART错误后先等待异步终止，再覆盖启动失败及半包丢弃。 */
    push(good,7); rx_error(); Jetson_Uart_Process();
    assert(jetson_uart_status.rx_restarts == 1);
    assert(dma.state == HAL_DMA_STATE_ABORT && dma.locked && uart.RxState == 0u);
    dma_abort_irq();
    assert(dma.state == HAL_DMA_STATE_READY && !dma.locked);
    fail_start=1; Jetson_Uart_Process();
    assert(jetson_uart_status.rx_restarts == 2);
    fail_start=0; Jetson_Uart_Process();
    assert(jetson_uart_status.rx_restarts == 3);
    assert(dma.state == HAL_DMA_STATE_BUSY && dma.locked && uart.RxState == 1u);
    push(good+7,7); assert(jetson_uart_status.received == 0);
    push(good,14); assert(jetson_uart_status.received == 1);
    puts("actual H7 UART parser checks passed");
    return 0;
}
''')
    executable = directory / "uart-probe"
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-I", str(stub),
                    str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)

    def session(mode):
        master, slave = pty.openpty()
        stopped = threading.Event()
        board_errors = []
        received_bytes = 0
        process = subprocess.Popen([str(executable), "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        def board():
            nonlocal received_bytes
            count = 0
            status = (0, 0, 0)
            corrupted = False
            try:
                while not stopped.is_set():
                    if select.select([master], [], [], .005)[0]:
                        data = os.read(master, 4096)
                        received_bytes += len(data)
                        if mode == "bad-first" and not corrupted:
                            data = bytearray(data)
                            data[0] ^= 1  # Drop a request; next valid header must resynchronise.
                            corrupted = True
                        process.stdin.write(data)
                        process.stdin.flush()
                        reply = process.stdout.read(len(data) * 12)
                        assert len(reply) == len(data) * 12
                        status = struct.unpack("<III", reply[-12:])
                    values = list(struct.unpack("<48f", raw[:192]))
                    count += 1
                    values[6:10] = (count & 65535, count >> 16, count & 65535, count >> 16)
                    values[32:39] = [0. if mode == "old" else 1., status[0] & 65535,
                                    status[0] >> 16, status[1] & 65535, status[1] >> 16,
                                    status[2] & 65535, status[2] >> 16]
                    payload = struct.pack("<47f", *values[:47])
                    frame = payload + struct.pack("<f", binascii.crc_hqx(payload, 0xffff)) + link.TAIL
                    # Real serial stream fragmentation in both OS reads and writes.
                    os.write(master, b"noise" + frame[:37])
                    os.write(master, frame[37:])
            except BaseException as exc:
                board_errors.append(repr(exc))

        import serial
        worker = threading.Thread(target=board, daemon=True)
        try:
            with serial.Serial(os.ttyname(slave), 1152000, timeout=.01, write_timeout=.1) as connection:
                worker.start()
                report = link.check_link(connection, 1.1)
        finally:
            stopped.set()
            worker.join(timeout=2.)
            process.stdin.close()
            process.wait(timeout=2.)
            process.stdout.close()
            os.close(master)
            os.close(slave)
        assert not worker.is_alive() and not board_errors, board_errors
        if mode == "normal":
            assert report["ok"] and report["probe_ack"] >= 5 and report["timeouts"] == 0, report
        elif mode == "bad-first":
            assert not report["ok"] and report["timeouts"] == 1 and report["probe_ack"] > 0, report
        else:
            assert not report["ok"] and report["probe_sent"] == 0 and received_bytes == 0, report
        return report

    def deadline_session(delay):
        # A deterministic read/scheduling delay crosses the deadline in one read.
        class Clock:
            now = 0.
            def monotonic(self):
                return self.now

        clock = Clock()

        class Connection:
            in_waiting = link.FRAME_BYTES
            sequence = token = reads = 0
            def read(self, size):
                if self.reads:
                    clock.now += delay
                self.reads += 1
                values = list(struct.unpack("<48f", raw[:192]))
                values[6:8] = (self.reads, 0)
                values[32:39] = (1., self.sequence & 65535, self.sequence >> 16,
                                 self.token & 65535, self.token >> 16, 0, 0)
                return encode(values)

            def write(self, packet):
                _, self.sequence, self.token = struct.unpack("<4sII", packet[:12])
                return len(packet)

        original_time = link.time
        try:
            link.time = clock
            report = link.check_link(Connection(), 1.1)
        finally:
            link.time = original_time
        assert report["probe_sent"] > 0
        if delay < link.ACK_TIMEOUT_S:
            assert report["ok"] and report["probe_ack"] == report["probe_sent"] and not report["timeouts"], report
        else:
            assert not report["ok"] and report["timeouts"] == report["probe_sent"] and not report["probe_ack"], report
        return report

    return {"actual_c_parser": "fragmentation, noise, CRC, DMA wrap/restart, uint32 wrap",
            "dma_error_recovery": "modeled HAL ABORT/TC/READY/lock; actual driver; no hardware fault injection",
            "pty": {mode: session(mode) for mode in ("normal", "bad-first", "old")},
            "ack_deadline": {str(delay): deadline_session(delay) for delay in (.249, .25, .3)}}


def build_h7(directory):
    import yaml
    config = yaml.safe_load((H7 / ".eide/eide.yml").read_text())
    sources = set()
    libraries = set()
    missing_references = []
    def visit(folder):
        for item in folder["files"]:
            path = H7 / item["path"]
            if not path.is_file():
                if path.suffix.lower() in (".c", ".s", ".a"):
                    missing_references.append(item["path"])
                continue
            if path.suffix.lower() in (".c", ".s"):
                sources.add(path)
            elif path.suffix == ".a":
                libraries.add(path)
        for child in folder["folders"]:
            visit(child)
    visit(config["virtualFolder"])
    for name in config["srcDirs"]:
        sources.update((H7 / name).rglob("*.c"))
    target = config["targets"]["CtrBoard-H7_ALL"]
    attrs = target["cppPreprocessAttrs"]
    gcc = os.environ.get("ARM_CC") or shutil.which("arm-none-eabi-gcc")
    assert gcc, "ARM GCC missing"
    common = [gcc, "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard",
              "--specs=nano.specs", "--specs=nosys.specs", "-g3", "-O0", "-Wall",
              "-ffunction-sections", "-fdata-sections"]
    common += ["-D" + d for d in attrs["defineList"]]
    common += ["-I" + str(H7 / inc) for inc in attrs["incList"]]
    summaries = []
    for machine in (0, 1):
        output = directory / f"h7-machine-{machine}"
        output.mkdir(parents=True, exist_ok=True)
        commands = []
        for i, source in enumerate(sorted(sources)):
            obj = output / f"{i:03}-{source.stem}.o"
            language = ["-x", "assembler-with-cpp"] if source.suffix.lower() == ".s" else ["-std=gnu11"]
            commands.append(common + [f"-DMACHINE_DEFAULT={machine}"] + language + ["-c", str(source), "-o", str(obj)])
        def compile_one(command):
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return result.returncode, result.stdout
        with ThreadPoolExecutor(max_workers=8) as pool:
            compiled = list(pool.map(compile_one, commands))
        (output / "compile.log").write_text("\n".join(r[1] for r in compiled))
        (output / "commands.json").write_text(json.dumps(commands, indent=2))
        assert all(code == 0 for code, _ in compiled), f"compile failed: {output / 'compile.log'}"
        elf = output / "firmware.elf"
        link = common[:5] + ["--specs=nano.specs", "--specs=nosys.specs", "-nostartfiles",
                "-Wl,--gc-sections", "-T" + str(H7 / target["toolchainConfigMap"]["GCC"]["scatterFilePath"]),
                "-Wl,-Map," + str(output / "firmware.map")]
        link += [command[-1] for command in commands]
        link += ["-Wl,--start-group"] + [str(p) for p in sorted(libraries)] + ["-lc", "-lm", "-Wl,--end-group", "-o", str(elf)]
        result = subprocess.run(link, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / "link.log").write_text(result.stdout)
        (output / "link-command.json").write_text(json.dumps(link, indent=2))
        assert result.returncode == 0, f"link failed: {output / 'link.log'}"
        summaries.append({"machine": machine, "translation_units": len(commands), "linked": True,
                          "absent_eide_virtual_references": missing_references,
                          "elf": str(elf), "warning_log": str(output / "compile.log")})
    return summaries


def main():
    global H7
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h7-root", type=Path, default=H7, help="firmware project to check/build (default: deployment/H7_RL)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-h7", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="ARM compile/link only; no native C compiler needed")
    parser.add_argument("--dynamics-only", action="store_true", help="offline MuJoCo dynamics checks; no firmware compiler needed")
    parser.add_argument("--dynamics-run", type=Path, help="with --dynamics-only: verify saved prediction states and statistics")
    parser.add_argument("--coupling-only", action="store_true", help="offline bounded wheel inertia-allocation checks")
    parser.add_argument("--coupling-run", type=Path, help="with --coupling-only: verify a saved allocation study")
    args = parser.parse_args()
    H7 = args.h7_root.resolve()
    usb = "CDC_Transmit_HS(" in (H7 / "imcalib/bsp/Vofa_send.c").read_text()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.coupling_only:
        if args.build_only or args.build_h7 or args.dynamics_only:
            parser.error("--coupling-only cannot be combined with firmware builds or --dynamics-only")
        from check_real2sim_coupling_huananhu import run_checks, verify_run
        with tempfile.TemporaryDirectory(prefix="real2sim-coupling-check-") as temporary:
            results = run_checks(Path(temporary))
        report = {"ok": True, "scope": "offline inertia-allocation checks; no hardware acceptance", "results": results}
        if args.coupling_run:
            report['recorded_run_verification'] = verify_run(args.coupling_run)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.dynamics_only:
        if args.build_only or args.build_h7:
            parser.error("--dynamics-only cannot be combined with firmware builds")
        from check_real2sim_dynamics_huananhu import run_checks, verify_run
        with tempfile.TemporaryDirectory(prefix="real2sim-dynamics-check-") as temporary:
            results = run_checks(Path(temporary))
        report = {"ok": True, "scope": "offline dynamics software checks; no hardware acceptance", "results": results}
        if args.dynamics_run:
            report['recorded_run_verification'] = verify_run(args.dynamics_run)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    results = {}
    if not args.build_only:
        with tempfile.TemporaryDirectory(prefix="real2sim-check-") as temporary:
            raw, c_result = c_packets(Path(temporary), usb)
            results = {"actual_h7_packer": c_result, "stream": check_stream(raw),
                       "kinematics": check_geometry(), "serial": check_serial(raw, Path(temporary)),
                       "sessions": check_sessions(Path(temporary), raw),
                       "v2_sessions": check_v2_sessions(Path(temporary), raw),
                       "viewer_timing": check_viewer_timing(Path(temporary), raw)}
            if usb:
                results["usb_sender"] = check_usb_sender(Path(temporary), raw)
                results["wheel_response"] = check_wheel_response(Path(temporary))
                if c_result["protocol_version"] == 2:
                    results["policy_producer"] = check_policy_producer(Path(temporary))
                    results["output_producer"] = check_output_producer(Path(temporary))
                    results["wheel_bench"] = check_wheel_bench(Path(temporary))
            else:
                results["bidirectional_uart"] = check_probe(Path(temporary), raw)
    if args.build_h7 or args.build_only:
        results["h7_builds"] = build_h7(args.out.parent)
    report = {"ok": True, "h7_root": str(H7), "transport": "USB CDC" if usb else "UART",
              "scope": "software and model checks; no H7 hardware/axis/branch acceptance", "results": results}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
