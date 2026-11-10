"""Finite host checks of the actual Fudan H7_ros graph, observation FIFO and PD.

Uses ST's desktop runtime and an existing native C compiler; never connects to a board.
Artifacts include exact compiler command, vectors, actual C outputs and source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from ct_mjlab.fudan_ppo_walk.export import Policy
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv


HARNESS = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "rl_policy.h"
#include "rl_torque.h"
#include "networkzn1.h"
#include "networkzn1_data.h"
static uint64_t tick;
uint64_t Mono_Ns_Get(void) { return ++tick * 1000u; }
static machine_rl_cfg_t config = CONFIG_INITIALIZER;
#define FAULT_NONE 0u
static struct { uint8_t reserved; } wheel_bench;
static struct { uint8_t enabled,recovery_enabled; } standup_control;
static struct { uint8_t fallen; } robot_state;
static struct { rl_recovery_state_t recovery; } rl_control;
static rc_command_t rc_command;
static const machine_cfg_t test_machine={.rl={.configured=1}};
const machine_cfg_t *const machine=&test_machine;
static uint32_t ctrl_fault;
PERMISSION_FUNCTION
static void output_gate_check(void)
{
    unsigned zero_output,ctrl_strategy=CTRL_STRATEGY_RL;
    rc_command.online=1; rc_command.mode.rc_enable=1; rc_command.mode.engage=1;
    rc_command.mode.strategy=CTRL_STRATEGY_RL;
    for(unsigned phase=0;phase<5;phase++) {
        for(unsigned stood=0;stood<2;stood++) {
            rl_control.recovery.phase=phase; rl_control.recovery.has_stood=stood;
            zero_output=0;
            OUTPUT_GATE
            assert(zero_output==(phase!=RL_RECOVERY_NORMAL || !stood));
            robot_state.fallen=0;
            assert(Robot_Control_Enable_Allowed()==(phase!=RL_RECOVERY_FAILED));
            robot_state.fallen=1;
            assert(!Robot_Control_Enable_Allowed());
        }
    }
    robot_state.fallen=0; rl_control.recovery.phase=RL_RECOVERY_NORMAL;
    ctrl_fault=1; assert(!Robot_Control_Enable_Allowed()); ctrl_fault=0;
    rc_command.online=0; assert(!Robot_Control_Enable_Allowed());
}
static void recovery_check(void)
{
    rl_recovery_state_t s = {0};
    const float load[2] = {65.700435f,65.700435f}, empty[2] = {0.f,0.f};
    RL_Recovery_Update(&s,1,1,0,.26f,load,-1.f);
    assert(s.phase == RL_RECOVERY_HOLD);
    for(unsigned i=0;i<100;i++) { RL_Recovery_Update(&s,1,1,0,.26f,load,-1.f); }
    assert(s.phase == RL_RECOVERY_NORMAL && s.has_stood && s.tracking_ms == 500);
    for(unsigned i=0;i<20;i++) { RL_Recovery_Update(&s,1,1,0,.26f,empty,-1.f); }
    assert(s.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&s,1,1,0,.26f,load,-1.f);
    assert(s.phase == RL_RECOVERY_FAILED);
    RL_Recovery_Update(&s,0,1,0,.26f,load,-1.f);
    assert(s.phase == RL_RECOVERY_IDLE && !s.has_stood);
    RL_Recovery_Update(&s,1,0,0,.26f,load,-1.f);
    assert(s.phase == RL_RECOVERY_IDLE && !s.support_valid);
    assert(fabsf(RL_Command_Height(.0f)-.22f)<1e-7f);
    assert(fabsf(RL_Command_Height(.5f)-.30f)<1e-7f);
}
int main(int argc, char **argv)
{
    assert(argc == 8);
    output_gate_check();
    recovery_check();
    static ai_u64 memory[(AI_NETWORKZN1_DATA_ACTIVATIONS_SIZE+7)/8];
    const ai_handle activations[] = {AI_HANDLE_PTR(memory)};
    ai_handle network = AI_HANDLE_NULL;
    ai_error error = ai_networkzn1_create_and_init(&network,activations,NULL);
    assert(error.type == AI_ERROR_NONE);
    ai_u16 ni=0,no=0;
    ai_buffer *in=ai_networkzn1_inputs_get(network,&ni), *out=ai_networkzn1_outputs_get(network,&no);
    assert(ni==2 && no==1 && AI_NETWORKZN1_IN_1_SIZE==25 && AI_NETWORKZN1_IN_2_SIZE==125);
    FILE *input=fopen(argv[1],"rb"), *raw=fopen(argv[2],"wb"), *clipped=fopen(argv[3],"wb");
    assert(input && raw && clipped);
    rl_policy_t policy = {0};
    rl_observation_state_t obs = {0};
    float row[150], action[6], output[6];
    assert(RL_Policy_Init(&policy));
    unsigned n=0;
    while(fread(row,sizeof(row),1,input)==1) {
        in[0].data=AI_HANDLE_PTR(row); in[1].data=AI_HANDLE_PTR(row+25); out[0].data=AI_HANDLE_PTR(output);
        assert(ai_networkzn1_run(network,in,out)==1);
        assert(fwrite(output,sizeof(output),1,raw)==1);
        memcpy(obs.obs,row,sizeof(obs.obs)); memcpy(obs.history,row+25,sizeof(obs.history)); obs.valid=1;
        assert(RL_Policy_Run(&policy,&obs,action));
        assert(fwrite(action,sizeof(action),1,clipped)==1);
        RL_Observation_Set_Last_Action(&obs,action);
        assert(memcmp(obs.last_action,action,sizeof(action))==0);
        n++;
    }
    assert(n==256 && policy.run_ok==256);
    obs.obs[0]=NAN;
    assert(!RL_Policy_Run(&policy,&obs,action) && policy.run_fail==1);
    for(unsigned i=0;i<6;i++) { assert(action[i]==0.f); }
    fclose(input); fclose(raw); fclose(clipped);
    ai_networkzn1_destroy(network);
    input=fopen(argv[4],"rb"); raw=fopen(argv[5],"wb"); assert(input && raw);
    RL_Observation_Reset(&obs);
    float sample_row[30]; n=0;
    while(fread(sample_row,sizeof(sample_row),1,input)==1) {
        rl_observation_sample_t sample;
        rl_recovery_state_t recovery={0}; rc_command_t cmd={0};
        if(sample_row[29]!=0.f) { RL_Observation_Reset(&obs); }
        memcpy(sample.gyro,sample_row,12); memcpy(sample.gravity,sample_row+3,12);
        memcpy(sample.pos,sample_row+6,16); memcpy(sample.vel,sample_row+10,24);
        RL_Observation_Set_Last_Action(&obs,sample_row+16);
        cmd.vel=sample_row[22]; cmd.yaw=sample_row[23]; cmd.len=sample_row[24];
        recovery.phase=(rl_recovery_phase_t)sample_row[25]; recovery.has_stood=(uint8_t)sample_row[26]; recovery.tracking_ms=(uint16_t)sample_row[27];
        uint8_t valid=RL_Observation_Build(&obs,&config,&sample,&cmd,&recovery,(uint8_t)sample_row[28]);
        assert(fwrite(obs.obs,sizeof(obs.obs),1,raw)==1); assert(fwrite(obs.history,sizeof(obs.history),1,raw)==1);
        const float flags[2]={obs.command_velocity,(float)valid}; assert(fwrite(flags,sizeof(flags),1,raw)==1); n++;
    }
    assert(n==80); fclose(input); fclose(raw);
    input=fopen(argv[6],"rb"); raw=fopen(argv[7],"wb"); assert(input && raw);
    float torque_row[16]; n=0;
    while(fread(torque_row,sizeof(torque_row),1,input)==1) {
        rl_motor_state_t motor={0}; rl_torque_state_t state={0}; torque_output_t torque;
        memcpy(motor.pos,torque_row+6,sizeof(motor.pos)); memcpy(motor.vel,torque_row+10,sizeof(motor.vel)); motor.valid=1;
        assert(RL_Torque_Compute(&motor,&config,torque_row,&state,&torque));
        assert(fwrite(state.target,sizeof(state.target),1,raw)==1); assert(fwrite(state.motor_torque,sizeof(state.motor_torque),1,raw)==1);
        assert(fwrite(torque.dm,sizeof(torque.dm),1,raw)==1); assert(fwrite(torque.dji,sizeof(torque.dji),1,raw)==1); n++;
    }
    assert(n==256); fclose(input); fclose(raw);
    puts("actual SDK/FIFO/PD/recovery checks passed; desktop only");
    return 0;
}
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--h7-root', type=Path, required=True)
    parser.add_argument('--st-host', type=Path, required=True, help='ST inspector/workspace directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(14000)
    network, metadata = load_policy(args.checkpoint)
    assert metadata['reset_contract'].endswith('unclipped-history-v2')
    policy = Policy(network).eval()
    vectors = rng.uniform(-1, 1, (256, 150)).astype(np.float32)
    vectors[0] = 0
    vectors[1, :] = .5
    with torch.no_grad():
        expected = policy(torch.from_numpy(vectors[:, :25]), torch.from_numpy(vectors[:, 25:])).numpy()
    evaluator = ReferenceEvaluator(onnx.load(args.onnx))
    with np.errstate(over='ignore'):
        reference = np.concatenate([evaluator.run(None, {'obs': x[None,:25], 'obs_history': x[None,25:]})[0] for x in vectors])
    onnx_error = float(np.abs(reference-expected).max())
    assert np.isfinite(reference).all() and onnx_error <= 1e-4, onnx_error
    vectors.tofile(args.out/'inputs.bin'); expected.tofile(args.out/'pytorch.bin'); reference.tofile(args.out/'onnx.bin')
    # Feed actual training frame() with the same already-mapped sensor measurements.
    rows = rng.uniform(-1, 1, (80, 30)).astype(np.float32)
    rows[:, 16:22] *= 100
    rows[:, 22:24] *= 4
    rows[:, 24] = np.linspace(.15,.35,80)
    rows[:, 25:30] = [3,1,500,1,0]
    rows[[0,30],29] = 1
    rows[10:14,25:28] = [2,0,0]
    rows[20,28] = 0
    rows[40,25] = 4
    rows[50,0] = 1000  # FIFO must retain 250 while current obs clips to 100.
    rows[60,0] = np.nan
    rows.tofile(args.out/'sensor-inputs.bin')
    history = np.zeros(125,np.float32); initialized = False; obs_reference=[]
    nominal = np.array([-.604720554,.608324173,.604720554,-.608324173],np.float32)
    for row in rows:
        if row[29]: history[:]=0; initialized=False
        valid = bool(row[28] and row[25]!=4)
        vx = np.float32(np.clip(row[22],-2.5,2.5)) if row[25]==3 and row[26] and row[27]>=500 else np.float32(0)
        yaw = np.float32(np.clip(row[23],-2,2)) if vx != 0 or (row[25]==3 and row[26] and row[27]>=500) else np.float32(0)
        height = np.float32(np.clip(row[24]+np.float32(.06-.02352),.22,.30))
        q = np.array([row[6],row[7],0,row[8],row[9],0],np.float32)[None]
        qnom = np.array([nominal[0],nominal[1],0,nominal[2],nominal[3],0],np.float32)[None]
        env = NS(env=NS(sim=NS(data=NS(qpos=torch.from_numpy(q)))), action=NS(_motor_q_adr=list(range(6)),nominal_positions=torch.from_numpy(qnom),raw_action=torch.from_numpy(row[None,16:22])),angular=torch.from_numpy(row[None,:3]), gravity=torch.from_numpy(row[None,3:6]),commands=torch.tensor([[vx,yaw,height]]),velocity=torch.from_numpy(row[None,10:16][:,[0,1,4,2,3,5]]),cfg=NS(normalization=NS(angular_velocity=.25,command=[2.,.25,5.],joint_velocity=.05)))
        env.device = 'cpu'
        frame = FudanEnv.frame(env).numpy()[0] if valid else np.zeros(25,np.float32)
        if valid and not np.isfinite(frame).all(): valid=False
        if valid:
            history = np.concatenate([history[25:],frame]) if initialized else np.tile(frame,5)
            initialized=True
        else: initialized=False; vx=np.float32(0)
        obs_reference.append(np.r_[np.clip(frame,-100,100),history,vx,float(valid)])
    obs_reference=np.array(obs_reference,np.float32)
    obs_reference.tofile(args.out/'sensor-reference.bin')
    torque_inputs=rng.uniform(-1,1,(256,16)).astype(np.float32)
    torque_inputs[:,:6]*=100; torque_inputs[:,10:]*=50
    torque_inputs.tofile(args.out/'torque-inputs.bin')
    order=[0,1,3,4,2,5]
    ref=np.array([-.664720554,.667594582,.664720554,-.667594582],np.float32)
    targets=np.c_[np.clip(nominal+.5*torque_inputs[:,order[:4]],ref-3.14,ref+3.14),np.clip(torque_inputs[:,order[4:]]*10,-150,150)]
    torques=np.c_[np.clip(60*(targets[:,:4]-torque_inputs[:,6:10])-2*torque_inputs[:,10:14],-40,40),np.clip(.1*(targets[:,4:]-torque_inputs[:,14:]),-3.9,3.9)]
    torque_reference=np.c_[targets,torques,torques*np.array([1,1,-1,-1,1,-1],np.float32)]
    # Use the actual big-machine RL initializer, including original physical signs and gains.
    config_text=(args.h7_root/'imcalib/config/machine_config.c').read_text()
    start=config_text.index('{',config_text.index('.rl ='))
    depth=1; end=start+1
    while depth:
        depth += (config_text[end]=='{')-(config_text[end]=='}'); end+=1
    actuation=(args.h7_root/'imcalib/task/task_actuation.c').read_text()
    match=re.search(r'uint8_t Robot_Control_Enable_Allowed\(void\)\s*\{',actuation)
    assert match
    depth=1; gate_end=match.end()
    while depth:
        depth+=(actuation[gate_end]=='{')-(actuation[gate_end]=='}'); gate_end+=1
    permission=actuation[match.start():gate_end]
    gate=re.search(r'if \(ctrl_strategy == CTRL_STRATEGY_RL && \(rl_control\.recovery\.phase != RL_RECOVERY_NORMAL[^\n]+',actuation)
    assert gate
    harness=HARNESS.replace('CONFIG_INITIALIZER',config_text[start:end]).replace('PERMISSION_FUNCTION',permission).replace('OUTPUT_GATE',gate.group())
    (args.out/'check.c').write_text(harness)
    (args.out/'main.h').write_text('#define __HAL_RCC_CRC_CLK_ENABLE() ((void)0)\n')
    (args.out/'mono_ns.h').write_text('#include <stdint.h>\nuint64_t Mono_Ns_Get(void);\n')
    generated=args.h7_root/'X-CUBE-AI/App'
    sources=[args.out/'check.c',args.h7_root/'imcalib/Algorithm/rl_policy.c',args.h7_root/'imcalib/Algorithm/rl_observation.c',args.h7_root/'imcalib/Algorithm/rl_torque.c']+[generated/x for x in ('networkzn1.c','networkzn1_data.c','networkzn1_data_params.c')]
    command=[os.environ['HOST_CC'],'-O2','-std=c11']+['-I'+str(p) for p in (args.out,args.h7_root/'imcalib/Algorithm',args.h7_root/'imcalib/common',args.h7_root/'imcalib/config',generated,args.st_host/'include')]+[str(p) for p in sources]+['-Wl,--start-group']+[str(args.st_host/'lib/static'/x) for x in ('libruntime.a','libst_cmsis_nn.a','libcmsis-nn.a')]+['-Wl,--end-group','-lm','-o',str(args.out/'check')]
    (args.out/'build-command.json').write_text(json.dumps(command,indent=2))
    run=subprocess.run(command,capture_output=True,text=True); (args.out/'build.log').write_text(run.stdout+run.stderr); assert run.returncode==0,run.stderr
    names=['inputs.bin','c-raw.bin','c-clipped.bin','sensor-inputs.bin','sensor-actual.bin','torque-inputs.bin','torque-actual.bin']
    run=subprocess.run([str(args.out/'check')]+[str(args.out/x) for x in names],capture_output=True,text=True)
    (args.out/'run.log').write_text(run.stdout+run.stderr); assert run.returncode==0,run.stdout+run.stderr
    actual=np.fromfile(args.out/'c-raw.bin',np.float32).reshape(256,6)
    raw_error=float(np.abs(actual-reference).max()); assert np.isfinite(actual).all() and raw_error<=1e-4,raw_error
    clipped=np.fromfile(args.out/'c-clipped.bin',np.float32).reshape(256,6)
    assert np.allclose(clipped,np.clip(actual,-100,100),atol=1e-5,rtol=0)
    sensor_actual=np.fromfile(args.out/'sensor-actual.bin',np.float32).reshape(80,152)
    assert np.array_equal(np.isnan(sensor_actual),np.isnan(obs_reference))
    sensor_error=float(np.nanmax(np.abs(sensor_actual-obs_reference)))
    assert sensor_error<=2e-5,sensor_error
    torque_actual=np.fromfile(args.out/'torque-actual.bin',np.float32).reshape(256,18)
    torque_error=float(np.abs(torque_actual-torque_reference).max()); assert torque_error<=2e-5,torque_error
    actuation=(args.h7_root/'imcalib/task/task_actuation.c').read_text()
    assert 'recovery |= (uint8_t)(machine->rl.configured' not in actuation
    assert 'rl_control.recovery.phase != RL_RECOVERY_NORMAL || !rl_control.recovery.has_stood' in actuation
    report=dict(ok=True,scope='CPU host: actual generated C, SDK wrapper, FIFO, training frame, PD and extracted output permission; no board execution or motion acceptance',checkpoint_sha256=sha(args.checkpoint),onnx_sha256=sha(args.onnx),samples=256,onnx_pytorch_max_abs=onnx_error,c_onnx_max_abs=raw_error,sensor_frames=80,sensor_training_frame_max_abs=sensor_error,torque_cases=256,torque_max_abs=torque_error,recovery_c_assertions=True,walk_only_output_gates_actual_c=True,source_sha256={str(p):sha(p) for p in sources+[args.h7_root/'imcalib/config/machine_config.c',args.h7_root/'imcalib/task/task_policy.c',args.h7_root/'imcalib/task/task_actuation.c']})
    (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
