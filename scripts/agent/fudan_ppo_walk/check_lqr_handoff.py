"""Run actual RC, recovery-preparation and action-clear C on the host; no hardware."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from scripts.agent.real2sim.check_h7_output_capture import extract


HARNESS = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rc_command.h"
#include "rl_observation.h"
#include "rl_torque.h"
#define GAS_SPRING_COMP_ENABLE 1u
#define STANDUP_DONE 7u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
#define __get_PRIMASK() 0u
#define __disable_irq() ((void)0)
#define __set_PRIMASK(x) ((void)(x))
static const machine_spring_cfg_t spring;
static const machine_gravity_cfg_t gravity;
static const machine_cfg_t test_machine={.lqr_configured=1,.spring=&spring,.gravity=&gravity,.leg_len_min=.14f,.leg_len_max=.33f,.rc={.vel_max=3,.yaw_max=5,.len_init=.15f,.len_rate=.2f},.rl={.configured=1}};
const machine_cfg_t *const machine=&test_machine;
static struct { uint8_t reserved; } wheel_bench;
static const struct { float len_min,len_max; } gain={.14f,.33f};
#define LQR_Gain_Info() (&gain)
static int16_t DR16_Deadline(int16_t x,unsigned d) { return abs(x)<=(int)d?0:x; }
@AXIS@
@RC@
static struct { uint8_t enabled,phase; } standup_control;
static uint8_t lqr_running;
static uint8_t rl_recovery_wait_ticks;
static struct { uint8_t valid; struct { float force; } leg[2]; } rl_support;
static struct { rl_recovery_state_t recovery; rl_observation_state_t observation; rl_torque_state_t torque_state; } rl_control;
static struct { float a[6]; uint8_t updated,rl_ready; uint32_t reset_count,last_ok_tick; } action_state;
static struct {
    rc_command_t rc;
    struct { float quat[4]; } imu;
    int left,right,compensation;
    uint8_t joint_ready,drive,height_valid,fallen;
    float height,joint_torque[4];
} control_frame;
typedef struct { const void *imu,*leg[2],*compensation; uint8_t joint_ready,permit; float torque[2][2]; } air_input_t;
static float measured_force[2]={65.700435f,65.700435f};
static void Air_Detection_Update(void *unused,const air_input_t *in) {
    (void)unused;rl_support.valid=in->permit;
    for(unsigned i=0;i<2;i++) { rl_support.leg[i].force=measured_force[i]; }
}
@CLEAR@
@RECOVERY@
static ctrl_strategy_t ctrl_strategy;
static void change_strategy(ctrl_strategy_t strategy) {
    @EDGE@
    ctrl_strategy=strategy;
}
static void ticks(unsigned n) { for(unsigned i=0;i<n;i++) { Control_RL_Recovery_Update(); } }
int main(void) {
    rc_command_t cmd={0};dr16_t r={.online=1,.s1=DR16_SW_DOWN,.s2=DR16_SW_MID};
    Rc_Command_Update(&cmd,&r,.001f); assert(cmd.mode.strategy==CTRL_STRATEGY_DISABLE && !cmd.mode.rc_enable);
    r.s1=DR16_SW_MID;Rc_Command_Update(&cmd,&r,.001f);
    assert(cmd.mode.strategy==CTRL_STRATEGY_LQR && cmd.mode.rc_enable && cmd.mode.engage);
    r.s1=DR16_SW_UP;Rc_Command_Update(&cmd,&r,.001f);
    assert(cmd.mode.strategy==CTRL_STRATEGY_RL && cmd.mode.rc_enable && cmd.mode.engage);
    r.s1=DR16_SW_MID;Rc_Command_Update(&cmd,&r,.001f);assert(cmd.mode.strategy==CTRL_STRATEGY_LQR);
    r.s1=DR16_SW_DOWN;Rc_Command_Update(&cmd,&r,.001f);assert(!cmd.mode.rc_enable);
    r.online=0;Rc_Command_Update(&cmd,&r,.001f);assert(!cmd.mode.rc_enable && cmd.vel==0 && cmd.yaw==0);
    r.online=1;r.s1=DR16_SW_UP;r.s2=DR16_SW_UP;Rc_Command_Update(&cmd,&r,.001f);assert(cmd.mode.strategy==CTRL_STRATEGY_WHEEL_BENCH);
    wheel_bench.reserved=1;r.s1=DR16_SW_MID;r.s2=DR16_SW_MID;Rc_Command_Update(&cmd,&r,.001f);assert(!cmd.mode.rc_enable);wheel_bench.reserved=0;
    control_frame.rc.online=control_frame.rc.mode.rc_enable=control_frame.rc.mode.engage=1;
    control_frame.rc.mode.strategy=CTRL_STRATEGY_LQR;
    control_frame.drive=control_frame.height_valid=control_frame.joint_ready=1;
    control_frame.height=.26f-MACHINE_RL_LEG_TO_HEIGHT;control_frame.imu.quat[0]=1;
    standup_control.enabled=1;ticks(1100);assert(rl_control.recovery.phase==RL_RECOVERY_IDLE);
    lqr_running=1;standup_control.phase=0;ticks(1100);assert(rl_control.recovery.phase==RL_RECOVERY_IDLE);
    standup_control.phase=STANDUP_DONE;ticks(1100);
    assert(rl_control.recovery.phase==RL_RECOVERY_NORMAL && rl_control.recovery.has_stood && rl_control.recovery.tracking_ms==500);
    control_frame.rc.mode.strategy=CTRL_STRATEGY_RL;
    ticks(1);assert(rl_control.recovery.phase==RL_RECOVERY_NORMAL && rl_control.recovery.has_stood);
    ctrl_strategy=CTRL_STRATEGY_LQR;action_state.a[0]=12;action_state.rl_ready=1;rl_control.observation.history_initialized=1;rl_control.observation.last_action[0]=12;
    unsigned before=action_state.reset_count;change_strategy(CTRL_STRATEGY_RL);
    assert(action_state.reset_count==before+1 && !action_state.rl_ready && !action_state.a[0] && !rl_control.observation.history_initialized && !rl_control.observation.last_action[0]);
    change_strategy(CTRL_STRATEGY_RL);assert(action_state.reset_count==before+1);
    change_strategy(CTRL_STRATEGY_LQR);assert(action_state.reset_count==before+2);
    change_strategy(CTRL_STRATEGY_RL);assert(action_state.reset_count==before+3);
    control_frame.rc.mode.engage=0;ticks(1);assert(rl_control.recovery.phase==RL_RECOVERY_IDLE);
    control_frame.rc.mode.engage=1;control_frame.rc.mode.strategy=CTRL_STRATEGY_LQR;ticks(100);
    assert(rl_control.recovery.phase==RL_RECOVERY_HOLD && !rl_control.recovery.has_stood);
    control_frame.rc.mode.strategy=CTRL_STRATEGY_RL;ticks(1);assert(rl_control.recovery.phase==RL_RECOVERY_HOLD && !rl_control.recovery.has_stood);
    ticks(1100);assert(rl_control.recovery.phase==RL_RECOVERY_NORMAL);
    measured_force[0]=measured_force[1]=0;ticks(220);assert(rl_control.recovery.phase==RL_RECOVERY_FAILED);
    control_frame.rc.mode.strategy=CTRL_STRATEGY_LQR;lqr_running=0;ticks(1);assert(rl_control.recovery.phase==RL_RECOVERY_IDLE);
    lqr_running=1;measured_force[0]=measured_force[1]=65.700435f;control_frame.height=.15f;ticks(2200);
    assert(rl_control.recovery.phase==RL_RECOVERY_RECOVER && !rl_control.recovery.has_stood);
    control_frame.drive=0;ticks(1);assert(rl_control.recovery.phase==RL_RECOVERY_IDLE);
    return 0;
}
'''


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];h7=root/'deployment/H7_RL_ros'
    rc=h7/'imcalib/task/rc_command.c';act=h7/'imcalib/task/task_actuation.c';control=h7/'imcalib/task/robot_control.c'
    text=act.read_text()
    edge=re.search(r'if \(strategy != ctrl_strategy && \(strategy == CTRL_STRATEGY_RL \|\| ctrl_strategy == CTRL_STRATEGY_RL\)\)\s*\{\s*Action_State_Clear\(\);\s*\}',text)
    assert edge
    code=HARNESS.replace('@AXIS@',extract(rc.read_text(),'Rc_Axis')).replace('@RC@',extract(rc.read_text(),'Rc_Command_Update')).replace('@RECOVERY@',extract(text,'Control_RL_Recovery_Update')).replace('@CLEAR@',extract(control.read_text(),'Action_State_Clear')).replace('@EDGE@',edge.group())
    (args.out/'check.c').write_text(code)
    command=[os.environ['CC'],'-O2','-std=gnu11','-Wall','-Werror']+['-I'+str(h7/x) for x in ('imcalib/task/inc','imcalib/common','imcalib/config','imcalib/Algorithm')]+[str(args.out/'check.c'),str(h7/'imcalib/Algorithm/rl_observation.c'),str(h7/'imcalib/Algorithm/rl_torque.c'),'-lm','-o',str(args.out/'check')]
    (args.out/'build-command.json').write_text(json.dumps(command,indent=2))
    r=subprocess.run(command,capture_output=True,text=True);(args.out/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    r=subprocess.run([str(args.out/'check')],capture_output=True,text=True);(args.out/'run.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    result=dict(ok=True,scope='actual extracted RC/recovery/action-clear C with AIR/RTOS substitutes; no board or motion acceptance',lqr_rl_manual_switch=True,stable_lqr_prepares_existing_rl_gate=True,incomplete_preparation_does_not_bypass_gate=True,rl_mode_edges_clear_actions_and_history=True,fault_feedback_and_unsupported_height_preserve_protection=True,joint_capture_rc_entry_disabled=True,source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (rc,act,control,h7/'imcalib/Algorithm/rl_observation.c',Path(__file__))})
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
