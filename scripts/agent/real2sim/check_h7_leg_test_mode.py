"""Exercise the actual H7_ros RC decoder and leg-test solver with host stubs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.agent.real2sim.check_h7_output_capture import extract

ROOT = Path(__file__).resolve().parents[3]
H7 = ROOT / 'deployment/H7_RL_ros'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rc = (H7 / 'imcalib/task/rc_command.c').read_text()
    act = (H7 / 'imcalib/task/task_actuation.c').read_text()
    harness = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define DR16_SW_UP 1
#define DR16_SW_DOWN 2
#define DR16_SW_MID 3
#define DR16_CH_LIMIT 660
#define RC_DEADBAND_LEN 10
#define RC_DEADBAND_VEL 10
#define RC_DEADBAND_YAW 10
#define CTRL_STRATEGY_RL 0
#define CTRL_STRATEGY_LQR 1
#define CTRL_STRATEGY_DISABLE 2
#define CTRL_STRATEGY_COMPENSATION 4
#define CTRL_STRATEGY_WHEEL_BENCH 5
#define CTRL_STRATEGY_LEG_TEST 6
#define CTRL_STRATEGY_JOINT_CAPTURE 7
#define GAS_SPRING_COMP_ENABLE 1
#define MACHINE_DEFAULT 0
#define MACHINE_ID_BIG_WHEELLEG 0
#define MACHINE_RL_HEIGHT_MIN .2f
#define MACHINE_RL_HEIGHT_MAX .3f
#define MACHINE_RL_LEG_TO_HEIGHT .03f
#define MACHINE_LQR_DT .001f
#define DM_MOTOR_NUM 4
#define FAULT_NONE 0
typedef struct {int16_t ch1,ch2,wheel;uint8_t s1,s2,online;} dr16_t;
typedef struct {float vel,yaw,len,leg_test_rate;int32_t leg_test_steps;uint8_t leg_test_arm,s1,s2,online;struct {int strategy;uint8_t rc_enable,engage;} mode;} rc_command_t;
typedef struct {float kp,kd;} pid_config_t;
typedef struct {float virtual_leg_length,d_virtual_leg_length;uint8_t valid,force_valid;} leg_output_t;
typedef struct {leg_output_t output;} leg_state_t;
typedef struct {float dm[4],dji[2];uint8_t valid;} torque_output_t;
typedef struct {struct {float vel_max,yaw_max,len_init,len_rate;} rc;struct {uint8_t configured;} rl;struct {pid_config_t leg_len[2];} lqr;float leg_len_min,leg_len_max,dm_trq_clamp;uint8_t lqr_configured;void *spring,*gravity;} machine_t;
static machine_t config={.rc={3.f,5.f,.15f,.2f},.rl={1},.lqr={{{2300.f,25000.f},{2300.f,25000.f}}},.leg_len_min=.14f,.leg_len_max=.33f,.dm_trq_clamp=54.f,.lqr_configured=1};
static machine_t *machine=&config;
static struct {float len_min,len_max;} gain={.15f,.31f};
static struct {uint8_t reserved;} wheel_bench;
static struct {leg_state_t left,right;struct {uint8_t online;float leg_test_rate;int32_t leg_test_steps;} rc;uint8_t drive;} control_frame;
static uint32_t ctrl_fault;
static float leg_test_target[2],leg_test_goal[2];static int32_t leg_test_applied_steps;static uint8_t leg_test_running,leg_test_blocked,leg_test_recovering;
static uint16_t leg_test_wait_ticks;
static int16_t DR16_Deadline(int16_t v,int d) {return abs(v)<=d?0:v;}
static void *LQR_Gain_Info_raw(void) {return &gain;}
#define LQR_Gain_Info() ((__typeof__(&gain)) LQR_Gain_Info_raw())
static float clampf(float x,float lo,float hi) {return fminf(fmaxf(x,lo),hi);}
static void Torque_Output_Clear(torque_output_t *t) {memset(t,0,sizeof(*t));}
static int force_nan;
static uint8_t Leg_Force_Map_Forward(const leg_state_t *leg,float force,float swing,float out[2]) {
 assert(leg->output.force_valid && swing==0.f);out[0]=force_nan?NAN:force*.2f;out[1]=force*.3f;return 1;}
@AXIS@
@RC@
@SOLVER@
static void step(rc_command_t *cmd, dr16_t *r, int n) {for(int i=0;i<n;i++) Rc_Command_Update(cmd,r,.001f);}
static void reset_solver(float left,float right) {
 leg_test_running=leg_test_blocked=leg_test_recovering=0;leg_test_wait_ticks=0;leg_test_applied_steps=0;
 memset(leg_test_target,0,sizeof(leg_test_target));memset(leg_test_goal,0,sizeof(leg_test_goal));
 control_frame.drive=control_frame.rc.online=1;control_frame.rc.leg_test_rate=0;control_frame.rc.leg_test_steps=0;ctrl_fault=0;force_nan=0;
 control_frame.left.output=(leg_output_t){left,0,1,1};control_frame.right.output=(leg_output_t){right,0,1,1};
}
static void run_solver(torque_output_t *torque,int n,int follow) {
 for(int i=0;i<n;i++) {
  Torque_Output_Clear(torque);solve_leg_test(torque);
  if(torque->valid){assert(torque->dji[0]==0 && torque->dji[1]==0);for(int j=0;j<4;j++)assert(fabsf(torque->dm[j])<=40.f);}
  if(follow){control_frame.left.output.virtual_leg_length=leg_test_target[0];control_frame.right.output.virtual_leg_length=leg_test_target[1];}
 }
}
int main(void) {
 rc_command_t cmd={0};dr16_t r={.s1=DR16_SW_MID,.s2=DR16_SW_DOWN,.online=1};
 step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_LQR && cmd.mode.rc_enable && !cmd.mode.engage);
 for(int left=1;left<=3;left++){for(int right=1;right<=3;right++){r.s1=left;r.s2=right;step(&cmd,&r,1);assert((cmd.mode.strategy==CTRL_STRATEGY_LQR)==(left==DR16_SW_MID && right!=DR16_SW_UP));assert(cmd.mode.strategy!=CTRL_STRATEGY_JOINT_CAPTURE);}}
 r.s1=DR16_SW_DOWN;r.s2=DR16_SW_MID;r.wheel=0;step(&cmd,&r,500);assert(cmd.leg_test_arm==1);
 r.s2=DR16_SW_DOWN;r.s1=DR16_SW_MID;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_LQR && !cmd.mode.engage);
 r.s1=DR16_SW_DOWN;r.s2=DR16_SW_MID;step(&cmd,&r,1100);r.s2=DR16_SW_DOWN;step(&cmd,&r,5100);r.s1=DR16_SW_MID;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_LQR && !cmd.mode.engage);
 r.s1=DR16_SW_DOWN;r.s2=DR16_SW_MID;r.wheel=660;step(&cmd,&r,1100);assert(cmd.leg_test_arm==0 && cmd.leg_test_steps==0);
 r.wheel=0;step(&cmd,&r,1100);assert(cmd.leg_test_arm==2);step(&cmd,&r,10000);assert(cmd.leg_test_arm==2);
 r.s2=DR16_SW_DOWN;step(&cmd,&r,1);r.s1=DR16_SW_MID;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_LEG_TEST && cmd.leg_test_steps==0);
 step(&cmd,&r,110);r.wheel=660;step(&cmd,&r,1);assert(cmd.leg_test_steps==1);
 step(&cmd,&r,3000);assert(cmd.leg_test_steps==1); /* 按住不连跳。 */
 r.wheel=-660;step(&cmd,&r,1000);assert(cmd.leg_test_steps==1); /* 未回中直接反向不跳。 */
 r.wheel=0;step(&cmd,&r,50);r.wheel=-660;step(&cmd,&r,1);assert(cmd.leg_test_steps==1);
 r.wheel=0;step(&cmd,&r,110);r.wheel=-660;step(&cmd,&r,1);assert(cmd.leg_test_steps==0);
 step(&cmd,&r,3000);assert(cmd.leg_test_steps==0);
 r.s2=DR16_SW_MID;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_DISABLE);
 r.s1=DR16_SW_UP;r.s2=DR16_SW_UP;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_DISABLE);
 r.s1=DR16_SW_DOWN;r.s2=DR16_SW_DOWN;step(&cmd,&r,1);r.s1=DR16_SW_MID;r.s2=DR16_SW_MID;step(&cmd,&r,1);assert(cmd.mode.strategy==CTRL_STRATEGY_LQR && cmd.mode.engage);
 r.online=0;step(&cmd,&r,1);assert(!cmd.mode.rc_enable && cmd.leg_test_steps==0);
 torque_output_t torque={0};reset_solver(.2f,.24f);control_frame.drive=0;
 run_solver(&torque,1000,0);assert(leg_test_blocked && !torque.valid);control_frame.drive=1;run_solver(&torque,1,0);assert(!torque.valid);
 reset_solver(.2f,.24f);control_frame.rc.leg_test_steps=7;run_solver(&torque,1,0);
 assert(torque.valid && torque.dm[0]==0 && leg_test_target[0]==.2f && leg_test_goal[1]==.24f);
 control_frame.rc.leg_test_rate=.05f;run_solver(&torque,1000,0);assert(leg_test_target[0]==.2f); /* 不重放接管前步数，也不再按速率累加。 */
 control_frame.rc.leg_test_steps=8;run_solver(&torque,1,0);
 assert(fabsf(leg_test_goal[0]-.21f)<1e-7f && leg_test_target[0]>.2f && leg_test_target[0]<.200051f);
 run_solver(&torque,400,0);assert(fabsf(leg_test_target[0]-.21f)<1e-7f && fabsf(leg_test_target[1]-.25f)<1e-7f);
 run_solver(&torque,3000,0);assert(fabsf(leg_test_goal[0]-.21f)<1e-7f && fabsf(leg_test_target[0]-.21f)<1e-7f);
 control_frame.rc.leg_test_steps=7;control_frame.rc.leg_test_rate=-.05f;run_solver(&torque,400,0);assert(fabsf(leg_test_target[0]-.2f)<1e-7f);
 control_frame.rc.leg_test_steps=1000;run_solver(&torque,4000,0);assert(leg_test_goal[0]==.33f && leg_test_target[1]==.33f);
 control_frame.rc.leg_test_steps=-1000;run_solver(&torque,5000,0);assert(leg_test_goal[0]==.14f && leg_test_target[1]==.14f);
 control_frame.drive=0;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);control_frame.drive=1;run_solver(&torque,1,0);assert(!torque.valid);
 reset_solver(.1f,.2f);run_solver(&torque,1,0);assert(leg_test_target[0]==.1f);run_solver(&torque,1,0);assert(leg_test_target[0]==.14f && torque.dm[0]>0);
 reset_solver(.33109847f,.3397248f);run_solver(&torque,1,0);assert(leg_test_recovering && torque.dm[0]==0);
 control_frame.rc.leg_test_steps=1;control_frame.rc.leg_test_rate=.05f;run_solver(&torque,100,0);assert(leg_test_goal[0]==.33109847f && torque.dm[2]==0);
 control_frame.rc.leg_test_steps=0;control_frame.rc.leg_test_rate=-.05f;run_solver(&torque,1,0);
 assert(fabsf(leg_test_goal[0]-.32109847f)<1e-7f && leg_test_target[0]>.3310f && torque.dm[0]<0);
 run_solver(&torque,250,1);assert(leg_test_recovering && leg_test_target[1]<=.33f);
 control_frame.rc.leg_test_rate=0;run_solver(&torque,1,1);assert(!leg_test_recovering);
 control_frame.rc.leg_test_steps=1;run_solver(&torque,400,0);assert(leg_test_target[0]<=.33f && leg_test_target[1]<=.33f);
 leg_test_goal[0]=leg_test_goal[1]=leg_test_target[0]=leg_test_target[1]=.317096412f;
 control_frame.left.output=(leg_output_t){.329116136f,0,1,1};control_frame.right.output=(leg_output_t){.330134511f,0,1,1};
 run_solver(&torque,1,0);assert(torque.valid && !leg_test_blocked && torque.dm[2]<0 && torque.dm[3]<0);
 control_frame.right.output.d_virtual_leg_length=-2.f;run_solver(&torque,1,0);assert(torque.dm[2]==0 && torque.dm[3]==0 && !leg_test_blocked);
 control_frame.right.output.d_virtual_leg_length=0;
 control_frame.left.output=(leg_output_t){.139f,0,1,1};control_frame.right.output=(leg_output_t){.3f,0,1,1};
 leg_test_goal[0]=leg_test_target[0]=.14f;run_solver(&torque,1,0);assert(torque.dm[0]>0 && torque.dm[1]>0);
 control_frame.left.output.d_virtual_leg_length=2.f;run_solver(&torque,1,0);assert(torque.dm[0]==0 && torque.dm[1]==0);
 control_frame.left.output.d_virtual_leg_length=0;control_frame.rc.leg_test_steps-=10;run_solver(&torque,4000,0);assert(leg_test_goal[0]==.14f && torque.dm[0]>=0);
 control_frame.left.output.force_valid=0;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);
 control_frame.left.output.force_valid=1;leg_test_blocked=0;control_frame.left.output.virtual_leg_length=NAN;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);
 control_frame.left.output.virtual_leg_length=.2f;leg_test_blocked=0;control_frame.rc.online=0;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);
 control_frame.rc.online=1;leg_test_blocked=0;ctrl_fault=1;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);
 reset_solver(.2f,.2f);run_solver(&torque,1,0);force_nan=1;run_solver(&torque,1,0);assert(leg_test_blocked && !torque.valid);
 return 0;
}
'''.replace('@AXIS@', extract(rc, 'Rc_Axis')).replace('@RC@', extract(rc, 'Rc_Command_Update')).replace('@SOLVER@', extract(act, 'solve_leg_test'))
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'check.c'; source.write_text(harness)
        exe = Path(tmp) / 'check'
        subprocess.run([os.environ.get('CC','cc'),'-std=gnu11','-Wall','-Werror',str(source),'-lm','-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
        dispatch = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define DM_MOTOR_NUM 4
#define DJI_MOTOR_NUM 2
#define DJI_MOTOR_WHEEL_LFT 0
#define DJI_MOTOR_WHEEL_RGT 1
#define CTRL_STRATEGY_RL 0
#define CTRL_STRATEGY_COMPENSATION 4
#define CTRL_STRATEGY_WHEEL_BENCH 5
#define CTRL_STRATEGY_LEG_TEST 6
#define CTRL_STRATEGY_JOINT_CAPTURE 7
#define RL_RECOVERY_NORMAL 3
#define RL_RECOVERY_RECOVER 2
#define FAULT_NONE 0
#define HAL_OK 0
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
typedef struct {float dm[4],dji[2];uint8_t valid;} torque_output_t;
typedef struct {uint8_t bench;} compensation_debug_t;
typedef struct {uint32_t tick_ms,policy_sequence;uint16_t flags;} mirror_output_snapshot_t;
static uint64_t output_dispatch_started_ns;
static struct {uint32_t sequence;} mirror_output_snapshot;
static struct {uint32_t reset_count;} action_state;
static struct {uint8_t drive;} control_frame;
static struct {uint8_t reserved;} wheel_bench;
static struct {struct {uint8_t phase,has_stood;} recovery;} rl_control;
static struct {struct {int strategy;} mode;} rc_command;
static compensation_debug_t pending_compensation,compensation_debug;
static uint8_t ctrl_strategy,torque_output_enabled=1,leg_test_running=1,leg_test_blocked;
static struct {uint8_t running,blocked;} joint_capture;
static uint32_t ctrl_fault;static float rl_output_dm_cmd_nm[4],rl_output_wheel_cmd_nm[2];
static uint8_t output_debug_dm_sent,output_debug_dji_sent,allow=1;
static unsigned zero_calls,drive_calls;static float sent_wheels[2],sent_dm[4];
static uint64_t Mono_Ns_Get(void){return 10000;}
static void Latency_Dispatch_Begin(uint32_t a,uint32_t b,uint32_t c){(void)a;(void)b;(void)c;}
static uint32_t HAL_GetTick(void){return 0;}
static uint8_t Wheel_Bench_Output_Valid(uint32_t a){(void)a;return 0;}
static uint8_t Robot_Control_Enable_Allowed(void){return allow;}
static int Dm_Send_Zero(void){zero_calls++;memset(sent_dm,0,sizeof(sent_dm));return 0;}
static int Dji_All_Stop(void){sent_wheels[0]=sent_wheels[1]=0;return 0;}
static int Dm_Send_Torque(float *a){drive_calls++;memcpy(sent_dm,a,sizeof(sent_dm));return 0;}
static int Dji_Send_Wheel_Torque(float a,float b){sent_wheels[0]=a;sent_wheels[1]=b;return 0;}
static void Mirror_Output_Publish(mirror_output_snapshot_t *d,const torque_output_t *a){(void)d;(void)a;}
@DISPATCH@
int main(void){
 torque_output_t t={{4,5,6,7},{0,0},1};mirror_output_snapshot_t d={0};
 ctrl_strategy=CTRL_STRATEGY_LEG_TEST;rc_command.mode.strategy=ctrl_strategy;control_frame.drive=1;
 output_dispatch(&t,&d);assert(drive_calls==1 && sent_dm[0]==4 && sent_wheels[0]==0 && sent_wheels[1]==0 && !leg_test_blocked);
 ctrl_fault=1;output_dispatch(&t,&d);assert(zero_calls==1 && leg_test_blocked && sent_dm[0]==0 && sent_wheels[0]==0 && sent_wheels[1]==0);
 ctrl_fault=0;allow=0;leg_test_blocked=0;output_dispatch(&t,&d);assert(zero_calls==2 && leg_test_blocked);
 allow=1;ctrl_strategy=CTRL_STRATEGY_JOINT_CAPTURE;rc_command.mode.strategy=ctrl_strategy;joint_capture.running=1;
 output_dispatch(&t,&d);assert(sent_dm[0]==4 && !joint_capture.blocked);
 ctrl_fault=1;output_dispatch(&t,&d);assert(sent_dm[0]==0 && joint_capture.blocked);
 return 0;
}
'''.replace('@DISPATCH@', extract(act, 'output_dispatch'))
        (Path(tmp) / 'dispatch.c').write_text(dispatch)
        subprocess.run([os.environ.get('CC','cc'),'-std=gnu11','-Wall','-Werror',str(Path(tmp)/'dispatch.c'),'-lm','-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump({'passed':True,'step_mm':10,'hold_does_not_repeat':True,'center_rearm_ms':100,'slew_limit_m_s':.05,'scope':'actual RC and leg solver with synthetic state/map stubs; no full scheduler, hardware or physics'},f,indent=2)
    print('Actual RC arming and leg solver checks passed')


if __name__ == '__main__':main()
