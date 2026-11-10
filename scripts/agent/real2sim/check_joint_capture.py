"""Exercise production joint-capture C, geometry guard and actual task wrapper; no hardware."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from scripts.agent.real2sim.check_h7_output_capture import extract
from scripts.agent.real2sim.precheck_pace_capture import starting_pose


class State(ctypes.Structure):
    _fields_=[('center',ctypes.c_float*4),('target',ctypes.c_float*4),('start_ms',ctypes.c_uint32),
              ('elapsed_ms',ctypes.c_uint32),('running',ctypes.c_uint8),('blocked',ctypes.c_uint8),
              ('stage',ctypes.c_uint8),('axis',ctypes.c_uint8),('saturated',ctypes.c_uint16)]


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    root=Path(__file__).resolve().parents[3];h7=root/'deployment/H7_RL_ros';algorithm=h7/'imcalib/Algorithm'
    compiler=os.environ.get('CC','cc');floats=ctypes.c_float*4;signs=ctypes.c_int8*6
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);library=temp/'capture.so'
        sources=[str(algorithm/n) for n in ('joint_capture.c','joint_chirp.c')]
        subprocess.run([compiler,'-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC',*sources,'-lm','-o',str(library)],check=True)
        lib=ctypes.CDLL(str(library));update=lib.Joint_Capture_Update
        update.argtypes=[ctypes.POINTER(State),ctypes.c_uint32,ctypes.c_uint8,*([ctypes.POINTER(ctypes.c_float)]*3),
            ctypes.POINTER(ctypes.c_int8),*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        update.restype=ctypes.c_uint8
        guard=lib.Joint_Capture_Geometry;guard.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_float,ctypes.c_float];guard.restype=ctypes.c_uint8
        pose,fw=starting_pose(.265,h7/'imcalib/config/machine_config.c')
        state=State();q=floats(*fw);v=floats(0,0,0,0);previous=floats(5,-5,6,-6)
        sign=signs(1,1,-1,-1,1,-1);zero=floats(0,0,0,0);kp=floats(60,60,60,60);kd=floats(2,2,2,2);out=floats()
        def run(time,valid=1):return update(ctypes.byref(state),time,valid,q,v,previous,sign,zero,kp,kd,40.,out)
        assert run(1000) and np.max(abs(np.array(out[:])-previous[:]))<1e-5
        assert run(8000) and state.elapsed_ms==7000 and state.stage==2
        v[0]=100.;assert run(8002) and state.saturated&1 and out[0]==-40. # 没有速度停机门槛。
        v[0]=0.;assert not run(8004,0) and state.blocked and out[:]==[0.]*4
        assert not run(8006,1) and out[:]==[0.]*4 # 故障后不能自动续跑。
        state=State();assert run(1000);assert run(121000) and state.stage==3
        assert run(400000) and state.stage==3
        state=State();q[0]=float('nan');assert not run(0) and state.blocked
        guard_count=0
        for length in np.linspace(.15,.345,25):
            pose,fw=starting_pose(float(length),h7/'imcalib/config/machine_config.c')
            for shift in (-.08,0.,.08):
                changed=fw+shift
                for side in (0,1):assert pose._solve_leg(side,pose.sign[2*side:2*side+2]*(changed[2*side:2*side+2]-pose.zero[2*side:2*side+2]))
                expected=.14<=length<=.33
                for name in ('lf1','rf1'):
                    joint=pose.model.joint(name);angle=pose.data.qpos[joint.qposadr[0]]
                    expected &= joint.range[0]<=angle<=joint.range[1]
                assert bool(guard(floats(*changed),.21,.25))==bool(expected),(length,shift)
                guard_count+=1
        assert not guard(floats(float('nan'),0,0,0),.21,.25)
        # 用实际任务包装函数验证5ms反馈龄、故障及2ms保持，硬件仅以数据替身替代。
        wrapper=extract((h7/'imcalib/task/task_actuation.c').read_text(),'solve_joint_capture')
        harness=r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "joint_capture.h"
#define FAULT_NONE 0
#define MACHINE_RL_CTRL_DIV 2
typedef struct {float dm[4],dji[2];uint8_t valid;} torque_output_t;
static joint_capture_t joint_capture;static uint8_t joint_capture_divider;static torque_output_t joint_capture_torque;
static struct {uint8_t drive,joint_ready,motor_valid;struct {uint8_t online;} rc;float motor_pos[4],motor_vel[4];} control_frame;
static struct {float leg_lu,leg_lg,dm_trq_clamp;struct {int8_t motor_sign[6];float motor_zero[4];struct {float p_gains[4],d_gains[4];} torque;} rl;} config;
static __typeof__(config) *machine=&config;
static uint64_t now,latency_source_rx_ns[8];static float rl_output_dm_cmd_nm[4];static uint32_t ctrl_fault;
static uint64_t Mono_Ns_Get(void){return now;}
static void Torque_Output_Clear(torque_output_t *t){memset(t,0,sizeof(*t));}
@WRAPPER@
static void fresh(void){for(int i=0;i<4;i++)latency_source_rx_ns[i+1]=now-1000000;}
int main(void){
 torque_output_t t={0};now=1000000000;config.leg_lu=.21f;config.leg_lg=.25f;config.dm_trq_clamp=54.f;
 control_frame.rc.online=control_frame.motor_valid=control_frame.joint_ready=1;
 for(int i=0;i<4;i++){config.rl.motor_sign[i]=i<2?1:-1;config.rl.torque.p_gains[i]=60;config.rl.torque.d_gains[i]=2;control_frame.motor_pos[i]=i%2?.48435023f:-.48435023f;}
 fresh();solve_joint_capture(&t);assert(!t.valid&&!joint_capture.blocked); /* 等驱动使能不提前锁死。 */
 control_frame.drive=1;solve_joint_capture(&t);assert(t.valid&&joint_capture.running&&t.dji[0]==0&&t.dji[1]==0);
 now+=1000000;fresh();solve_joint_capture(&t);assert(t.valid&&joint_capture_divider==1);
 now+=1000000;fresh();solve_joint_capture(&t);assert(t.valid&&joint_capture.elapsed_ms==2);
 now+=6000000;solve_joint_capture(&t);assert(!t.valid&&joint_capture.blocked);
 fresh();solve_joint_capture(&t);assert(!t.valid);
 memset(&joint_capture,0,sizeof(joint_capture));ctrl_fault=1;solve_joint_capture(&t);assert(!t.valid&&joint_capture.blocked);
 return 0;
}
'''.replace('@WRAPPER@',wrapper)
        (temp/'wrapper.c').write_text(harness)
        subprocess.run([compiler,'-std=gnu11','-Wall','-Werror','-I',str(algorithm),str(temp/'wrapper.c'),*sources,'-lm','-o',str(temp/'wrapper')],check=True)
        subprocess.run([str(temp/'wrapper')],check=True)
    result=dict(passed=True,calibrated_geometry_cases=guard_count,bumpless_torque_takeover=True,
        invalid_latches=True,feedback_age_and_fault_wrapper_checked=True,pd_2ms_hold_checked=True,
        no_actual_speed_stop_gate=True,source='actual C controller, task wrapper and shared XML; no hardware',new_physics_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
