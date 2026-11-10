"""Check actual C target/state packing and Python pairing; no USB or motor access."""
import argparse
import binascii
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types

from scripts.agent.real2sim.check_h7_output_capture import extract


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    root=Path(__file__).resolve().parents[3];h7=root/'deployment/H7_RL_ros'
    code=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "output_capture.h"
#include "joint_capture.h"
#define CTRL_STRATEGY_JOINT_CAPTURE 7
typedef struct {float virtual_leg_length,virtual_leg_angle;uint8_t valid;} leg_output_t;
typedef struct {leg_output_t output;} leg_state_t;
typedef struct {float motor_pos[4],motor_vel[4],joint_torque[4];struct {float len;} rc;leg_state_t left,right;uint8_t motor_valid;} control_frame_t;
static uint64_t latency_frame_captured_ns,latency_source_rx_ns[8];
static float leg_test_target[2];static uint8_t leg_test_running,leg_test_blocked,leg_test_recovering;
static joint_capture_t joint_capture;static uint8_t joint_capture_divider;
static struct {float dm_trq_clamp;struct {int8_t motor_sign[6];float motor_zero[4];struct {float p_gains[4],d_gains[4];} torque;} rl;} config;
static __typeof__(config) *machine=&config;
static output_capture_record_t records[3];static unsigned count;
static uint32_t Latency_Age_Us(uint64_t now,uint64_t source){return (uint32_t)((now-source)/1000);}
void Output_Capture_Push(const output_capture_record_t *r){assert(count<3);records[count++]=*r;}
@HELPER@
int main(int argc,char **argv){
 assert(argc==2);control_frame_t f={0};output_capture_record_t c={0};c.strategy=7;c.id=42;c.begin_ns=10000000;c.end_ns=10010000;
 latency_frame_captured_ns=9990000;config.dm_trq_clamp=54;config.rl.torque.p_gains[0]=60;config.rl.torque.d_gains[0]=2;
 joint_capture.running=1;joint_capture.stage=2;joint_capture.axis=3;joint_capture.elapsed_ms=98000;joint_capture.saturated=4;
 for(int i=0;i<4;i++){config.rl.motor_sign[i]=i<2?1:-1;joint_capture.target[i]=.1f*(i+1);f.motor_pos[i]=i+1;f.motor_vel[i]=i+11;f.joint_torque[i]=i+21;latency_source_rx_ns[i+1]=9989000;}
 joint_capture_divider=1;Capture_Leg_State(&f,&c);assert(count==0);
 joint_capture_divider=0;Capture_Leg_State(&f,&c);assert(count==3&&records[2].kind==5&&records[2].id==42);
 FILE *file=fopen(argv[1],"wb");assert(file);assert(fwrite(records,sizeof(records),1,file)==1);assert(fclose(file)==0);return 0;
}
'''.replace('@HELPER@',extract((h7/'imcalib/task/task_actuation.c').read_text(),'Capture_Leg_State'))
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);(temp/'wire.c').write_text(code)
        subprocess.run([os.environ.get('CC','cc'),'-std=gnu11','-Wall','-Werror','-I',str(h7/'imcalib/bsp'),'-I',str(h7/'imcalib/Algorithm'),str(temp/'wire.c'),'-lm','-o',str(temp/'wire')],check=True)
        subprocess.run([str(temp/'wire'),str(temp/'raw')],check=True)
        spec=importlib.util.spec_from_file_location('pace_wire',root/'sim2sim/h7_output_capture.py');decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
        control=decoder.RECORD.pack(1,1,10000000,10010000,42,0,*([0.]*8),0,0,*([0]*4),60,7,0,0,0,0,0)
        body=bytearray((temp/'raw').read_bytes())
        for i in range(3):struct.pack_into('<I',body,88*i,2+i)
        def packet(payload,count):
            raw=decoder.HEADER.pack(b'H7OUT001',1,88,count,0,1,count,0,count,count,0)+payload
            return raw+struct.pack('<H',binascii.crc_hqx(raw,0xffff))
        raw=packet(control+body,4)
        fake=types.ModuleType('real2sim_huananhu');fake.load_session=lambda _:({'stream_sha256':'synthetic','model':{},'machine_config':{}},None,None)
        fake.iter_session_events=lambda _:iter([{'type':'connected'},{'type':'data','t_ns':1,'data_hex':raw.hex()},{'type':'stop'}]);sys.modules['real2sim_huananhu']=fake
        report=decoder.analyze_session(temp,temp/'decoded')
        assert report['complete_observed_span'] and report['joint_target_records_match_state']
        rows=list(csv.DictReader((temp/'decoded/outputs.csv').open()));r=rows[-1]
        assert r['kind']=='joint_target' and [round(float(r[f'joint_{i}_target_firmware_rad']),3) for i in range(4)]==[.1,.2,-.3,-.4]
        assert (float(r['joint_kp']),float(r['joint_kd']),float(r['joint_torque_limit_nm']))==(60,2,40)
        assert (int(r['joint_elapsed_ms']),int(r['joint_stage']),int(r['joint_axis']))==(98000,2,3)
        task_source=(h7/'imcalib/task/task_actuation.c').read_text()
        profile=16 if 'OUTPUT_CAPTURE_JOINT_RIGHT_LENGTH |' in task_source else (8 if 'OUTPUT_CAPTURE_JOINT_RIGHT_POSE |' in task_source else 4)
        assert int(r['flags_or_hal']) & 28 == profile
        raw=packet(control+body[:176],3)
        missing=decoder.analyze_session(temp,temp/'missing')
        assert not missing['complete_observed_span'] and not missing['joint_target_records_match_state']
    result=dict(passed=True,actual_C_state_target_packing=True,missing_target_rejected=True,
                firmware_coordinate_mapping_checked=True,pd_phase_pairing_checked=True,hardware_access=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
