"""Check actual leg-capture helper and decoder with synthetic data; no hardware."""
import argparse
import binascii
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types

from scripts.agent.real2sim.check_h7_output_capture import extract

ROOT = Path(__file__).resolve().parents[3]
H7 = ROOT / 'deployment/H7_RL_ros'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        helper = extract((H7 / 'imcalib/task/task_actuation.c').read_text(), 'Capture_Leg_State')
        source = r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "@OUTPUT_HEADER@"
#define CTRL_STRATEGY_JOINT_CAPTURE 7
static uint8_t joint_capture_divider;
static struct {float target[4];uint32_t elapsed_ms;uint8_t running,blocked,stage,axis;uint16_t saturated;} joint_capture;
static struct {float dm_trq_clamp;struct {int8_t motor_sign[6];float motor_zero[4];struct {float p_gains[4],d_gains[4];} torque;} rl;} config;
static __typeof__(config) *machine=&config;
#include <math.h>
typedef struct {float virtual_leg_length,virtual_leg_angle;uint8_t valid;} leg_output_t;
typedef struct {leg_output_t output;} leg_state_t;
typedef struct {float motor_pos[4],motor_vel[4],joint_torque[4];struct {float len;} rc;leg_state_t left,right;uint8_t motor_valid;} control_frame_t;
static uint64_t latency_frame_captured_ns,latency_source_rx_ns[8];
static float leg_test_target[2]={.2f,.3f};static uint8_t leg_test_running=1,leg_test_blocked=0,leg_test_recovering=0;
static output_capture_record_t records[2];static unsigned count;
static uint32_t Latency_Age_Us(uint64_t now,uint64_t source) { return (uint32_t)((now-source)/1000u); }
void Output_Capture_Push(const output_capture_record_t *r) {assert(count<2);records[count++]=*r;}
@HELPER@
int main(int argc,char **argv) {
 assert(argc==2 && sizeof(output_capture_record_t)==88);
 control_frame_t f={0};output_capture_record_t c={0};latency_frame_captured_ns=10000000;c.end_ns=10001000;c.id=42;c.strategy=1;c.flags=32;
 for(int i=0;i<4;i++){latency_source_rx_ns[i+1]=10000000-(i+1)*1000;f.motor_pos[i]=i+1;f.motor_vel[i]=i+11;f.joint_torque[i]=i+21;}
 f.rc.len=.25f;f.left.output.virtual_leg_length=.2f;f.right.output.virtual_leg_length=.3f;f.left.output.virtual_leg_angle=-.1f;f.right.output.virtual_leg_angle=.4f;f.left.output.valid=f.right.output.valid=f.motor_valid=1;
 Capture_Leg_State(&f,&c);assert(count==2 && records[0].id==42 && records[1].id==42);
 FILE *out=fopen(argv[1],"wb");assert(out);assert(fwrite(records,sizeof(records),1,out)==1);assert(fclose(out)==0);return 0;
}
'''.replace('@HELPER@', helper).replace('@OUTPUT_HEADER@', str(H7 / 'imcalib/bsp/output_capture.h'))
        (tmp / 'check.c').write_text(source)
        module_path = ROOT / 'sim2sim/h7_output_capture.py'
        spec = importlib.util.spec_from_file_location('h7_leg_output_capture', module_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        record = module.RECORD
        c_checked = bool(os.environ.get('CC') or shutil.which('cc'))
        if c_checked:
            subprocess.run([os.environ.get('CC','cc'), '-std=gnu11', '-Wall', '-Werror', str(tmp / 'check.c'), '-lm', '-o', str(tmp / 'check')], check=True)
            subprocess.run([str(tmp / 'check'), str(tmp / 'records.bin')], check=True)
            raw = (tmp / 'records.bin').read_bytes()
        else:
            right_angle = struct.unpack('<I', struct.pack('<f', .4))[0]
            state = record.pack(0,3,1000000,1001000,42,0,1.,2.,3.,4.,11.,12.,13.,14.,1,2,3,0,4,0,32,1,0,0,0,0,0)
            geometry = record.pack(0,4,1000000,1001000,42,0,21.,22.,23.,24.,.25,.2,.3,-.1,right_angle,0,0,0,0,0,7,1,0,0,0,0,0)
            raw = state + geometry
        control = record.pack(1, 1, 999000, 1001000, 42, 0, *([0.]*8), 0, 0, *([0]*4), 32, 1, 0, 0, 0, 0, 0)
        # C helper leaves sequence assignment to the actual queue; emulate it here.
        state = bytearray(raw[:88]); geometry = bytearray(raw[88:]); struct.pack_into('<I', state, 0, 2); struct.pack_into('<I', geometry, 0, 3)
        header = module.HEADER.pack(b'H7OUT001', 1, 88, 3, 0, 1, 3, 0, 3, 3, 0)
        packet = header + control + state + geometry
        packet += struct.pack('<H', binascii.crc_hqx(packet, 0xffff))
        meta = {'capture_complete': True, 'stream_sha256': 'synthetic', 'model': {}, 'machine_config': {}}
        fake = types.ModuleType('real2sim_huananhu'); fake.load_session=lambda _: (meta,None,None)
        fake.iter_session_events=lambda _: iter([{'type':'connected'}, {'type':'data','t_ns':1,'data_hex':packet.hex()}, {'type':'stop'}])
        sys.modules['real2sim_huananhu'] = fake
        out = tmp / 'analysis'; report = module.analyze_session(tmp, out)
        import csv
        with (out / 'outputs.csv').open() as f: rows=list(csv.DictReader(f))
        assert report['complete_observed_span'] and report['leg_records_match_control']
        assert [float(rows[1][f'leg_{i}_position_rad']) for i in range(4)] == [1,2,3,4]
        assert [float(rows[1][f'leg_{i}_velocity_rad_s']) for i in range(4)] == [11,12,13,14]
        assert [int(rows[1][f'leg_{i}_rx_age_us']) for i in range(4)] == [1,2,3,4]
        assert [float(rows[2][f'leg_{i}_feedback_torque_nm']) for i in range(4)] == [21,22,23,24]
        assert int(rows[2]['leg_valid_flags'])&128, 'New leg-test records must identify the10mm control version'
        assert abs(float(rows[2]['leg_length_command_m'])-.25)<1e-6 and abs(float(rows[2]['right_leg_angle_rad'])-.4)<1e-6
        if c_checked:
            assert int(rows[2]['leg_valid_flags']) & 64, 'New boundary control version is not recorded'
            assert abs(float(rows[2]['left_leg_test_target_m'])-.2)<1e-6 and abs(float(rows[2]['right_leg_test_target_m'])-.3)<1e-6
        bad_header = module.HEADER.pack(b'H7OUT001',1,88,2,0,1,2,0,2,2,0)
        bad = bad_header + control + state
        bad += struct.pack('<H', binascii.crc_hqx(bad, 0xffff))
        fake.iter_session_events=lambda _: iter([{'type':'connected'}, {'type':'data','t_ns':1,'data_hex':bad.hex()}, {'type':'stop'}])
        missing = module.analyze_session(tmp, tmp / 'missing')
        assert missing['leg_records_present'] and not missing['leg_records_match_control'] and not missing['complete_observed_span']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:json.dump({'decoder_passed':True,'c_helper_checked':c_checked,'scope':'synthetic packet only; no hardware, dynamics, or timing-load validation'}, f, indent=2)
    print('Decoder passed; C helper checked:', c_checked)


if __name__ == '__main__':main()
