#!/usr/bin/env python3
"""Finite host checks of actual capture/driver/USB code, optional two-machine ARM link.

Hardware, clocks, IRQs and USB completion are stubs. No serial port is opened.
"""
from __future__ import annotations
import argparse
import binascii
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
H7=ROOT/'deployment/H7_RL_ros'
sys.path.insert(0,str(ROOT/'sim2sim'))
import h7_output_capture as capture


def extract(text,name):
    start=re.search(r'^\w[\w \*]*\b'+name+r'\([^;]*?\)\s*\{',text,re.M)
    if not start:raise ValueError(name)
    depth=1
    for end in range(start.end(),len(text)):
        depth+=(text[end]=='{')-(text[end]=='}')
        if not depth:return text[start.start():end+1]+'\n'
    raise ValueError('unclosed '+name)


def native(directory):
    stubs=directory/'stubs';stubs.mkdir()
    (stubs/'main.h').write_text('''
#ifndef CAPTURE_MAIN_STUB
#define CAPTURE_MAIN_STUB
#include <stdint.h>
extern uint32_t irq_mask;
#define __get_PRIMASK() irq_mask
#define __disable_irq() (irq_mask=1u)
#define __set_PRIMASK(x) (irq_mask=(x))
#define __DMB() __asm__ __volatile__("" ::: "memory")
#endif
''')
    (stubs/'machine_config.h').write_text('#define MACHINE_DEFAULT 0u\n#define MACHINE_VOFA_PORT 10u\n')
    (stubs/'usart.h').write_text('')
    (stubs/'usbd_cdc_if.h').write_text('''
#include <stdint.h>
#define USBD_OK 0u
#define USBD_FAIL 2u
uint8_t CDC_Transmit_Ready_HS(void);
uint8_t CDC_Transmit_HS(uint8_t *,uint16_t);
''')
    harness=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
uint32_t irq_mask;
#include "@CAPTURE@"
#include "@LATENCY_HEADER@"
uint16_t Latency_Capture_Packet(uint8_t *out) { (void)out; return 0; }
void Latency_Capture_Commit(void) {}
#include "@SENDER@"
static uint8_t configured,busy,fail_submit;
static uint8_t *pending;
static uint16_t pending_size;
static unsigned submissions;
uint8_t CDC_Transmit_Ready_HS(void) { return configured && !busy; }
uint8_t CDC_Transmit_HS(uint8_t *data,uint16_t n) {
    submissions++;
    if(fail_submit) return USBD_FAIL;
    pending=data;pending_size=n;busy=1;return USBD_OK;
}
static uint64_t now_ns=1000000000u;
static uint64_t Mono_Ns_Get(void) { now_ns+=1000;return now_ns; }
typedef unsigned HAL_StatusTypeDef;
typedef struct { unsigned marker; } FDCAN_HandleTypeDef;
static FDCAN_HandleTypeDef bus;
static struct { uint8_t dji_type,dji_bus; float dji_gear_ratio; struct { int out; } dji_sign[2]; } config;
static const __typeof__(config) *machine=&config;
#define DJI_MOTOR_NUM 2u
#define DJI_MOTOR_WHEEL_LFT 0u
#define DJI_MOTOR_WHEEL_RGT 1u
#define DJI_M2006 0u
@CONSTANTS@
typedef struct { uint32_t feedback_id,control_id; } dji_motor_config_t;
static dji_motor_config_t dji_motor_config[2]={{0x202,0x200},{0x201,0x200}};
static volatile int16_t wheel_current[4];
static unsigned driver_status;
static FILE *golden;
static uint8_t last_data[8];
static FDCAN_HandleTypeDef *Can_Bus_Handle(unsigned id) { (void)id;return &bus; }
static HAL_StatusTypeDef Can_Bus_Transmit(FDCAN_HandleTypeDef *h,uint32_t id,uint8_t *data,unsigned n) {
    assert(h==&bus && id==0x200 && n==8);memcpy(last_data,data,8);
    assert(fwrite(data,1,8,golden)==8);assert(fputc(driver_status,golden)!=EOF);return driver_status;
}
#define HAL_ERROR 1u
@DRIVER@
static void reset_queue(void) {
    write_index=read_index=sequence=dropped=high_water=packet_sequence=prepared_read=prepared_count=0;
}
static void control(unsigned i) {
    output_capture_record_t r={0};r.kind=OUTPUT_CAPTURE_CONTROL;r.begin_ns=1000000000ull+i*2000000ull;r.end_ns=r.begin_ns+10000;
    r.id=i;r.flags=61;r.strategy=0;r.bench_phase=3;r.bench_reason=2;r.wheel_online=3;
    r.torque_nm[4]=3;r.torque_nm[5]=-3;r.wheel_speed[0]=i;r.wheel_speed[1]=-(float)i;
    r.wheel_age_us[0]=1000;r.wheel_age_us[1]=2000;
    Output_Capture_Push(&r);
}
static void drain(FILE *f) {
    uint8_t packet[OUTPUT_CAPTURE_PACKET_MAX];uint16_t n=Output_Capture_Packet(packet);
    assert(fwrite(packet,1,n,f)==n);Output_Capture_Commit();
}
static uint16_t reference_crc(const uint8_t *data,unsigned n) {
    uint16_t crc=0xffffu;
    for(unsigned i=0;i<n;i++) {
        crc^=(uint16_t)data[i]<<8;
        for(unsigned bit=0;bit<8;bit++) { crc=crc&0x8000u?(uint16_t)((crc<<1)^0x1021u):(uint16_t)(crc<<1); }
    }
    return crc;
}
int main(int argc,char **argv) {
    assert(argc==7);golden=fopen(argv[1],"wb");assert(golden);
    FILE *stream=fopen(argv[2],"wb"),*overflow=fopen(argv[3],"wb"),*wrapped=fopen(argv[4],"wb");
    assert(stream && overflow && wrapped);
    assert(Capture_Crc16((const uint8_t*)"123456789",9)==0x29b1);
    uint8_t crc_data[4264];unsigned lengths[]={0,1,2,7,8,15,16,31,32,63,64,87,88,127,128,255,256,511,512,636,640,1024,2152,4264};
    for(unsigned i=0;i<sizeof(crc_data);i++) { crc_data[i]=(uint8_t)(i*73u+19u); }
    for(unsigned i=0;i<sizeof(lengths)/sizeof(lengths[0]);i++) {
        assert(Capture_Crc16(crc_data,lengths[i])==reference_crc(crc_data,lengths[i]));
    }
    uint8_t a[OUTPUT_CAPTURE_PACKET_MAX],b[OUTPUT_CAPTURE_PACKET_MAX];
    irq_mask=1;control(1);assert(irq_mask==1);irq_mask=0;
    control(2);uint16_t n=Output_Capture_Packet(a);assert(n==40+2*88+2);
    assert(Output_Capture_Packet(b)==n && memcmp(a,b,n)==0); /* prepare is not dequeue */
    control(3);assert(fwrite(a,1,n,stream)==n);Output_Capture_Commit();assert(write_index-read_index==1);
    drain(stream);assert(write_index==read_index);
    config.dji_sign[0].out=1;config.dji_sign[1].out=-1;
    float torques[]={-100,-6,-3.9f,-.1f,0,.1f,3.9f,6,100};
    for(unsigned type=0;type<2;type++) {
        config.dji_type=type;config.dji_gear_ratio=type?15.3333333f:36.f;
        for(unsigned status=0;status<4;status++) for(unsigned i=0;i<9;i++) for(unsigned j=0;j<9;j++) {
            driver_status=status;
            assert(Dji_Send_Wheel_Torque(torques[i],torques[j])==status);
            assert(irq_mask==0);
            if(write_index-read_index>=12) drain(stream);
        }
    }
    for(driver_status=0;driver_status<4;driver_status++) assert(Dji_All_Stop()==driver_status);
    while(write_index!=read_index) drain(stream);
    /* Ring-full must retain old entries and expose a sequence gap after recovery. */
    reset_queue();for(unsigned i=0;i<140;i++) control(i);assert(dropped==12 && high_water==128);
    drain(overflow);control(141);
    while(write_index!=read_index) drain(overflow);
    reset_queue();write_index=read_index=0xfffffff0u;sequence=0xfffffff0u;packet_sequence=0xffffffffu;
    for(unsigned i=0;i<32;i++) control(i);
    while(write_index!=read_index) drain(wrapped);
    /* Actual Vofa_Send: one submission, prefix unchanged, failed submission retains queue. */
    reset_queue();control(1);float values[160]={0};values[3]=7101;values[4]=2;
    values[159]=Capture_Crc16((const uint8_t*)values,636);
    assert(!Vofa_Send(values,160) && read_index==0 && submissions==0);
    configured=1;fail_submit=1;assert(!Vofa_Send(values,160) && read_index==0 && submissions==1);
    fail_submit=0;assert(Vofa_Send(values,160) && read_index==1 && submissions==2);
    assert(pending_size==644+40+88+2 && memcmp(pending,values,640)==0);
    FILE *transport=fopen(argv[5],"wb");assert(transport);
    assert(fwrite(pending,1,pending_size,transport)==pending_size);fclose(transport);
    uint8_t held[2800];memcpy(held,pending,pending_size);control(2);
    assert(!Vofa_Send(values,160) && read_index==1 && memcmp(held,pending,pending_size)==0);
    busy=0;assert(Vofa_Send(values,160) && read_index==2);
    /* Virtual 500/1000Hz dispatch and 100Hz drain, including a 40ms USB stall. */
    FILE *stress=fopen(argv[6],"wb"),*saved=golden;golden=tmpfile();assert(stress && golden);
    for(unsigned division=1;division<=2;division++) {
        reset_queue();driver_status=0;
        for(unsigned tick=0;tick<2000;tick++) {
            now_ns=1000000000ull+tick*1000000ull;
            if(tick%division==0) {
                assert(Dji_Send_Wheel_Torque(.1f,-.1f)==0);
                output_capture_record_t r={0};r.kind=OUTPUT_CAPTURE_CONTROL;r.begin_ns=now_ns;r.end_ns=now_ns+1000;r.id=tick;
                Output_Capture_Push(&r);
            }
            if(tick%10==9 && !(tick>=400 && tick<440)) drain(stress);
        }
        while(write_index!=read_index) drain(stress);
        assert(dropped==0);
    }
    fclose(stress);fclose(golden);golden=saved;
    fclose(golden);fclose(stream);fclose(overflow);fclose(wrapped);return 0;
}
'''
    driver=(H7/'imcalib/drivers/dji.c').read_text()
    names=('Dji_Torque_To_Current','Dji_Send_Current','Dji_Send_Wheel_Torque','Dji_All_Stop')
    header=(H7/'imcalib/drivers/dji.h').read_text()
    constants='\n'.join(line for line in header.splitlines() if re.match(r'#define DJI_(CURRENT_MAX|NM_FULL|RATIO_STD)_',line))
    text=harness.replace('@LATENCY_HEADER@', str(H7/'imcalib/bsp/latency_capture.h')).replace('@CAPTURE@',str(H7/'imcalib/bsp/output_capture.c')).replace('@SENDER@',str(H7/'imcalib/bsp/Vofa_send.c'))
    text=text.replace('@CONSTANTS@',constants).replace('@DRIVER@','\n'.join(extract(driver,n) for n in names))
    source=directory/'probe.c';source.write_text(text);exe=directory/'probe'
    args=[os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-I',str(stubs),'-I',str(H7/'imcalib/bsp'),str(source),'-lm','-o',str(exe)]
    compiled=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (directory/'compile.log').write_text(compiled.stdout);compiled.check_returncode()
    files=[directory/n for n in ('driver.bin','stream.bin','overflow.bin','wrapped.bin','transport.bin','stress.bin')]
    subprocess.run([str(exe),*map(str,files)],check=True)
    # Run the exact pre-instrumentation driver with the same commands and HAL stub.
    old=(ROOT/'artifacts/h7-output-capture-20261021/source-baseline/deployment/H7_RL_ros/imcalib/drivers/dji.c').read_text()
    for name in ('Dji_Torque_To_Current','Dji_Send_Wheel_Torque','Dji_All_Stop'):
        assert extract(old,name)==extract(driver,name), 'original driver behavior changed: '+name
    before=ROOT/'artifacts/h7-output-capture-20261021/source-baseline/deployment/H7_RL_ros/imcalib/Algorithm/rl_torque.c'
    assert before.read_bytes()==(H7/'imcalib/Algorithm/rl_torque.c').read_bytes(), 'RL controller changed'
    old_text=text.replace('\n'.join(extract(driver,n) for n in names),'\n'.join(extract(old,n) for n in names))
    # Baseline does not enqueue driver records, so use the same queue assertions on control-only cases.
    old_source=directory/'baseline.c';old_source.write_text(old_text);old_exe=directory/'baseline'
    compiled=subprocess.run(args[:-4]+[str(old_source),'-lm','-o',str(old_exe)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (directory/'compile-baseline.log').write_text(compiled.stdout);compiled.check_returncode()
    baseline_files=[directory/('baseline-'+p.name) for p in files]
    subprocess.run([str(old_exe),*map(str,baseline_files)],check=True)
    assert files[0].read_bytes()==baseline_files[0].read_bytes(), 'driver payload or status changed'
    parser=capture.CaptureParser();packets=[]
    raw=files[1].read_bytes()
    for i in range(0,len(raw),17):packets+=parser.feed(raw[i:i+17])
    assert parser.rejected==0
    flat=[r for p in packets for r in p['records']]
    can=[r for r in flat if r[1]==2]
    assert len(can)==2*4*9*9+4 and len(flat)==len(can)+3
    golden=files[0].read_bytes()
    for i,r in enumerate(can):
        assert struct_pack_current(r[16:20])==golden[i*9:i*9+8] and r[20]==golden[i*9+8]
    control=[r for r in flat if r[1]==1][0]
    assert control[10:14]==(3.,-3.,1.,-1.) and control[20:27]==(61,0,3,2,0,3,0)
    damaged=bytearray(raw);damaged[50]^=1;bad=capture.CaptureParser();bad.feed(damaged);assert bad.rejected
    overflow=capture.CaptureParser().feed(files[2].read_bytes());assert max(p['dropped'] for p in overflow)==12
    wrap=capture.CaptureParser().feed(files[3].read_bytes());seq=[r[0] for p in wrap for r in p['records']]
    assert 0 in seq and all((b-a)&0xffffffff==1 for a,b in zip(seq,seq[1:]))
    import real2sim_huananhu as mirror
    wire=files[4].read_bytes();legacy=mirror.FrameParser();assert len(legacy.feed(wire))==1
    assert len(capture.CaptureParser().feed(wire)[0]['records'])==1
    stressed=capture.CaptureParser().feed(files[5].read_bytes())
    assert sum(len(p['records']) for p in stressed)==6000 and all(p['dropped']==0 for p in stressed)
    return {'actual_CAN_requests_checked':len(can),'driver_payload_and_HAL_results_equal_baseline':True,
            'original_RL_and_current_conversion_and_wheel_send_functions_unchanged':True,
            'queue_overflow_gap_retry_preemption_wrap_IRQ_state':True,'actual_USB_busy_failure_buffer_lifetime':True,
            'partial_reads_crc_type_and_record_fields':True,
            'legacy_mirror_prefix_and_capture_sideband_both_readable':True,
            'virtual_500_1000Hz_40ms_stall_records':6000},raw


def struct_pack_current(values):
    import struct
    return struct.pack('>4h',*values)


def session_checks(directory, raw):
    import real2sim_huananhu as mirror
    config=H7/'imcalib/config/machine_config.c'
    manifest=directory/'expected-build.json';manifest.write_text('{"hex_sha256":"example-not-boot-proof"}')
    session=directory/'synthetic-session';rec=mirror.SessionRecorder(session,machine_config=config,firmware_manifest=manifest)
    rec.record('connected',elapsed_ns=0)
    for i in range(0,len(raw),997):rec.record('data',elapsed_ns=i+1,data_hex=raw[i:i+997].hex())
    rec.finish(elapsed_ns=len(raw)+1)
    result=capture.analyze_session(session,directory/'analysis')
    assert result['records']==655 and result['complete_observed_span']
    assert result['firmware_expectation']['expected_build']['hex_sha256']=='example-not-boot-proof'
    rows=list(csv.DictReader((directory/'analysis/outputs.csv').open()))
    assert all(r['can_current_slot_0']=='' for r in rows if r['kind']=='control')
    assert all(r['request_0_firmware_nm']=='' for r in rows if r['kind']=='can')
    cases={}
    original=raw
    corrupted=bytearray(raw);corrupted[50]^=1
    for name,data in [('crc',bytes(corrupted)),('truncated',original[:-10]),
                      ('overflow',(directory/'overflow.bin').read_bytes())]:
        target=directory/('session-'+name);record=mirror.SessionRecorder(target,machine_config=config)
        record.record('connected',elapsed_ns=0)
        for i in range(0,len(data),997):record.record('data',elapsed_ns=i+1,data_hex=data[i:i+997].hex())
        record.finish(elapsed_ns=len(data)+1)
        audit=capture.analyze_session(target,directory/('analysis-'+name))
        assert not audit['complete_observed_span'];cases[name]=True
    old=directory/'session-old';record=mirror.SessionRecorder(old,machine_config=config)
    record.record('connected',elapsed_ns=0)
    record.record('data',elapsed_ns=1,data_hex=(directory/'transport.bin').read_bytes()[:644].hex())
    record.finish(elapsed_ns=2)
    try:capture.analyze_session(old,directory/'analysis-old')
    except ValueError as e:assert 'old firmware' in str(e)
    else:raise AssertionError('old firmware accepted as complete-output capture')
    return {'raw_session_hash_and_archived_build':True,'end_to_end_C_binary_to_session_CSV':True,
            'records':result['records'],'failed_CAN_attempts_explicit':result['failed_CAN_attempts'],
            'old_firmware_has_no_capture_rejected':True,'incomplete_sessions_detected':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build-h7',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:
        native_result,raw=native(a.out)
        report={'ok':True,'scope':'finite host software/build checks, no H7 timing or hardware acceptance',
                'native':native_result,'session':session_checks(a.out,raw)}
        if a.build_h7:
            import check_real2sim_huananhu as existing
            existing.H7=H7;report['build']=existing.build_h7(a.out)
        report['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),ROOT/'sim2sim/h7_output_capture.py',H7/'imcalib/bsp/output_capture.c',
                            H7/'imcalib/bsp/output_capture.h',H7/'imcalib/bsp/Vofa_send.c',H7/'imcalib/drivers/dji.c',
                            H7/'imcalib/task/task_actuation.c',H7/'imcalib/task/inc/robot_control.h']}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    except BaseException as e:
        (a.out/'failed.json').write_text(json.dumps({'ok':False,'error':str(e)})+'\n');raise
if __name__=='__main__':main()
