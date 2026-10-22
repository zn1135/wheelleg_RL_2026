#!/usr/bin/env python3
"""Finite actual-C timing capture/CAN checks, decoder roundtrip and optional ARM link.

No device, CAN bus, serial port, policy inference or simulation is run. IRQs,
task identities, FDCAN registers and clocks are explicit host stubs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
H7 = ROOT / 'deployment/H7_RL_ros'
sys.path.insert(0, str(ROOT / 'sim2sim'))
import h7_latency_capture as decoder
import real2sim_huananhu as mirror
from check_h7_output_capture import extract


def native(out):
    stub = out / 'stubs'
    stub.mkdir()
    (stub / 'main.h').write_text('''#include <stdint.h>
extern uint32_t irq_mask;
#define __get_PRIMASK() irq_mask
#define __disable_irq() (irq_mask=1u)
#define __set_PRIMASK(x) (irq_mask=(x))
#define __DMB() __asm__ __volatile__("" ::: "memory")
''')
    (stub / 'FreeRTOS.h').write_text('')
    (stub / 'task.h').write_text('''typedef void *TaskHandle_t;
extern TaskHandle_t current_owner;
#define xTaskGetCurrentTaskHandle() current_owner
''')
    (stub / 'mono_ns.h').write_text('#include <stdint.h>\nuint64_t Mono_Ns_Get(void);\n')
    can = (H7 / 'imcalib/bsp/can_bus.c').read_text()
    functions = '\n'.join(extract(can, name) for name in (
        'Can_Latency_Finish', 'Can_Bus_Transmit', 'HAL_FDCAN_TxBufferCompleteCallback'))
    source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
uint32_t irq_mask;
void *current_owner=(void*)1;
static uint64_t now_ns=1000000;
uint64_t Mono_Ns_Get(void) { now_ns+=100;return now_ns; }
#include "@CAPTURE@"
typedef unsigned HAL_StatusTypeDef;
typedef struct { unsigned Identifier,DataLength; } FDCAN_TxHeaderTypeDef;
typedef struct { unsigned TXFQS,TXBTO,TXBRP; } registers_t;
typedef struct { registers_t *Instance; } FDCAN_HandleTypeDef;
#define CAN_BUS_NUM 3
#define HAL_OK 0u
#define HAL_ERROR 1u
#define FDCAN_TXFQS_TFQPI 31u
#define FDCAN_TXFQS_TFQPI_Pos 0u
#define taskENTER_CRITICAL() do { assert(!irq_mask);irq_mask=1; } while(0)
#define taskEXIT_CRITICAL() do { irq_mask=0; } while(0)
static struct { FDCAN_TxHeaderTypeDef tx_template;unsigned last_tx_tick; } can_bus[3];
static latency_record_t latency_tx[3][32];
static uint8_t latency_tx_active[3][32];
static uint32_t latency_tx_sequence;
static registers_t registers[3];
static FDCAN_HandleTypeDef handles[3]={{&registers[0]},{&registers[1]},{&registers[2]}};
static unsigned full,fail,enqueues;
static uint8_t last_data[8];
static int32_t Can_Bus_Hw_Index(FDCAN_HandleTypeDef *h) {
 for(unsigned i=0;i<3;i++) { if(h==&handles[i]) return i; }
 return -1;
}
static unsigned HAL_GetTick(void) { return 123u; }
static unsigned HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef *h) { (void)h;return !full; }
static unsigned HAL_FDCAN_AddMessageToTxFifoQ(FDCAN_HandleTypeDef *h,FDCAN_TxHeaderTypeDef *header,uint8_t *data) {
 assert(header->DataLength==8u);enqueues++;memcpy(last_data,data,8);
 if(fail) return HAL_ERROR;
 unsigned bit=1u<<h->Instance->TXFQS;
 h->Instance->TXBRP|=bit;h->Instance->TXBTO&=~bit;
 return HAL_OK;
}
@CAN@
static void reset_capture(void) {
 write_index=read_index=sequence=dropped=high_water=packet_sequence=prepared_read=prepared_count=can_pending=0;
 memset(latency_tx_active,0,sizeof(latency_tx_active));memset(registers,0,sizeof(registers));
 Latency_Dispatch_End();irq_mask=0;full=fail=enqueues=0;
}
static void emit(FILE *f) {
 uint8_t packet[LATENCY_PACKET_MAX];unsigned n=Latency_Capture_Packet(packet);
 assert(fwrite(packet,1,n,f)==n);Latency_Capture_Commit();
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
 assert(argc==2);FILE *f=fopen(argv[1],"wb");assert(f);
 assert(sizeof(latency_record_t)==64);
 assert(Latency_Age_Us(100,0)==UINT32_MAX);
 assert(Latency_Age_Us(100,101)==UINT32_MAX);
 assert(Latency_Age_Us(11001,1)==11);
 assert(Latency_Age_Us(((uint64_t)UINT32_MAX+1)*1000,1)==UINT32_MAX);
 assert(Latency_Crc((const uint8_t*)"123456789",9)==0x29b1);
 uint8_t crc_data[8232];unsigned lengths[]={0,1,2,7,8,15,16,31,32,63,64,87,88,127,128,255,256,511,512,636,640,1024,2152,4264,8232};
 for(unsigned i=0;i<sizeof(crc_data);i++) { crc_data[i]=(uint8_t)(i*73u+19u); }
 for(unsigned i=0;i<sizeof(lengths)/sizeof(lengths[0]);i++) {
  assert(Latency_Crc(crc_data,lengths[i])==reference_crc(crc_data,lengths[i]));
 }
 latency_record_t record={0};record.kind=LATENCY_SAMPLE;record.begin_ns=record.end_ns=now_ns;
 irq_mask=1;Latency_Capture_Push(&record);assert(irq_mask==1);irq_mask=0;
 for(unsigned i=0;i<LATENCY_CAPACITY;i++) Latency_Capture_Push(&record);
 assert(dropped==1 && high_water==LATENCY_CAPACITY);
 uint8_t packet[LATENCY_PACKET_MAX],before[LATENCY_PACKET_MAX];
 unsigned n=Latency_Capture_Packet(packet);memcpy(before,packet,n);
 assert(read_index==0 && prepared_count==128);
 assert(Latency_Capture_Packet(packet)==n && !memcmp(before,packet,n));
 Latency_Capture_Commit();assert(read_index==128);
 reset_capture();sequence=UINT32_MAX;write_index=read_index=UINT32_MAX-2;
 Latency_Capture_Push(&record);assert(records[write_index%LATENCY_CAPACITY-1].sequence==0);
 reset_capture();
 Latency_Dispatch_Begin(0,42,9);Latency_Can_Context(&record);
 assert(record.flags&128 && record.value[6]==0 && record.value[7]==42 && record.reset_count==9);
 record=(latency_record_t){0};current_owner=(void*)2;Latency_Can_Context(&record);
 assert(!record.flags && record.value[7]==0);current_owner=(void*)1;
 Latency_Tick(0);assert(Latency_Tick_Read(0).sequence==1);
 uint8_t data[8]={1,2,3,4,5,6,7,8};
 for(unsigned b=0;b<3;b++) {
   handles[b].Instance->TXFQS=31;
   assert(Can_Bus_Transmit(&handles[b],0x200+b,data,8)==HAL_OK);
   assert(!memcmp(last_data,data,8));assert(latency_tx_active[b][31]);
 }
 assert(can_pending==3 && write_index==0);
 /* One bus completion must not consume another bus's same slot. */
 registers[0].TXBRP=0;registers[0].TXBTO=1u<<31;
 HAL_FDCAN_TxBufferCompleteCallback(&handles[0],1u<<31);
 assert(can_pending==2 && write_index==1 && records[0].value[0]==0x200);
 assert(records[0].flags==129 && records[0].value[7]==42 && records[0].value[6]==0);
 HAL_FDCAN_TxBufferCompleteCallback(&handles[0],1u<<31);assert(write_index==1);
 /* Reused buffer completed before its IRQ: emit a polled upper bound first. */
 registers[1].TXBRP=0;registers[1].TXBTO=1u<<31;
 Can_Bus_Transmit(&handles[1],0x201,data,8);
 assert(records[1].flags==130 && can_pending==2);
 /* A stale TXBTO IRQ cannot finish the new still-pending request. */
 HAL_FDCAN_TxBufferCompleteCallback(&handles[1],1u<<31);assert(can_pending==2);
 full=1;assert(Can_Bus_Transmit(&handles[0],0x200,data,8)==HAL_ERROR);
 assert(records[2].flags==131 && can_pending==2);full=0;fail=1;
 assert(Can_Bus_Transmit(&handles[0],0x200,data,8)==HAL_ERROR);assert(can_pending==2);fail=0;
 for(unsigned b=1;b<3;b++) {
   registers[b].TXBRP=0;registers[b].TXBTO=1u<<31;
   HAL_FDCAN_TxBufferCompleteCallback(&handles[b],1u<<31);
 }
 assert(can_pending==0);Latency_Dispatch_End();
 /* Emit real C records for all timing stages, reset pairing and hold counters. */
 record=(latency_record_t){0};record.kind=LATENCY_SAMPLE;record.correlation=7;record.reset_count=9;
 record.begin_ns=10000000;record.end_ns=10001000;
 for(unsigned i=0;i<8;i++)record.value[i]=100+i;
 Latency_Capture_Push(&record);
 record=(latency_record_t){0};record.kind=LATENCY_POLICY;record.flags=4;record.correlation=42;record.reset_count=9;
 record.begin_ns=10002000;record.end_ns=10720000;record.value[0]=7;record.value[1]=1000;record.value[2]=3;record.value[3]=710;
 Latency_Capture_Push(&record);
 record=(latency_record_t){0};record.kind=LATENCY_PD;record.flags=3;record.correlation=17;record.reset_count=9;
 record.begin_ns=12000000;record.end_ns=12004000;record.value[0]=42;record.value[1]=1280;
 Latency_Capture_Push(&record);
 record.flags=1;record.begin_ns+=2000000;record.end_ns+=2000000;record.value[1]=3280;record.correlation++;
 Latency_Capture_Push(&record);
 record=(latency_record_t){0};record.kind=LATENCY_ACTUATION;record.begin_ns=13000000;record.end_ns=13010000;
 record.value[0]=13;Latency_Capture_Push(&record);
 record.begin_ns+=3000000;record.end_ns+=3000000;record.value[0]=16;Latency_Capture_Push(&record);
 emit(f);fclose(f);puts("actual C capture/CAN boundary checks passed");return 0;
}
'''.replace('@CAPTURE@', str(H7 / 'imcalib/bsp/latency_capture.c')).replace('@CAN@', functions)
    (out / 'native.c').write_text(source)
    executable = out / 'native'
    command = [os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-I', str(stub),
               '-I', str(H7 / 'imcalib/bsp'), str(out / 'native.c'), '-o', str(executable)]
    subprocess.run(command, check=True)
    subprocess.run([str(executable), str(out / 'wire.bin')], check=True)
    return (out / 'wire.bin').read_bytes()


def session_checks(out, wire):
    def recording(name, data):
        path = out / name
        recorder = mirror.SessionRecorder(path, machine_config=H7 / 'imcalib/config/machine_config.c')
        recorder.record('connected', elapsed_ns=0)
        for i in range(0, len(data), 17):
            recorder.record('data', elapsed_ns=i + 1, data_hex=data[i:i + 17].hex())
        recorder.finish(elapsed_ns=len(data) + 1)
        return path
    session = recording('session', wire)
    report = decoder.decode_session(session, out / 'analysis')
    assert report['records_complete_in_observed_span']
    assert report['last_pending_can'] == 0
    assert report['can_results'] == {'1': 3, '2': 1, '3': 2}
    first = next(s for s in report['statistics'] if s['metric'] == 'publish_to_first_pd_use_us')
    assert first['count'] == 1 and first['mean_us'] == 1280
    assert report['unavailable_pairs_and_trigger_notes']['actuation_trigger_advances_not_one'] == 1
    assert any(s['metric'] == 'cache_to_action_publish_us' and s['mean_us'] == 719 for s in report['statistics'])
    damaged = bytearray(wire);damaged[50] ^= 1
    # A lone corrupt or truncated batch cannot masquerade as valid new timing data.
    for name, data in [('crc', damaged), ('truncated', wire[:-1]), ('old', b'H7OUT001')]:
        try:
            decoder.decode_session(recording(name, data), out / (name + '-analysis'))
        except ValueError as error:
            assert 'no H7LAT001' in str(error)
        else:
            raise AssertionError(name + ' accepted as complete timing capture')
    return report


def combined_transport(out, raw):
    """Compile separate real queue/sender translation units; verify one USB buffer."""
    stub = out / 'transport-stubs'
    stub.mkdir()
    (stub / 'main.h').write_text('''#include <stdint.h>
extern uint32_t irq_mask;
#define __get_PRIMASK() irq_mask
#define __disable_irq() (irq_mask=1u)
#define __set_PRIMASK(x) (irq_mask=(x))
#define __DMB() __asm__ __volatile__("" ::: "memory")
''')
    (stub / 'FreeRTOS.h').write_text('')
    (stub / 'task.h').write_text('typedef void *TaskHandle_t;\n#define xTaskGetCurrentTaskHandle() ((void*)1)\n')
    (stub / 'mono_ns.h').write_text('#include <stdint.h>\nuint64_t Mono_Ns_Get(void);\n')
    (stub / 'usart.h').write_text('')
    (stub / 'machine_config.h').write_text('#define MACHINE_DEFAULT 0u\n#define MACHINE_VOFA_PORT 10u\n')
    (stub / 'usbd_cdc_if.h').write_text('''#include <stdint.h>
#define USBD_OK 0u
uint8_t CDC_Transmit_Ready_HS(void);
uint8_t CDC_Transmit_HS(uint8_t *,uint16_t);
''')
    source = out / 'transport.c'
    source.write_text(r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "latency_capture.h"
#include "output_capture.h"
#include "Vofa_send.h"
uint32_t irq_mask;
static unsigned busy,failed;
static uint8_t *pending;
static uint16_t pending_size;
static FILE *stress;
uint64_t Mono_Ns_Get(void) { return 1000000; }
uint8_t CDC_Transmit_Ready_HS(void) { return !busy; }
uint8_t CDC_Transmit_HS(uint8_t *data,uint16_t n) {
 if(failed) { return 2; }
 pending=data;pending_size=n;busy=1;
 if(stress) { assert(fwrite(data,1,n,stress)==n); }
 return 0;
}
int main(int argc,char **argv) {
 assert(argc==4);float frame[160];FILE *f=fopen(argv[1],"rb");assert(f);
 assert(fread(frame,1,sizeof(frame),f)==sizeof(frame));fclose(f);
 latency_record_t t={0};t.kind=LATENCY_PD;t.begin_ns=1000000;t.end_ns=1002000;
 output_capture_record_t o={0};o.kind=OUTPUT_CAPTURE_CONTROL;o.begin_ns=1000000;o.end_ns=1002000;
 for(unsigned i=0;i<LATENCY_BATCH;i++)Latency_Capture_Push(&t);
 for(unsigned i=0;i<OUTPUT_CAPTURE_BATCH;i++)Output_Capture_Push(&o);
 failed=1;assert(!Vofa_Send(frame,160));failed=0;
 assert(Vofa_Send(frame,160));
 assert(pending_size==644+OUTPUT_CAPTURE_PACKET_MAX+LATENCY_PACKET_MAX);
 uint8_t copy[644+OUTPUT_CAPTURE_PACKET_MAX+LATENCY_PACKET_MAX];memcpy(copy,pending,pending_size);
 assert(!Vofa_Send(frame,160) && !memcmp(copy,pending,pending_size));
 f=fopen(argv[2],"wb");assert(f);assert(fwrite(pending,1,pending_size,f)==pending_size);fclose(f);
 busy=0;assert(Vofa_Send(frame,160));assert(pending_size==644+42+42);
 /* 50Hz导出低于实测78Hz；覆盖1kHz最坏记录量及40ms接收停顿。 */
 stress=fopen(argv[3],"wb");assert(stress);
 for(unsigned division=1;division<=2;division++) {
  for(unsigned tick=0;tick<4000;tick++) {
   if(tick%division==0) {
    Output_Capture_Push(&o);Output_Capture_Push(&o);
    for(unsigned i=0;i<6;i++) { Latency_Capture_Push(&t); }
   }
   if(tick%20==0) { Latency_Capture_Push(&t); }
   if(tick%20==19 && !(tick>=400 && tick<440)) {
    busy=0;assert(Vofa_Send(frame,160));
   }
  }
 }
 fclose(stress);stress=NULL;
 return 0;
}
''')
    executable = out / 'transport'
    command = [os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-I', str(stub), '-I', str(H7 / 'imcalib/bsp'),
               str(source), str(H7 / 'imcalib/bsp/latency_capture.c'), str(H7 / 'imcalib/bsp/output_capture.c'),
               str(H7 / 'imcalib/bsp/Vofa_send.c'), '-o', str(executable)]
    subprocess.run(command, check=True)
    (out / 'mirror-input.bin').write_bytes(raw)
    subprocess.run([str(executable), str(out / 'mirror-input.bin'), str(out / 'combined.bin'),
                    str(out / 'paced-transport.bin')], check=True)
    combined = (out / 'combined.bin').read_bytes()
    import h7_output_capture as outputs
    m, o, t = mirror.FrameParser(), outputs.CaptureParser(), decoder.LatencyParser()
    fs, os_, ts = [], [], []
    for i in range(0, len(combined), 17):
        chunk = combined[i:i + 17]
        fs.extend(m.feed(chunk));os_.extend(o.feed(chunk));ts.extend(t.feed(chunk))
    assert len(fs) == len(os_) == len(ts) == 1
    assert len(os_[0]['records']) == 48 and len(ts[0]['records']) == 128
    assert ts[0]['dropped'] == os_[0]['dropped'] == 0
    # Decode the actual C sender's sustained output, including both real queues.
    paced_o, paced_t = outputs.CaptureParser(), decoder.LatencyParser()
    paced_outputs, paced_timings = [], []
    raw = (out / 'paced-transport.bin').read_bytes()
    for i in range(0, len(raw), 997):
        paced_outputs.extend(paced_o.feed(raw[i:i + 997]))
        paced_timings.extend(paced_t.feed(raw[i:i + 997]))
    assert paced_o.rejected == paced_t.rejected == 0
    assert sum(len(p['records']) for p in paced_outputs) == 12000
    assert sum(len(p['records']) for p in paced_timings) == 36400
    assert max(p['queued'] for p in paced_outputs) > 48
    assert all(p['dropped'] == 0 for p in paced_outputs + paced_timings)
    return dict(actual_separate_C_translation_units=True, max_combined_bytes=len(combined),
                mirror_frames=len(fs), output_records=48, timing_records=128,
                USB_busy_preserves_buffer=True, failed_submit_retains_both_queues=True,
                successful_submit_commits_both_queues=True, fragmented_all_three_formats=True,
                paced_export_Hz=50, virtual_dispatch_Hz=[1000, 500], USB_stall_ms=40,
                paced_output_records=12000, paced_timing_records=36400,
                paced_both_queues_zero_drops=True)


def policy_boundaries(out):
    # Extend the existing actual-source harness; this still runs the real task_policy.c.
    path = out / 'regression/policy-producer.c'
    text = path.read_text().replace('int main(void)', 'int earlier_checks(void)')
    text = text.replace('void Latency_Capture_Push(const latency_record_t *record) { assert(record->kind==LATENCY_POLICY); }',
                        'static latency_record_t last_timing;\nvoid Latency_Capture_Push(const latency_record_t *record) { last_timing=*record; }')
    text = text.replace('    return 1;\n}\nstatic void reset_case', '    return mode == 4u ? 0u : 1u;\n}\nstatic void reset_case')
    text += r'''
int main(void) {
 earlier_checks();
 reset_case(0);mirror_policy_snapshot.sequence=UINT32_MAX;ctrl_task_body();
 assert(mirror_policy_snapshot.sequence==0 && last_timing.flags==4 && last_timing.correlation==0);
 assert(last_timing.value[3]==321 && last_timing.end_ns==latency_policy_published_ns);
 reset_case(0);rl_control.observation.sample_valid=0;ctrl_task_body();
 assert(last_timing.flags==1 && last_timing.value[3]==UINT32_MAX);
 reset_case(4);rl_control.policy.run_us=999;ctrl_task_body();
 assert(last_timing.flags==2 && last_timing.value[3]==UINT32_MAX);
 reset_case(0);rc_command.online=0;ctrl_task_body();assert(last_timing.flags==0);
 reset_case(2);ctrl_task_body();assert(last_timing.flags==3 && last_timing.value[3]==321);
 puts("actual policy timing wrap/invalid/SDK failure/reset checks passed");return 0;
}
'''
    source = out / 'policy-boundaries.c'
    source.write_text(text)
    executable = out / 'policy-boundaries'
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-I', str(out / 'regression/policy-stub'),
                    str(source), '-lm', '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    return dict(actual_policy_sequence_wrap=True, SDK_failure_has_no_stale_duration=True,
                invalid_sample_and_unselected_explicit=True, reset_discard_keeps_valid_SDK_duration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--build-h7', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    wire = native(args.out)
    decoded = session_checks(args.out, wire)
    report = dict(ok=True, scope='finite synthetic software checks; no hardware or timing performance acceptance',
                  decoder_record_count=decoded['record_count'], can_results=decoded['can_results'],
                  cases=['same bus/slot completion correlation', 'cross-bus same slot', 'stale completion mask',
                         'reuse before IRQ', 'HAL full/failure', 'task-context ownership', 'zero output-sequence wrap',
                         'capture queue overflow', 'producer sequence wrap', 'retry without commit',
                         'CRC, truncation, old firmware', '17-byte fragments', 'first PD use', 'trigger advances'])
    import check_real2sim_huananhu as checks
    checks.H7 = H7
    regression_dir = args.out / 'regression'
    regression_dir.mkdir()
    raw, _ = checks.c_packets(regression_dir, True)
    report['regression'] = {
        'policy_pairing_and_reset': checks.check_policy_producer(regression_dir),
        'output_gate_and_requests': checks.check_output_producer(regression_dir),
        'usb_busy_and_failure': checks.check_usb_sender(regression_dir, raw),
    }
    report['combined_transport'] = combined_transport(args.out, raw)
    report['policy_timing_boundaries'] = policy_boundaries(args.out)
    if args.build_h7:
        report['build'] = checks.build_h7(args.out)
    report['sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), ROOT / 'sim2sim/h7_latency_capture.py',
                                   H7 / 'imcalib/bsp/latency_capture.c', H7 / 'imcalib/bsp/can_bus.c']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
