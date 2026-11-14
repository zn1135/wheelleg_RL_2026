"""Audit the allowed firmware delta and verify retained hardware/control sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def old(path):return subprocess.check_output(['git','show','HEAD:'+str(path)])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--elf',type=Path,required=True);a=p.parse_args()
 root=Path('deployment/H7_RL_ros')
 assert not subprocess.check_output(['git','diff','HEAD','--','deployment/H7_RL'])
 assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','deployment/H7_RL'])
 config=root/'imcalib/config/machine_config.c'
 assert config.read_bytes()==old(config).replace(b'.wheel_gain = {0.1f, 0.1f},',b'.wheel_gain = {0.045f, 0.045f},')
 preserved=['imcalib/Algorithm/rl_observation.c','imcalib/Algorithm/rl_torque.c','imcalib/drivers/dm.c','imcalib/drivers/dji.c','imcalib/drivers/hi229.c','imcalib/task/rc_command.c']
 for name in preserved:
  path=root/name;assert path.read_bytes()==old(path),name
 for name in ['imcalib/bsp/Vofa_send.c','imcalib/task/task_actuation.c']:
  path=root/name;before=old(path).decode();after=path.read_text()
  if 'Vofa_send' in name:
   assert before[before.index('uint8_t Vofa_Send('):]==after[after.index('uint8_t Vofa_Send('):]
  else:
   expected=before.replace('#include "joint_capture.h"','#include "joint_capture.h"\n#include "jetson_uart.h"').replace('diagnostic.flags |= 0x01u;','diagnostic.flags |= 0x01u;\n                        Jetson_Uart_Pd(diagnostic.policy_sequence, pd_timing.reset_count);')
   assert after==expected,'actuation delta exceeds callback'
 symbols=subprocess.check_output(['/usr/bin/arm-none-eabi-nm',str(a.elf)],text=True)
 assert 'ai_networkzn1_run' not in symbols and 'RL_Policy_Run' not in symbols and 'RL_Policy_Init' not in symbols
 assert 'Jetson_Uart_Process' in symbols and 'RL_Torque_Compute' in symbols
 report=dict(ok=True,H7_RL_unchanged=True,physical_sources_unchanged=True,only_control_parameter_change='wheel_gain 0.1 -> 0.045',usb_packet_body_unchanged=True,actuation_only_pd_trace_callback=True,local_network_absent_from_linked_firmware=True,elf_sha256=sha(a.elf))
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
