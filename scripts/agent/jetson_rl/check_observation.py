"""Compare Jetson C++ continuous FIFO vectors against actual read-only H7_RL source."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--cc',required=True);p.add_argument('--core',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 subprocess.run([str(a.core),str(a.out/'input.bin'),str(a.out/'cpp.bin')],check=True)
 root=Path('deployment/H7_RL');text=(root/'imcalib/config/machine_config.c').read_text();start=text.index('{',text.index('    .rl ='));end=start+1;depth=1
 while depth: depth+=(text[end]=='{')-(text[end]=='}');end+=1
 config=text[start:end]
 harness=r'''
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "rl_observation.h"
const machine_cfg_t test_machine={.rl=CONFIG};
const machine_cfg_t *const machine=&test_machine;
int main(int argc,char **argv) {
 assert(argc==3);FILE *in=fopen(argv[1],"rb"),*out=fopen(argv[2],"wb");assert(in&&out);
 rl_observation_state_t state={0};float row[26];unsigned n=0;
 while(fread(row,sizeof(row),1,in)==1) {
  if(row[25]) RL_Observation_Reset(&state);
  rl_observation_sample_t sample;memcpy(sample.gyro,row,12);memcpy(sample.gravity,row+3,12);memcpy(sample.pos,row+6,16);memcpy(sample.vel,row+10,24);
  memcpy(state.last_action,row+19,24);rc_command_t cmd={0};cmd.velocity_target=row[16];cmd.yaw_rate_target=row[17];cmd.leg_length_target=row[18];
  rl_recovery_state_t recovery={.phase=RL_RECOVERY_NORMAL};assert(RL_Observation_Build(&state,&machine->rl,&sample,&cmd,&recovery,1));
  assert(fwrite(state.obs,sizeof(state.obs),1,out)==1);assert(fwrite(state.history,sizeof(state.history),1,out)==1);n++;
 }
 assert(n==256);fclose(in);fclose(out);return 0;
}
'''.replace('CONFIG',config)
 (a.out/'actual.c').write_text(harness)
 dirs=sorted({p.parent for p in (root/'imcalib').rglob('*.h')})
 cmd=[a.cc,'-O2','-std=c11']+['-I'+str(d) for d in dirs]+[str(a.out/'actual.c'),str(root/'imcalib/Algorithm/rl/rl_observation.c'),'-lm','-o',str(a.out/'actual')]
 result=subprocess.run(cmd,capture_output=True,text=True);(a.out/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
 subprocess.run([str(a.out/'actual'),str(a.out/'input.bin'),str(a.out/'actual.bin')],check=True)
 actual=np.fromfile(a.out/'actual.bin',np.float32).reshape(-1,150);cpp=np.fromfile(a.out/'cpp.bin',np.float32).reshape(-1,150)
 np.testing.assert_allclose(cpp,actual,atol=1e-4,rtol=1e-4)
 report=dict(ok=True,frames=len(cpp),max_abs=float(np.max(abs(cpp-actual))),scope='actual H7_RL observation/FIFO vs Jetson C++; no board')
 (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
