"""Export frozen 17679 and compare actual H7_RL generated C with PyTorch/ONNX.

No board access or training. Uses the repository Python environment and existing ST host SDK.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from ct_mjlab.fudan_ppo_walk.export import export, Policy
from ct_mjlab.fudan_ppo_walk.policy import load_policy

EXPECTED = '5b22740659e44bbb1ca1173e9289b75f5848059d0972d4257b79be3c0d217f2b'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--st-host',type=Path,required=True);p.add_argument('--cc',required=True)
    a=p.parse_args();assert sha(a.checkpoint)==EXPECTED
    a.out.mkdir(parents=True,exist_ok=False)
    export(a.checkpoint,a.out/'model')
    network,metadata=load_policy(a.checkpoint);assert metadata['reset_contract'].endswith('unclipped-history-v2')
    policy=Policy(network).eval();rng=np.random.default_rng(17679)
    x=rng.uniform(-1,1,(256,150)).astype(np.float32);x[0]=0;x[1]=.5;x[2]=100;x[3]=-100
    with torch.no_grad(): y=policy(torch.from_numpy(x[:,:25]),torch.from_numpy(x[:,25:])).numpy()
    ref=ReferenceEvaluator(onnx.load(a.out/'model/policy.onnx'))
    z=np.concatenate([ref.run(None,{'obs':v[None,:25],'obs_history':v[None,25:]})[0] for v in x])
    np.testing.assert_allclose(z,y,atol=1e-4,rtol=1e-4)
    x.tofile(a.out/'inputs.bin');y.tofile(a.out/'pytorch.bin');z.tofile(a.out/'onnx.bin')
    harness=a.out/'network_check.c'
    harness.write_text(r'''
#include <assert.h>
#include <stdio.h>
#include "networkzn1.h"
#include "networkzn1_data.h"
int main(int argc,char **argv) {
 assert(argc==3);static ai_u64 arena[(AI_NETWORKZN1_DATA_ACTIVATIONS_SIZE+7)/8];
 const ai_handle activation[]={AI_HANDLE_PTR(arena)};ai_handle network=AI_HANDLE_NULL;
 ai_error e=ai_networkzn1_create_and_init(&network,activation,NULL);assert(!e.type);
 ai_u16 ni=0,no=0;ai_buffer *in=ai_networkzn1_inputs_get(network,&ni),*out=ai_networkzn1_outputs_get(network,&no);
 assert(ni==2&&no==1);FILE *fi=fopen(argv[1],"rb"),*fo=fopen(argv[2],"wb");assert(fi&&fo);
 float x[150],y[6];unsigned count=0;
 while(fread(x,sizeof(x),1,fi)==1) {
  for(unsigned i=0;i<2;i++) { unsigned n=AI_BUFFER_SIZE_UNPAD(&in[i]);assert(n==25||n==125);in[i].data=AI_HANDLE_PTR(x+(n==125?25:0)); }
  out[0].data=AI_HANDLE_PTR(y);assert(ai_networkzn1_run(network,in,out)==1);assert(fwrite(y,sizeof(y),1,fo)==1);count++;
 }
 assert(count==256);fclose(fi);fclose(fo);ai_networkzn1_destroy(network);return 0;
}
''')
    root=Path('deployment/H7_RL');gen=root/'X-CUBE-AI/App'
    sources=[harness]+[gen/f for f in ('networkzn1.c','networkzn1_data.c','networkzn1_data_params.c')]
    command=[a.cc,'-O2','-std=c11','-I'+str(gen),'-I'+str(a.st_host/'include')]+list(map(str,sources))+['-Wl,--start-group']+[str(a.st_host/'lib/static'/lib) for lib in ('libruntime.a','libst_cmsis_nn.a','libcmsis-nn.a')]+['-Wl,--end-group','-lm','-o',str(a.out/'network_check')]
    (a.out/'build-command.json').write_text(json.dumps(command,indent=2))
    result=subprocess.run(command,capture_output=True,text=True);(a.out/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
    subprocess.run([str(a.out/'network_check'),str(a.out/'inputs.bin'),str(a.out/'h7-generated.bin')],check=True)
    actual=np.fromfile(a.out/'h7-generated.bin',np.float32).reshape(-1,6)
    np.testing.assert_allclose(actual,y,atol=1e-4,rtol=1e-4)
    report=dict(checkpoint_sha256=EXPECTED,onnx_sha256=sha(a.out/'model/policy.onnx'),samples=256,
        onnx_max_abs=float(np.max(np.abs(z-y))),h7_generated_max_abs=float(np.max(np.abs(actual-y))),
        source_sha256={str(s):sha(s) for s in sources[1:]},ok=True,scope='actual generated C desktop runtime; no board or behavior validation')
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
