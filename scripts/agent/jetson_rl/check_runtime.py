"""Exercise actual C++ CLI/serial/async logger through a PTY; GPU inference alone is a stub."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import time
import zlib
import resource
import shutil

ROOT=Path(__file__).resolve().parents[3]
def frame(kind,session,reset,seq,us,payload=b''):
 b=struct.pack('<4sBBHQIIQ',b'JRL2',2,kind,len(payload),session,reset,seq,us)+payload
 return b+struct.pack('<I',zlib.crc32(b))
def state(session,reset,seq,us,flags,ack=0,warm=0,accepted=(0.,)*6):
 b=bytearray(224);struct.pack_into('<8I',b,0,flags,0,ack,warm,ack,3000 if ack else 0,1767902,0)
 struct.pack_into('<f',b,52,-1);struct.pack_into('<f',b,104,.22352);struct.pack_into('<f',b,180,1);struct.pack_into('<6f',b,108,*accepted)
 return frame(2,session,reset,seq,us,b)
def receive(fd,buf,timeout=.1):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  if select.select([fd],[],[],.01)[0]:
   try:buf.extend(os.read(fd,4096))
   except OSError:continue
  while len(buf)>=8:
   at=buf.find(b'JRL2')
   if at<0:buf.clear();break
   if at:del buf[:at]
   if len(buf)<8:break
   n=struct.unpack_from('<H',buf,6)[0]+36
   if len(buf)<n:break
   p=bytes(buf[:n]);del buf[:n];assert zlib.crc32(p[:-4])==struct.unpack_from('<I',p,n-4)[0]
   return p
 return None

def case(exe,out,kind):
 master,slave=pty.openpty();port=os.ttyname(slave);log=out/kind
 args=[str(exe),'record' if kind=='record' else 'control','--port',port,'--log-dir',str(log),'--seconds','3']
 if kind!='record':args+=['--engine',str(out/'fake.engine')]
 def limit():
  signal.signal(signal.SIGXFSZ,signal.SIG_IGN);resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096))
 proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,"CUDA_CACHE_DISABLE":"1"},preexec_fn=limit if kind=='write_failure' else None)
 buf=bytearray();session=0;ack=0;warm=0;actions=0;stop=False;accepted=(0.,)*6
 try:
  time.sleep(.08)
  for seq in range(1,240):
   if proc.poll() is not None:break
   us=seq*10000;flags=16|(4 if session else 0)
   if session and seq>=4:flags|=3
   if kind=='record':flags=16
   if kind=='invalid_feedback' and actions>=2:flags&=~1
   os.write(master,state(session,1,seq,us,flags,ack,warm,accepted))
   p=receive(master,buf,.025 if kind!='record' else .001)
   if p:
    if kind=='record':raise AssertionError('record mode wrote to serial')
    typ=p[5]
    if typ==1:session=struct.unpack_from('<Q',p,8)[0]
    elif typ==3:
     actions+=1;this=struct.unpack_from('<I',p,20)[0]
     is_warm=struct.unpack_from('<I',p,56)[0]
     if actions==1:assert is_warm==1 and struct.unpack_from('<6f',p,32)==(0.,)*6
     else:assert is_warm==0
     if kind!='missing_ack' or actions==1:ack=this;warm=5;accepted=struct.unpack_from('<6f',p,32)
    elif typ==4:stop=True;break
   if kind=='timeout' and actions>=2:time.sleep(.13);p=receive(master,buf,.2);stop=bool(p and p[5]==4);break
   if kind=='normal' and actions>=8:proc.send_signal(signal.SIGINT);p=receive(master,buf,.2);stop=bool(p and p[5]==4);break
   time.sleep(.005)
  if kind!='record' and not stop:
   tail=receive(master,buf,.2);stop=bool(tail and tail[5]==4)
  stdout,stderr=proc.communicate(timeout=5)
  if kind=='record':assert proc.returncode==0
  else:assert actions>=1 and stop,(kind,actions,stop,stderr)
  if kind in ('timeout','missing_ack','write_failure','invalid_feedback'):assert proc.returncode!=0,(kind,stderr)
  if kind=='normal':assert proc.returncode==0,stderr
  return dict(kind=kind,actions=actions,stop_received=stop,exit=proc.returncode,stderr=stderr)
 finally:
  if proc.poll() is None:proc.kill();proc.wait()
  os.close(master);os.close(slave)

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--cxx');p.add_argument('--native-executable',type=Path);p.add_argument('--engine',type=Path);p.add_argument('--no-analysis',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 if a.native_executable:
  assert a.engine
  exe=a.native_executable
  shutil.copyfile(a.engine,a.out/'fake.engine');shutil.copyfile(str(a.engine)+'.meta',a.out/'fake.engine.meta')
 else:
  assert a.cxx
  exe=a.out/'runtime-check';command=[a.cxx,'-std=c++17','-O1','-pthread','-DJRL_CONFIG_SHA="host-test"','-Ideployment/jetson_rl/src','-Ideployment/H7_RL_ros/imcalib/common','deployment/jetson_rl/src/main.cpp','scripts/agent/jetson_rl/inference_stub.cpp','-o',str(exe)]
  r=subprocess.run(command,capture_output=True,text=True);(a.out/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
  (a.out/'fake.engine.meta').write_text('test stub only\n')
 results=[case(exe,a.out,kind) for kind in ('normal','timeout','missing_ack','write_failure','invalid_feedback','record')]
 summary=None
 if not a.no_analysis:
  spec=importlib.util.spec_from_file_location('report',ROOT/'deployment/jetson_rl/tools/report.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  summary=module.analyze(a.out/'normal',a.out/'analysis');assert summary['policy_frames']>=8 and not summary['crc_or_header_errors']
  salvaged=module.analyze(a.out/'write_failure',a.out/'failed-log-analysis');assert not salvaged['complete']
 report=dict(ok=True,scope='actual CLI/serial/logger; '+('native TensorRT' if a.native_executable else 'inference stub')+'; PTY simulated H7, no hardware control',cases=results,report=summary)
 (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
