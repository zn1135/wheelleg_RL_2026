"""Compare all120001 high-frequency small-amplitude C targets with an independent formula."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    root=Path(__file__).resolve().parents[3]
    source=root/'deployment/H7_RL_ros/imcalib/Algorithm/joint_chirp.c'
    compiler=os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if not compiler:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=False,c_runtime_checked=False,
            reason='Host C compiler is unavailable; ARM build does not replace host numeric comparison',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),hardware_access=False),indent=2)+'\n')
        raise SystemExit('Host C compiler unavailable; numeric comparison not run')
    with tempfile.TemporaryDirectory() as tmp:
        library=Path(tmp)/'chirp.so'
        subprocess.run([compiler,'-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC','-O2',str(source),'-lm','-o',str(library)],check=True)
        lib=ctypes.CDLL(str(library));function=lib.Joint_Chirp_Target
        floats=ctypes.c_float*4
        function.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint8)]
        function.restype=ctypes.c_uint8
        center=floats(-.5,.4,.5,-.4);actual=floats();stage=ctypes.c_uint8();axis=ctypes.c_uint8()
        actual_trace=np.empty((120001,4));labels=np.empty((120001,2),dtype=int)
        for tick in range(120001):
            assert function(tick,center,actual,ctypes.byref(stage),ctypes.byref(axis))==1
            actual_trace[tick]=actual[:];labels[tick]=[stage.value,axis.value]
        expected=np.tile(np.array(center[:]),(120001,1))
        for segment in range(4):
            ticks=np.arange(30000);local=ticks/1000.-5.
            amplitude,f0,f1=.05,2.,5.
            envelope=np.clip(np.minimum(local,20-local)/2,0,1)
            envelope=10*envelope**3-15*envelope**4+6*envelope**5
            phase=2*np.pi*(f0*local+.5*(f1-f0)/20*local**2)
            expected[segment*30000+ticks,segment%4]+=amplitude*envelope*np.sin(phase)
            assert np.all(labels[segment*30000:(segment+1)*30000]==[2,segment%4])
        error=float(abs(expected-actual_trace).max());assert error<5e-5,error
        assert np.max(abs(actual_trace[::30000]-center[:]))==0
        assert labels[-1].tolist()==[3,0]
        bad=floats(float('nan'),0,0,0)
        assert function(1,bad,actual,ctypes.byref(stage),ctypes.byref(axis))==0
        assert function(4294967295,center,actual,ctypes.byref(stage),ctypes.byref(axis))==1 and stage.value==3
    result=dict(passed=True,samples=120001,target_max_error_rad=error,segment_boundary_jump_rad=0,
        nonfinite_rejected=True,completed_does_not_restart=True,hardware_access=False,new_physics_steps=0,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limitation='Actual C trajectory check only; not hardware motion acceptance')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
