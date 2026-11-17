"""Refine the effective swing model using full-trial output error, training speeds only."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares

from scripts.agent.real2sim.identify_bilateral_swing import load_trials, sha


C_SOURCE = r'''
#include <math.h>
void simulate(int n,double dt,const double *u,const double *p,double q,double v,double *out) {
  for(int i=0;i<n;i++) {
    out[2*i]=q;out[2*i+1]=v;
    double load=p[3]*sin(q)+p[4]*cos(q)+p[5]*sin(2*q)+p[6]*cos(2*q);
    double freev=(v+dt*(u[i]-load)/p[0])/(1+dt*p[1]/p[0]);
    double threshold=dt*p[2]/(p[0]+dt*p[1]);
    v=copysign(fmax(fabs(freev)-threshold,0),freev);q+=dt*v;
  }
}
'''


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source = args.output/'reduced.c'
    source.write_text(C_SOURCE)
    library = args.output/'reduced.so'
    compiler = os.environ.get('CC') or shutil.which('cc')
    if not compiler:
        compiler = str(Path.home()/'.local/opt/h7-host-gcc/usr/bin/x86_64-linux-gnu-gcc-13')
    command = [compiler,'-O2','-Wall','-Werror','-shared','-fPIC']
    runtime = Path(compiler).parent.parent/'lib/gcc/x86_64-linux-gnu/13'
    compiler_env = os.environ.copy()
    if runtime.is_dir():
        for tool in ('ld','as'):
            (args.output/tool).symlink_to(Path(compiler).parent/('x86_64-linux-gnu-'+tool))
        command += ['-B'+str(args.output.resolve())+'/', '-B/usr/libexec/gcc/x86_64-linux-gnu/13/',
                    '-B'+str(runtime)+'/', '-isystem',str(runtime/'include')]
        compiler_env['LD_LIBRARY_PATH'] = str(Path(compiler).parent.parent/'lib/x86_64-linux-gnu')
    subprocess.run([*command,str(source),'-lm','-o',str(library)],check=True,env=compiler_env)
    lib = ctypes.CDLL(str(library.resolve()))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.simulate.argtypes = [ctypes.c_int,ctypes.c_double,pointer,pointer,ctypes.c_double,ctypes.c_double,pointer]
    result = json.loads((args.seed/'identification.json').read_text())
    trials,_ = load_trials(args.input)
    result['inverse_regression_seed'] = result['fitted']
    result['fitted'] = {}
    result['output_refinement'] = dict(method='Full-trial output error; bounded least squares; implicit viscous/Coulomb Euler seed',
        training_levels=[1,3,5],heldout_levels_never_scored_during_optimization=[2,4],
        source_sha256=sha(Path(__file__)),c_source_sha256=sha(source),
        cpu_seed_is_not_final_acceptance=True)
    for side in ('left','right'):
        prepared = []
        for tr in trials:
            if tr['side'] != side or tr['level'] not in (1,3,5):
                continue
            t = np.arange(round(tr['t'][-1]/.001)+1)*.001
            ix = np.clip(np.searchsorted(tr['t'],t,side='right')-1,0,len(tr['t'])-1)
            u = np.ascontiguousarray(-tr['tau'][ix].sum(1))
            actual = np.column_stack([np.interp(t,tr['t'],tr[key]) for key in ('theta','reported_speed')])
            out = np.empty((len(t),2))
            prepared.append((tr,t,u,actual,out))
        def residual(p):
            parts = []
            for tr,t,u,actual,out in prepared:
                lib.simulate(len(t),.001,u,np.ascontiguousarray(p),tr['theta'][0],tr['reported_speed'][0],out)
                # 整档等权，连续摆角/速度共同约束；不按每圈重置或偷用后续状态。
                error = out[::20]-actual[::20]
                error[:,0] /= .15
                error[:,1] /= .3
                parts.append(error.ravel()/np.sqrt(len(error)))
            return np.concatenate(parts)
        seed = np.array(result['inverse_regression_seed'][side]['coefficients'])
        starts = []
        for damping in (seed[1],.05,.2):
            p = seed.copy();p[1] = damping
            starts.append((float(np.mean(residual(p)**2)),p))
        initial = min(starts,key=lambda x:x[0])[1]
        solution = least_squares(residual,initial,bounds=([.03,0,.05,-3,-3,-3,-3],[.6,2,6,3,3,3,3]),
                                 x_scale=[.2,.1,2,1,1,1,1],diff_step=1e-4,max_nfev=120,
                                 ftol=1e-8,xtol=1e-8,gtol=1e-6)
        p = solution.x
        row = dict(result['inverse_regression_seed'][side])
        row.update(coefficients=p.tolist(),effective_inertia_kg_m2=float(p[0]),
                   effective_damping_nm_s_rad=float(p[1]),positive_direction_offset_nm=float(p[2]),
                   periodic_load_nm=p[3:].tolist(),refinement_success=bool(solution.success),
                   refinement_message=solution.message,refinement_nfev=solution.nfev,
                   training_seed_objective=float(np.mean(residual(seed)**2)),
                   training_final_objective=float(np.mean(residual(p)**2)),
                   active_bounds=solution.active_mask.tolist())
        # 窗口敏感性只属于原回归种子，不能误标成输出误差拟合的置信区间。
        row['inverse_seed_window_sensitivity'] = row.pop('window_sensitivity')
        result['fitted'][side] = row
        print(side,json.dumps(row,ensure_ascii=False),flush=True)
    (args.output/'identification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
