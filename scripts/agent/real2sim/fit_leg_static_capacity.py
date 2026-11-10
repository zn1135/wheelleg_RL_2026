"""Separate moving directional loss from configuration-dependent breakaway capacity; no integration."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    data=[inverse.load('wide',root/'inverse-preload-hypothesis'),inverse.load('original',Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
          inverse.load('slow',root/'inverse-near-slow'),inverse.load('fast',root/'inverse-near-fast')]
    candidates=[]
    for reference in (100.,130.,160.,190.):
        for slope in (1000.,2000.,3000.,4000.):
            base=dict(friction_nm=[],motor_friction_bias_nm=[],damping_nms_rad=[],armature_kg_m2=[.015795]*4,
                      spring_reference_force_n=[reference]*2,spring_stiffness_n_m=[slope]*2,spring_damping_ns_m=[50.]*2,spring_force_limit_n=[600.]*2,
                      spring_extension_stop_m=[.007]*2,spring_extension_margin_m=[.001]*2)
            residuals=[];scores=[]
            for j in range(4):
                aa=[];bb=[];observations=[]
                for d in data:
                    force=reference+slope*(d['compression']-.005)-50.*d['qvel'][:,inverse.SLIDE_INDEX]
                    load=d['target']-np.einsum('tij,tj->ti',d['spring_basis_nm_per_n'],force)
                    v=d['velocity'][:,j];mask=(abs(v)>.03)&(d['compression'][:,j//2]>.008);mask[:10]=mask[-10:]=False
                    design=np.column_stack((v>0,-(v<0).astype(float),v))
                    aa.append(design[mask]/np.sqrt(max(mask.sum(),1)));bb.append(load[mask,j]/np.sqrt(max(mask.sum(),1)))
                    observations.append((d,load[:,j],mask))
                a=np.concatenate(aa);b=np.concatenate(bb);coeff=nnls(a,b)[0]
                positive=max(float(coeff[0]),.02);negative=max(float(coeff[1]),.02)
                base['friction_nm'].append((positive+negative)/2);base['motor_friction_bias_nm'].append((positive-negative)/2);base['damping_nms_rad'].append(max(float(coeff[2]),.1))
                scores.append(float(np.mean((a@coeff-b)**2)))
                residuals.append(observations)
            for decay in (.008,.02,.05):
                extra=[];quiet_violations=[]
                for j,observations in enumerate(residuals):
                    required=[]
                    for d,load,_ in observations:
                        mask=(abs(d['velocity'][:,j])<.005)&(d['compression'][:,j//2]>.008);mask[:10]=mask[-10:]=False
                        excess=np.maximum(abs(load-base['motor_friction_bias_nm'][j])-base['friction_nm'][j],0.)
                        required.extend((excess[mask]*np.exp(d['compression'][mask,j//2]/decay)).tolist())
                    amplitude=min(max(float(np.percentile(required,95)),.1),40.) if required else .1
                    extra.append(amplitude)
                    for d,load,_ in observations:
                        mask=(abs(d['velocity'][:,j])<.005)&(d['compression'][:,j//2]>.008)
                        capacity=base['friction_nm'][j]+amplitude*np.exp(-d['compression'][:,j//2]/decay)
                        quiet_violations.append(float(np.mean(np.maximum(abs(load[mask]-base['motor_friction_bias_nm'][j])-capacity[mask],0.)**2)))
                profile=dict(base,motor_static_excess_nm=extra,motor_static_decay_m=[decay]*2,motor_static_velocity_scale_rad_s=[.02]*4)
                score=float(np.sqrt(np.mean(scores))+.3*np.sqrt(np.mean(quiet_violations)))
                candidates.append(dict(parameters=profile,inverse_selection_score=score,moving_loss_mse=scores,quiet_violation_mse=quiet_violations))
    selected=sorted(candidates,key=lambda x:x['inverse_selection_score'])[:12]
    profiles=[]
    for c in selected:
        for scale in (.5,1.,1.5):
            p=dict(c['parameters']);p['motor_static_excess_nm']=(np.asarray(p['motor_static_excess_nm'])*scale).tolist();profiles.append(p)
    (args.output/'parameters.json').write_text(json.dumps(profiles,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(scope='inverse-load development seeds only; extension-end samples excluded only from seed regression, never from GPU replay scoring',environment_steps=0,candidates=candidates,selected=selected),indent=2)+'\n')
    print(json.dumps(dict(candidates=len(profiles),best_inverse_score=selected[0]['inverse_selection_score'],best=selected[0]['parameters']),indent=2))


if __name__=='__main__':main()
