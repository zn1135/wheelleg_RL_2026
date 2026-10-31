#!/usr/bin/env python3
"""Audit the detailed rerun, contact kinematics and per-condition late failures offline."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


def main(args):
    r=json.loads((args.recording/"late-analysis.json").read_text())
    base=json.loads((args.recording/"result.json").read_text())
    assert r["parameter_updates"]==base["parameter_updates"]==0 and base["actor_and_encoder_unchanged"]
    assert digest(base["checkpoint"])==base["checkpoint_sha256"]
    assert all(digest(k)==v for k,v in base["sources"].items())
    with np.load(args.original/"initial.npz") as a,np.load(args.recording/"initial.npz") as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    with np.load(args.recording/"detail.npz") as d:
        v,ids,weight=d["trace"],d["env_ids"],d["weight"]
    assert v.shape==(20000,10,35) and np.isfinite(v).all() and not v[:,:,-1].any()
    force_error=float(np.max(np.abs(v[-2500:,:,22:24]-v[-2500:,:,:2])))
    assert force_error<1e-3
    with np.load(args.recording/"contact-fixtures.npz") as d:
        velocity=d["velocity"]+np.cross(d["omega"],d["point"]-d["origin"])
        slip=np.linalg.norm(np.einsum("nij,nj->ni",d["frame"],velocity)[:,1:],axis=-1)
        slip_error=float(np.abs(slip-d["slip"]).max())
        assert slip_error<1e-6
        util=np.linalg.norm(d["force"][:,1:]/np.maximum(d["mu"],1e-8),axis=-1)/np.maximum(d["force"][:,0],1e-8)
        assert np.allclose(util,d["utilization"],atol=1e-4)
        fixtures=len(slip)
    with np.load(args.recording/"episodes.npz") as d:
        for i,row in enumerate(r["rows"]):
            tail=v[-500:,i];w=weight[i]
            conditions=dict(each_wheel=(tail[:,:2]>=max(.1*w,5.)).all(-1),
                total_support=(tail[:,:2].sum(-1)>=.8*w)&(tail[:,:2].sum(-1)<=1.2*w),
                base_clear=tail[:,2]<=1.,tilt=np.abs(tail[:,3:5]).max(-1)<=np.deg2rad(35.),
                angular_speed=tail[:,7]<=.5,finite=np.isfinite(tail).all(-1),live=tail[:,16]>.5)
            assert row["final1_condition_bad_samples"]=={k:int((~x).sum()) for k,x in conditions.items()}
            assert row["reproduced_late_failure"]==bool(d["ever_stood"][ids[i]] and not d["passed"][ids[i]])
    late=[x for x in r["rows"] if x["reproduced_late_failure"]]
    angular_only=all(all(n==0 for k,n in x["final1_condition_bad_samples"].items() if k!="angular_speed") for x in late)
    samples=[x["final1_condition_bad_samples"]["angular_speed"] for x in late]
    result=dict(verified=True,parameter_updates=0,initial_state_and14_physics_arrays_equal=True,
        checkpoint_sha256=base["checkpoint_sha256"],original_cases=10,reproduced_late_failures=len(late),
        angular_speed_is_only_final1_failure_in_reproduced_cases=angular_only,
        final1_angular_violation_duration_ms=[min(samples)*2,max(samples)*2],
        last5_max_tilt_degrees=[min(x["last5_max_tilt_deg"] for x in r["rows"]),max(x["last5_max_tilt_deg"] for x in r["rows"])],
        loaded_slip_p95_m_s=[min(x["loaded_slip_p95_m_s"] for x in r["rows"]),max(x["loaded_slip_p95_m_s"] for x in r["rows"])],
        body_contact_max_N=float(v[-2500:,:,2].max()),
        contact_normal_sensor_max_error_N=force_error,contact_kinematic_fixtures=fixtures,slip_reconstruction_max_error=slip_error,
        proposed_kernel_local_cost=dict(current=1./.25,multiplier2_denominator005=.4/.05,multiplier3_denominator005=.6/.05),
        scope="Short angular-rate excursions in a rerun, not evidence of falling or gross slip; reward proposal is analytic only, not trained",
        source_sha256=digest(__file__))
    with args.output.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording",type=Path,required=True)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
