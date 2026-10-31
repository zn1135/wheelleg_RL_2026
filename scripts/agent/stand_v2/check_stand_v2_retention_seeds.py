#!/usr/bin/env python3
"""Verify matched randomization inputs and compare late-balance failures on new seeds."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


def paired_p(new_only, old_only):
    n=new_only+old_only
    return 1. if not n else min(1.,2.*sum(math.comb(n,i) for i in range(min(new_only,old_only)+1))/2**n)


def main(args):
    rows=[]
    for seed in (43,44,45):
        reports=[];episodes=[]
        for role in ("baseline","candidate"):
            folder=args.root/f"{role}-training-{seed}"
            report=json.loads((folder/"result.json").read_text())
            assert (args.root/f"{role}-{seed}.exit_code").read_text().strip()=="0"
            assert report["parameter_updates"]==0 and report["actor_and_encoder_unchanged"]
            assert report["conditions"]=="training" and report["num_envs"]==128
            assert digest(report["checkpoint"])==report["checkpoint_sha256"]
            assert all(digest(k)==v for k,v in report["sources"].items())
            with np.load(folder/"episodes.npz") as data:
                episodes.append({k:data[k].copy() for k in data.files})
            reports.append(report)
        a,b=reports
        for k in ("conditions","seed","num_envs","duration_s","physics_dt_s","policy_dt_s","samples","xml_sha256","sources"):
            assert a[k]==b[k],k
        with np.load(args.root/f"baseline-training-{seed}/initial.npz") as x,np.load(args.root/f"candidate-training-{seed}/initial.npz") as y:
            assert set(x.files)==set(y.files)
            assert all(np.array_equal(x[k],y[k]) for k in x.files)
        x,y=episodes
        for r,e in zip(reports,episodes):
            passed=e["ever_stood"] & e["survived"] & e["finite"] & (e["final_hold_steps"]>=500)
            assert np.array_equal(passed,e["passed"])
            assert r["passed"]==int(passed.sum()) and r["ever_stood"]==int(e["ever_stood"].sum())
        lx=x["ever_stood"] & ~x["passed"];ly=y["ever_stood"] & ~y["passed"]
        common=x["ever_stood"] & y["ever_stood"]
        new_only=int((common & ~lx & ly).sum());old_only=int((common & lx & ~ly).sum())
        rows.append(dict(seed=seed,n=128,initial_and_physics_equal=True,
            old_stood=a["ever_stood"],new_stood=b["ever_stood"],old_passed=a["passed"],new_passed=b["passed"],
            old_late=int(lx.sum()),new_late=int(ly.sum()),
            old_late_given_stood=float(lx.sum()/x["ever_stood"].sum()),new_late_given_stood=float(ly.sum()/y["ever_stood"].sum()),
            old_late_ids=np.flatnonzero(lx).tolist(),new_late_ids=np.flatnonzero(ly).tolist(),
            common_stood=int(common.sum()),common_old_late=int((common&lx).sum()),common_new_late=int((common&ly).sum()),
            common_new_only_late=new_only,common_old_only_late=old_only,
            common_paired_two_sided_p=paired_p(new_only,old_only),
            new_late_from_newly_stood=int((~x["ever_stood"]&ly).sum()),
            old_survived=a["survived"],new_survived=b["survived"],old_finite=a["finite"],new_finite=b["finite"]))
    keys=("n","old_stood","new_stood","old_passed","new_passed","old_late","new_late", "common_stood",
          "common_old_late","common_new_late","common_new_only_late","common_old_only_late","new_late_from_newly_stood")
    total={k:sum(r[k] for r in rows) for k in keys}
    total.update(old_late_given_stood=total["old_late"]/total["old_stood"],new_late_given_stood=total["new_late"]/total["new_stood"],
        common_paired_two_sided_p=paired_p(total["common_new_only_late"],total["common_old_only_late"]))
    result=dict(verified=True,parameter_updates=0,seeds=[43,44,45],rows=rows,total=total,
        new_late_higher_seed_count=sum(r["new_late"]>r["old_late"] for r in rows),
        all_three_seeds_new_late_higher=all(r["new_late"]>r["old_late"] for r in rows),
        scope="Three new evaluation seeds of fixed2999/3999, not independent training repetitions; never-stood and late failures remain separate",
        source_sha256=digest(__file__))
    with args.output.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
