#!/usr/bin/env python3
"""Evaluate already-saved preserved BC checkpoints; never starts training."""
import argparse,json,subprocess,sys
from pathlib import Path


def select(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for iteration in range(500,5001,500):
        checkpoint=args.run/f'bc_{iteration}.pt'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        target=args.final_evaluation if iteration==5000 else args.output/f'iter-{iteration:04d}'
        if iteration!=5000:
            command=[sys.executable,'scripts/agent/eval_bc.py','--dataset',str(args.dataset),
                '--corrections',str(args.corrections),'--pose-dataset',str(args.pose_dataset),
                '--checkpoint',str(checkpoint),'--output',str(target)]
            with (args.output/f'eval-{iteration:04d}.log').open('w') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads((target/'result.json').read_text())
        assert result['completed'] and result['optimizer_updates']==0 and result['independent_student_control']
        assert result['actor_updates']==15000+iteration and result['num_envs']==40
        output=args.output/f'comparison-{iteration:04d}.json'
        command=[sys.executable,'scripts/agent/assess_bc_closed_loop.py','--run',str(target),
            '--previous',str(args.previous),'--checkpoint',str(checkpoint),'--expected-updates',str(15000+iteration),
            '--dataset',str(args.dataset),'--output',str(output)]
        with (args.output/f'assess-{iteration:04d}.log').open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        checked=json.loads(output.read_text());assert checked['artifact_and_metric_checks_passed']
        metrics={}
        for name in ('last5s_mean_speed_m_s','last5s_xy_excursion_m','last5s_leg_velocity_rms_rad_s','last5s_mean_abs_yaw_rate_rad_s'):
            values=[r[name] for r in result['rows'] if r[name] is not None]
            metrics[name+'_max']=max(values,default=None)
            metrics[name+'_mean']=sum(values)/len(values) if values else None
        row=dict(iteration=iteration,total_actor_updates=15000+iteration,checkpoint=str(checkpoint),evaluation=str(target),
            checkpoint_sha256=result['checkpoint_sha256'],strict_rise=result['strict_rise'],final_balance=result['final_balance'],
            qualified=result['qualified'],full_20s=checked['full_20s'],no_balance_regression=checked['no_rise_or_balance_regression'],
            metrics=metrics)
        row['preserves_balance_and_smoothness']=bool(row['no_balance_regression'] and row['final_balance']==40 and
            row['full_20s']==40 and metrics['last5s_leg_velocity_rms_rad_s_max']<=.02 and
            metrics['last5s_mean_abs_yaw_rate_rad_s_max']<=.02)
        rows.append(row)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    eligible=[r for r in rows if r['preserves_balance_and_smoothness']]
    # 平衡及平滑不退化优先，再比较停车通过数和平均速度；不修改网络或训练预算。
    chosen=min(eligible,key=lambda r:(-r['qualified'],r['metrics']['last5s_mean_speed_m_s_mean'],r['iteration'])) if eligible else None
    result=dict(completed=True,optimizer_updates=0,all_saved_checkpoints=rows,selected=chosen,
        selection='All40 prior balance and smoothness retained, then qualified count, mean speed, earliest iteration',
        scope='Same40 acceptance cohort also used for checkpoint selection; not new unseen-pose generalization. No model updated.')
    (args.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset','corrections','pose-dataset','previous','final-evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    select(p.parse_args())
