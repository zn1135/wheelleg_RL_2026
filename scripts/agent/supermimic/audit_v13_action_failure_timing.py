"""Read frozen 96-case trajectories: action exceedances and failure timing; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCES=[('SM615','supermimic-v13-baseline615-current13-seed43-20261031'),
    ('A170','supermimic-v13-midpoint-seed43-20261031'),
    ('A368','supermimic-v13-final-seed43-20261031'),
    ('A700','supermimic-v13-continuation-a700-seed43-20261101-r1'),
    *[(f'A{n}',f'supermimic-v13-r2-a{n}-seed43-20261101') for n in (1200,1700,2200)],
    *[(f'A{n}',f'supermimic-v13-r3-a{n}-seed43-20261101') for n in (3200,4200,5200)]]
GROUPS={'legs':[0,1,3,4],'wheels':[2,5],'left_wheel':[2],'right_wheel':[5]}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stats(x):
    x=np.abs(x).reshape(-1)
    if not len(x):return None
    return dict(samples=int(x.size),count_above3=int((x>3).sum()),percent_above3=float((x>3).mean()*100),
        max_abs=float(x.max()),p95_abs=float(np.quantile(x,.95)),p99_abs=float(np.quantile(x,.99)),
        percent_above={str(b):float((x>b).mean()*100) for b in (2,3,4,6,10)})


def first_time(mask,times,consecutive=1):
    hits=np.flatnonzero(np.convolve(mask.astype(int),np.ones(consecutive,dtype=int),'valid')==consecutive)
    return float(times[hits[0]]) if len(hits) else None


def extent(values):
    x=[v for v in values if v is not None]
    return dict(count=len(x),min=min(x) if x else None,median=float(np.median(x)) if x else None,max=max(x) if x else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=[]
    for label,d in SOURCES:
        directory=ROOT/'artifacts'/d;s=json.loads((directory/'summary.json').read_text())
        assert s['seed']==43 and s['num_envs']==96 and s['policy_dt_s']==.01
        inputs.append((label,directory,s))
    # Use the earliest first-episode end across every model, not survivors only.
    common=min(row['episode_end_s'] for _,_,s in inputs for row in s['summary']['per_environment'])
    output=[];csv_rows=[]
    for label,directory,s in inputs:
        path=directory/'trajectory.npz';digest=sha(path);assert digest==s['trajectory_sha256']
        rows=sorted(s['summary']['per_environment'],key=lambda r:r['env'])
        assert [r['env'] for r in rows]==list(range(96))
        with np.load(path) as z:
            values=z['values'];stride=round(s['policy_dt_s']/s['physics_dt_s'])
            # Recorder stores the same latched raw action after each physical substep.
            actions=values[:,:,slice(*s['layout']['raw_action'])]
            assert len(actions)%stride==0
            assert np.array_equal(actions.reshape(-1,stride,96,6),np.broadcast_to(actions[::stride,None],actions.reshape(-1,stride,96,6).shape))
            a=actions[::stride].copy();valid=values[::stride,:,s['layout']['valid'][0]]>.5
            t=np.arange(len(a))*s['policy_dt_s']
            if 'policy_time_s' in z:np.testing.assert_allclose(t,z['policy_time_s'],atol=1e-10,rtol=0)
            ends=np.array([r['episode_end_s'] for r in rows]);expected=t[:,None]<ends[None,:]-1e-8
            assert np.array_equal(valid,expected) and np.isfinite(a[valid]).all()
        del actions,values
        common_mask=valid & (t[:,None]<common-1e-8)
        assert common_mask.sum()==round(common/.01)*96
        per_env=[]
        for env,r in enumerate(rows):
            failed=r['episode_end_reason']=='failure';end=r['episode_end_s']
            item=dict(env=env,case=r['case'],failed=failed,end_s=end,termination_terms=r.get('termination_terms'),groups={})
            for name,ids in GROUPS.items():
                x=a[:,env][:,ids];above=(np.abs(x)>3).any(-1)&valid[:,env]
                first=first_time(above,t);persistent=first_time(above,t,10)
                windows={str(w):stats(x[valid[:,env]&(t>=end-w-1e-8)&(t<end-1e-8)]) for w in (.1,.5,1.)} if failed else None
                item['groups'][name]=dict(first_above3_s=first,first_100ms_above3_start_s=persistent,
                    lead_to_failure_s=end-first if failed and first is not None else None,
                    failure_windows=windows)
                csv_rows.append(dict(model=label,env=env,group=name,failed=failed,end_s=end,
                    first_above3_s=first,first_100ms_above3_start_s=persistent,
                    lead_to_failure_s=end-first if failed and first is not None else None,
                    last_100ms_percent=windows['0.1']['percent_above3'] if windows else None,
                    last_500ms_percent=windows['0.5']['percent_above3'] if windows else None))
            per_env.append(item)
        groups={}
        for name,ids in GROUPS.items():
            failed=[r for r in per_env if r['failed']]
            groups[name]=dict(full_first_episode=stats(a[valid][:,ids]),common_window=stats(a[common_mask][:,ids]),
                first_above3_s=extent([r['groups'][name]['first_above3_s'] for r in per_env]),
                first_100ms_above3_start_s=extent([r['groups'][name]['first_100ms_above3_start_s'] for r in per_env]),
                failures_without_exceedance=sum(r['groups'][name]['first_above3_s'] is None for r in failed),
                lead_to_failure_s=extent([r['groups'][name]['lead_to_failure_s'] for r in failed]))
        record=dict(model=label,trajectory=str(path),sha256=digest,summary_sha256=sha(directory/'summary.json'),
            failed=sum(r['failed'] for r in per_env),survived=s['summary']['survived'],balanced=s['summary']['final_balanced'],
            failure_time_s=extent([r['end_s'] for r in per_env if r['failed']]),groups=groups,per_environment=per_env)
        output.append(record)
        print(json.dumps({k:record[k] for k in ('model','failed','failure_time_s','groups')},ensure_ascii=False),flush=True)
    report=dict(common_window_s=[0,common],threshold=3.,sampling='one raw action per 10ms policy tick; strict abs(a)>3; first episode only',
        denominator='action components: four legs or two wheels; not percent of failed episodes',
        timing='action issued at tick start, failure detected at tick end; physical actuator delay is not removed',
        interpretation='Temporal association only. No action-clipping intervention; does not establish causality or an optimal limit.',
        models=output,optimizer_updates=0,simulator_steps=0,script_sha256=sha(Path(__file__)))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (args.output/'per-environment.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines=['# V13动作超限与失败时刻（只读原轨迹）','',f'统一窗口：0～{common:.2f}s；每10ms取一次原始动作。比例分母为动作分量采样数，超限为绝对值严格大于3。完整窗口仅取首回合，失败后复位数据排除。','',
        '|模型|轮超限%（完整）|腿超限%（完整）|轮超限%（统一窗口）|轮幅度最大值|失败例数|失败时间s|',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in output:
        g=r['groups'];w=g['wheels'];f=r['failure_time_s'];ft='未失败（30s）' if not f['count'] else f"{f['min']:.2f}～{f['max']:.2f}"
        lines.append(f"|{r['model']}|{w['full_first_episode']['percent_above3']:.4f}|{g['legs']['full_first_episode']['percent_above3']:.4f}|{w['common_window']['percent_above3']:.4f}|{w['full_first_episode']['max_abs']:.4f}|{r['failed']}/96|{ft}|")
    lines+=['','完整窗口长短不同，不可直接据此判定训练趋势；统一窗口用于排除时长差异。',
        '每模型左右轮分开统计、100ms连续超限、首次超限到失败间隔及失败前100/500/1000ms明细见report.json和per-environment.csv。',
        '原生终止failure与最终双轮平衡不合格是不同事件；表中的失败指前者。所有模型仅seed43名义工况。',
        '未做限幅干预、训练或新仿真，不能由相关性选定最优范围；±3仍为待验证候选。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
