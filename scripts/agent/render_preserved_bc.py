#!/usr/bin/env python3
"""Render preserved BC cases, prioritizing actual loss of balance over parking-only failures."""
import argparse,json,subprocess,sys
from pathlib import Path
from check_lqr_student_closed_loop import digest


def render(args):
    source=args.run/'result.json';report=json.loads(source.read_text())
    assert report['trace_sha256']==digest(args.run/'trace.npz')
    args.output.mkdir(parents=True,exist_ok=False)
    # 仅改变渲染选择顺序，逐环境结果和真实轨迹均保持；原验收报告不写回。
    report['rows']=sorted(report['rows'],key=lambda row:(row['final_balance_v31'],row['env']))
    report['render_only_source_report_sha256']=digest(source)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'trace.npz').symlink_to((args.run/'trace.npz').resolve())
    subprocess.run([sys.executable,'scripts/agent/render_bc.py','--run',str(args.output),'--dataset',str(args.dataset)],check=True)
    path=args.output/'video.json';video=json.loads(path.read_text())
    video.update(source_run=str(args.run),source_report_sha256=digest(source),
        selection_override='Order unchanged per-env results by final-balance failure first, then original env; renderer then selects distinct failed and balanced-unqualified cases',
        selector_sha256=digest(Path(__file__)))
    path.write_text(json.dumps(video,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    render(p.parse_args())
