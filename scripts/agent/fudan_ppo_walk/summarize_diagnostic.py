"""Summarize a diagnostic subset without presenting it as the formal matrix."""
import argparse,json
from pathlib import Path
import numpy as np
from .score import score


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    protocol=json.loads((a.directory/'protocol.json').read_text())
    assert protocol['mode']=='diagnostic_not_formal_acceptance'
    rows=[]
    for sample in protocol['samples']:
        path=a.directory/'traces'/(sample['id']+'.npz')
        if path.exists():
            with np.load(path) as data:row=score(sample,data)
        else:row=dict(sample_id=sample['id'],passed=False,complete=False)
        row['case']=sample['case'];rows.append(row)
    report=dict(mode='diagnostic_not_formal_acceptance',total=len(rows),
        technical_passed=sum(r['passed'] for r in rows),complete=sum(r['complete'] for r in rows),
        numerical_failures=sum(r.get('numerical_failure',False) for r in rows),samples=rows,
        final_acceptance='pending_user_video_review')
    (a.directory/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
