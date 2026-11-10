"""Verify every1kHz gated output between the paired500Hz joint-controller updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sim2sim'))
from h7_output_capture import CaptureParser


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--paired',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.paired/'report.json').read_text())
    with np.load(args.paired/'paired-records.npz') as a:paired=a['kind1']
    first,last=int(paired[0,4]),int(paired[-1,4]);decoder=CaptureParser();digest=hashlib.sha256();rows=[]
    with (args.session/'stream.jsonl').open('rb') as file:
        file.seek(source['stream_window_start_byte'])
        while file.tell()<source['stream_window_end_byte']:
            line=file.readline();assert file.tell()<=source['stream_window_end_byte'];digest.update(line)
            event=json.loads(line)
            if event['type']!='data':continue
            for packet in decoder.feed(bytes.fromhex(event['data_hex'])):
                for row in packet['records']:
                    if row[1]==1 and first<=row[4]<=last:
                        rows.append([row[2],row[4],*row[6:12],row[20],row[5],row[21]])
    assert digest.hexdigest()==source['stream_window_sha256']
    raw=np.array(rows);assert np.all(np.diff(raw[:,1])==1)
    indices=np.searchsorted(paired[:,4],raw[:,1],side='right')-1
    error=raw[:,2:8]-paired[indices,6:12];wrong=np.any(error!=0,axis=1)
    flags,counts=np.unique(raw[:,8],return_counts=True)
    report=dict(scope='all gated dispatches checked; no hardware commands or physics',source_window_sha256=digest.hexdigest(),
                dispatches=len(raw),paired_updates=len(paired),all_ids_contiguous=True,
                max_held_request_error_nm=float(abs(error).max()),unexpected_request_count=int(wrong.sum()),
                fault_count=int(np.count_nonzero(raw[:,9])),non_strategy7_count=int(np.count_nonzero(raw[:,10]!=7)),
                dm_hal_failure_count=int(np.count_nonzero((raw[:,8].astype(int)&4)==0)),
                flags_histogram={str(int(f)):int(c) for f,c in zip(flags,counts)},
                period_us_range=(np.array([np.diff(raw[:,0]).min(),np.diff(raw[:,0]).max()])/1000).tolist(),
                mismatch_examples=raw[np.flatnonzero(wrong)[:10]].tolist())
    report['passed']=bool(not wrong.any() and not report['fault_count'] and not report['non_strategy7_count'] and not report['dm_hal_failure_count'])
    np.savez_compressed(args.output/'dispatches.npz',columns=np.array(['begin_ns','id','lf0','lf00','rf0','rf00','lfwheel','rfwheel','flags','fault','strategy']),values=raw)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
