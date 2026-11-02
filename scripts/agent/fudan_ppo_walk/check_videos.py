"""Decode delivered videos and verify frame counts, hashes, and full nominal coverage."""
import argparse,hashlib,json
from pathlib import Path
import imageio.v2 as imageio


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('experiment',type=Path);a=p.parse_args();results=[]
    for engine in ('gpu','cpu'):
        folder=a.experiment/f'video-{engine}';index=json.loads((folder/'index.json').read_text())
        assert not index['missing']
        matrix=index['videos'][0];assert len(matrix['chapters'])==47 and all(c['complete'] for c in matrix['chapters'])
        for video in index['videos']:
            path=Path(video['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==video['sha256']
            reader=imageio.get_reader(path);meta=reader.get_meta_data();count=reader.count_frames()
            assert count==video['frames'] and abs(meta['fps']-50.)<1e-6
            if path.stem=='nominal_matrix':
                imageio.imwrite(folder/'review-frame.png',reader.get_data(200))
                reader.get_data(count-1)
            reader.close();results.append(dict(path=str(path),decoded_frames=count,fps=meta['fps'],duration_s=count/50))
    numerical=json.loads((a.experiment/'video-cpu-numerical/index.json').read_text())
    assert len(numerical)==3
    for video in numerical:
        path=Path(video['video']);assert hashlib.sha256(path.read_bytes()).hexdigest()==video['video_sha256']
        reader=imageio.get_reader(path);count=reader.count_frames();reader.get_data(count-1);reader.close()
        results.append(dict(path=str(path),decoded_frames=count,complete=False,observed_s=video['observed_s']))
    (a.experiment/'video-verification.json').write_text(json.dumps(dict(passed=True,nominal_cases_per_engine=47,results=results),indent=2)+'\n')
    print('Video verification passed: both full 47-case matrices, random failures, and three explicit numerical-termination clips.')

if __name__=='__main__':main()
