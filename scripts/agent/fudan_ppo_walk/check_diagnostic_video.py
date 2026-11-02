"""Verify complete diagnostic chapters, source hashes, frame count and playback speed."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    protocol = json.loads((args.recording / 'protocol.json').read_text())
    assert index['mode'] == protocol['mode'] == 'diagnostic_not_formal_acceptance'
    assert not index['missing']
    video = index['videos'][0]
    chapters = video['chapters']
    assert [c['sample_id'] for c in chapters] == [s['id'] for s in protocol['samples']]
    assert len(chapters) == len(protocol['samples']) > 0 and all(c['complete'] for c in chapters)
    for chapter, sample in zip(chapters, protocol['samples'], strict=True):
        assert chapter['duration_s'] == sample['case']['duration']
        trace = args.recording / 'traces' / (sample['id'] + '.npz')
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == chapter['trajectory_sha256']
    path = Path(video['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == video['sha256']
    reader = imageio.get_reader(path)
    frames = reader.count_frames()
    fps = reader.get_meta_data()['fps']
    reader.get_data(frames - 1)
    reader.close()
    assert frames == video['frames'] == sum(round(c['duration_s'] * 50) for c in chapters)
    assert fps == 50
    result = dict(passed=True,mode=index['mode'],chapters=len(chapters),frames=frames,
        fps=fps,duration_s=frames/fps,physics_steps=0,optimizer_updates=0,
        checkpoint_sha256=index['checkpoint_sha256'],video_sha256=video['sha256'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
