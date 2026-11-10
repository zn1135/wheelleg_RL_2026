"""Verify GPU video coverage, normal speed and links to frozen checkpoint/trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    protocol = json.loads((args.recording / 'protocol.json').read_text())
    score = json.loads(args.acceptance.read_text())
    assert protocol['validation_backend'] == 'mujoco_warp_gpu'
    assert not index['missing']
    assert index['checkpoint_sha256'] == sha(args.checkpoint)
    assert index['mode'] == protocol['mode']
    samples = {sample['id']: sample for sample in protocol['samples']}
    scored = {row['sample_id']: row for row in score['samples']}
    nominal = [sample['id'] for sample in protocol['samples'] if not sample['randomized']]
    matrix = next(video for video in index['videos'] if Path(video['path']).stem == 'nominal_matrix')
    assert [chapter['sample_id'] for chapter in matrix['chapters']] == nominal
    assert index['nominal_cases_planned'] == len(nominal)
    results = []
    for video in index['videos']:
        path = Path(video['path'])
        assert sha(path) == video['sha256']
        for chapter in video['chapters']:
            sample = samples[chapter['sample_id']]
            assert chapter['complete'] and chapter['duration_s'] == sample['case']['duration']
            assert chapter['technical_pass'] == scored[sample['id']]['passed']
            assert sha(args.recording / 'traces' / (sample['id'] + '.npz')) == chapter['trajectory_sha256']
        reader = imageio.get_reader(path)
        try:
            count, fps = reader.count_frames(), reader.get_meta_data()['fps']
            assert abs(fps - 50.) < 1e-6
            assert count == video['frames'] == sum(round(c['duration_s'] * fps) for c in video['chapters'])
            reader.get_data(count - 1)
            if path.stem == 'nominal_matrix':
                # 抽取可复查画面；视频本身仍保留全部起步、刹停和失败过程。
                selections = [('zero', 'zero'), ('forward', 'forward'), ('braking', 'stop_forward'),
                              ('height', 'height_moving')]
                for label, category in selections:
                    chapter = next((c for c in video['chapters']
                                    if samples[c['sample_id']]['case']['category'] == category), None)
                    if chapter is not None:
                        offset = 8. if category.startswith('stop_') else min(4., chapter['duration_s'] / 2.)
                        imageio.imwrite(args.index.parent / ('review-' + label + '.png'),
                                        reader.get_data(round((chapter['start_s'] + offset) * fps)))
        finally:
            reader.close()
        results.append(dict(path=str(path), decoded_frames=count, fps=fps, duration_s=count / fps,
                            complete_chapters=len(video['chapters']), sha256=video['sha256']))
    random_failures = any(s['randomized'] and scored[s['id']].get('complete')
                          and not scored[s['id']]['passed'] for s in samples.values())
    assert not random_failures or any(Path(v['path']).stem == 'random_failure' for v in index['videos'])
    result = dict(passed=True, nominal_cases=len(nominal), normal_speed=True, results=results,
                  checkpoint_sha256=index['checkpoint_sha256'], physics_steps=0, optimizer_updates=0,
                  final_acceptance='pending_user_video_review',
                  scope='Decoded duration/coverage/hash checks; visual user acceptance remains separate')
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
