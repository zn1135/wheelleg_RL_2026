"""Extract complete failed chapters and annotate their recorded500Hz contact loads."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--video-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scores = json.loads((args.recording/'diagnostic_summary.json').read_text())
    protocol = json.loads((args.recording/'protocol.json').read_text())
    samples = {sample['id']: sample for sample in protocol['samples']}
    index = json.loads(args.video_index.read_text())
    original = index['videos'][0]
    source = Path(original['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original['sha256']
    chapters = {chapter['sample_id']: chapter for chapter in original['chapters']}
    failed = [row for row in scores['samples'] if not row['passed']]
    # 先看机身触地，再看高速前进，其他失败全部保留。
    failed.sort(key=lambda row: (not row.get('fell', False), samples[row['sample_id']]['case']['category'] != 'forward'))
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    reader = imageio.get_reader(source)
    frames = 0
    results = []
    combined = args.output/'all-failures.mp4'
    with imageio.get_writer(combined, fps=50, codec='libx264', quality=7) as writer:
        for row in failed:
            sid = row['sample_id']
            chapter = chapters[sid]
            assert chapter['complete'] and row['complete']
            trace_path = args.recording/'traces'/(sid+'.npz')
            assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == chapter['trajectory_sha256']
            with np.load(trace_path) as saved:
                trace = {key: saved[key] for key in saved.files}
            duration = chapter['duration_s']
            count = round(duration*50)
            start = frames
            clip = args.output/(sid+'.mp4')
            with imageio.get_writer(clip, fps=50, codec='libx264', quality=7) as single:
                for i in range(count):
                    frame = reader.get_data(round(chapter['start_s']*50)+i)
                    canvas = Image.new('RGB', (960,800), '#17212b')
                    canvas.paste(Image.fromarray(frame), (0,0))
                    draw = ImageDraw.Draw(canvas)
                    # 每20ms视频帧显示其覆盖的全部10个2ms采样的极值，避免漏掉短时失载。
                    lo = i*10
                    hi = min(lo+10,len(trace['time']))
                    weight = trace['weight'][lo:hi]
                    wheels = trace['wheel_support'][lo:hi]
                    ratio = wheels.sum(-1)/weight
                    each_ok = bool((wheels >= np.maximum(.1*weight,5.)[:,None]).all())
                    total_ok = bool(((ratio>=.8)&(ratio<=1.2)).all())
                    active = i/50 >= duration-1
                    color = '#66dd99' if each_ok and total_ok else '#ff7777'
                    peak = float(trace['base_contact'][lo:hi].max())
                    reason = 'Startup body contact (whole-trajectory failure)' if row.get('fell') else 'Wheel-load continuity during final1s'
                    lines = [f'FAIL REASON: {reason}',
                        f'20ms loads: L={wheels[:,0].min():.1f}..{wheels[:,0].max():.1f}N  R={wheels[:,1].min():.1f}..{wheels[:,1].max():.1f}N',
                        f'Total/weight={ratio.min():.3f}..{ratio.max():.3f}; required0.8..1.2; each wheel >= {max(.1*float(weight[0]),5.):.1f}N',
                        f'Final1s scoring window: {"ACTIVE" if active else "not yet"} | wheel check: {"OK" if each_ok and total_ok else "outside limits"}',
                        f'Body-force peak in this20ms: {peak:.1f}N | source500Hz, video50fps, normal1x']
                    for j,line in enumerate(lines):
                        draw.text((12,650+28*j),line,font=font,fill=color if j in (1,2,3) and active else 'white')
                    pixels = np.asarray(canvas)
                    writer.append_data(pixels)
                    single.append_data(pixels)
                    if (sid=='diagnostic_9' and i==13) or (sid=='diagnostic_5' and i==310):
                        imageio.imwrite(args.output/(sid+'-review.png'),pixels)
                    frames += 1
            results.append(dict(sample_id=sid,case=samples[sid]['case'],start_s=start/50,duration_s=count/50,
                failure_reason=reason,complete=True,path=str(clip),sha256=hashlib.sha256(clip.read_bytes()).hexdigest()))
    reader.close()
    verified = []
    for path,expected in [(combined,frames),*[(Path(row['path']),round(row['duration_s']*50)) for row in results]]:
        video = imageio.get_reader(path)
        actual = video.count_frames()
        assert actual==expected and video.get_meta_data()['fps']==50.
        video.get_data(actual-1)
        video.close()
        verified.append(dict(path=str(path),frames=actual,fps=50,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert {row['sample_id'] for row in results} == {row['sample_id'] for row in scores['samples'] if not row['passed']}
    report = dict(scope='Existing encoded frames and recorded loads only',source_video=str(source),
        source_video_sha256=original['sha256'],checkpoint_sha256=index['checkpoint_sha256'],
        failed_samples=len(results),chapters=results,all_frames=frames,duration_s=frames/50,verified=verified,
        physics_steps=0,model_forwards=0,optimizer_updates=0,mesh_rendering_steps=0,
        note='Full first episodes, no speed changes; load extrema cover all500Hz samples in each20ms frame')
    (args.output/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(failed_samples=len(results),frames=frames,duration_s=frames/50,verified_videos=len(verified)),indent=2))


if __name__ == '__main__':
    main()
