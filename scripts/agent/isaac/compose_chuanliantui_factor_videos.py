"""Compose existing paired replay videos with their recorded support forces."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GROUPS = (
    ('baseline', 'A | Original height recipe'),
    ('recovery_reference', 'B | Recovery reference'),
    ('skip_righting', 'C | Skip righting restart'),
    ('keep_height', 'D | Keep height rewards'),
    ('keep_contact', 'E | Keep contact penalty'),
    ('bearing_reward', 'F | Ground load reward'),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
    reports = {case: json.loads((args.campaign / case / 'final500/evaluation.json').read_text())
               for case, _ in GROUPS}
    assert all(r['completed_updates'] == 2500 for r in reports.values())
    record = dict(completed_updates=2500, sample=0, learning=False,
        source='existing recorded GPU qpos replay videos; composition only, no simulation',
        layout='two columns, three rows; A B / C D / E F', fps=10, poses={})
    for pose in ('ground', 'normal'):
        hashes = {r['poses'][pose]['initial_state_sha256'] for r in reports.values()}
        assert len(hashes) == 1, 'Replay initial states differ'
        paths = [args.campaign / case / 'final500' / f'comparison-{pose}.mp4' for case, _ in GROUPS]
        readers = [imageio.get_reader(str(path), 'ffmpeg') for path in paths]
        metas = [reader.get_meta_data() for reader in readers]
        assert all(m['size'] == (640, 320) and abs(m['fps'] - 10) < 1e-6 for m in metas)
        counts = [reader.count_frames() for reader in readers]
        assert len(set(counts)) == 1 and counts[0] == 201, counts
        width, height, title, footer = 640, 320, 36, 36
        tile_height = height + title + footer
        target = args.output / f'six-groups-{pose}.mp4'
        try:
            with imageio.get_writer(str(target), fps=10, codec='libx264', quality=8,
                                   macro_block_size=1, ffmpeg_params=['-threads', '2']) as writer:
                for frame in range(counts[0]):
                    canvas = Image.new('RGB', (2 * width, 3 * tile_height), '#eeeeee')
                    draw = ImageDraw.Draw(canvas)
                    for index, ((case, label), reader) in enumerate(zip(GROUPS, readers)):
                        x, y = index % 2 * width, index // 2 * tile_height
                        canvas.paste(Image.fromarray(reader.get_data(frame)), (x, y + title))
                        draw.text((x + 8, y + 5), label + ' | 2500 updates', fill='black', font=font)
                        if frame:
                            trace = reports[case]['poses'][pose]['trajectory'][frame - 1]
                            assert abs(trace['time_s'] - frame * .1) < 1e-6
                            left, right = trace['sample0_wheel_n']
                            other = trace['sample0_other_n']
                            text = f'Wheels: {left:.1f}/{right:.1f} N | Nonwheel: {other:.1f} N'
                        else:
                            text = 'Identical initial state | sample 0 | target height 0.30 m'
                        draw.text((x + 8, y + title + height + 5), text, fill='black', font=font)
                    writer.append_data(np.asarray(canvas))
                    if frame in (0, 100, 200):
                        canvas.save(args.output / f'{pose}-frame-{frame:03d}.png')
            verify = imageio.get_reader(str(target), 'ffmpeg')
            try:
                assert verify.count_frames() == 201
                meta = verify.get_meta_data()
                assert meta['size'] == (1280, 1176) and abs(meta['fps'] - 10) < 1e-6
            finally:
                verify.close()
            record['poses'][pose] = dict(frames=201, duration_s=20.1, size=[1280, 1176],
                initial_state_sha256=next(iter(hashes)),
                inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            print('COMPOSED_VIDEO ' + str(target), flush=True)
        finally:
            for reader in readers:
                reader.close()
    (args.output / 'composition.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
