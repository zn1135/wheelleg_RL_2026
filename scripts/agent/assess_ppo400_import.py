#!/usr/bin/env python3
"""Audit copied files, protected local changes, model assets and destination checks."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(args):
    manifest = json.loads((args.run/'manifest.json').read_text())
    assert manifest['applied'] and manifest['training_updates'] == 0
    edits = manifest.get('post_merge_edits', {})
    for row in manifest['files']:
        assert digest(ROOT/row['path']) == edits.get(row['path'], row['after_sha256']), row['path']
    for name, expected in manifest['protected_existing_changes'].items():
        assert digest(ROOT/name) == expected, name
    for row in manifest['assets']:
        assert digest(ROOT/row['path']) == row['sha256'], row['path']
    for name, expected in manifest['additional_files'].items():
        assert digest(ROOT/name) == expected, name
    for name, expected in manifest['physical_assets'].items():
        assert digest(ROOT/name) == expected, name
    entries = json.loads((args.run/'entries/result.json').read_text())
    stand = json.loads((args.run/'stand-check/result.json').read_text())
    behavior = json.loads((args.run/'assessment.json').read_text())
    assert entries['passed'] and entries['optimizer_updates'] == 0 and stand['passed'] and behavior['passed']
    actual = json.loads((args.run/'evaluation/result.json').read_text())
    rows = actual['rows']
    smooth = sum(row['full_20s'] and row['last5s_physical_leg_velocity_rms_rad_s'] <= .02 for row in rows)
    assert actual['strict_rise'] == actual['final_balance'] == smooth == 40
    video = json.loads((args.run/'video-case0/video.json').read_text())
    assert video['frames'] == 1200 and video['seconds'] == 20
    assert video['trace_sha256'][2] == actual['trace_sha256']
    assert digest(args.run/'video-case0/comparison.mp4') == video['video_sha256']
    result = dict(passed=True, optimizer_updates=0, merged_code_and_docs=len(manifest['files']),
        verified_assets=len(manifest['assets']), protected_existing_changes=len(manifest['protected_existing_changes']),
        destination_entries_passed=True, independent_stand_finite_check_passed=True,
        rise=40, final_balance=40, smooth=smooth, parking=actual['qualified'],
        xml_and_physical_assets_unchanged=True, video_frames=video['frames'],
        source_sha256=digest(Path(__file__)),
        scope='Destination file hashes, original local edits, real entry checks and cached40 complete20s replay. '
              'No training/Git/H7 changes. Parking, movement, new poses, hardware and independent CPU deployment not passed by this import.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    assess(p.parse_args())
