"""Grade paired replay reports; never trains, modifies models, or overwrites history."""
import argparse
import hashlib
import json
from pathlib import Path

from ct_mjlab.wheel_balance_assessment import paired_acceptance


def read(path):
    return json.loads(path.read_text()), dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--candidate', action='append', required=True, help='CASE=EVALUATION_JSON')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, provenance = read(args.source)
    results = {}
    for value in args.candidate:
        name, path = value.split('=', 1)
        if name in results:
            p.error('Duplicate candidate name')
        report, origin = read(Path(path))
        grade = paired_acceptance(report, source)
        for pose, row in grade['poses'].items():
            samples = report['poses'][pose].get('balance_diagnostics', {}).get('samples', [])
            row['lost_sample_diagnostics'] = [s for s in samples if s['sample'] in row['lost_source_samples']]
        results[name] = dict(input=origin, completed_updates=report['completed_updates'], **grade)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'acceptance.json').write_text(json.dumps(dict(source=provenance, learning=False,
        cases=results), ensure_ascii=False, indent=2) + '\n')
    lines = ['# 双轮平衡配对验收', '',
        '双轮必须实际接地并承重、机身近水平、无明显非轮支撑且静止稳定。不限制腿关节角；高度跟踪单独验收。',
        '最后5秒至少95%的控制步同时满足物理条件。原成功样本失效即判退化；新增成功不抵消丢失。',
        '“保留原能力”仅是相对基线的判定，不代表所有初态平衡或达到目标高度。', '',
        '| 组 | 普通平衡 | 地面平衡 | 原地面成功样本丢失 | 新增地面成功 | 判定 | 全样本高度达标 |',
        '|---|---:|---:|---|---|---|---|']
    for name, grade in results.items():
        normal, ground = grade['poses']['normal'], grade['poses']['ground']
        verdict = {'regressed_rejected': '退化，验收不通过', 'retained': '保留原平衡能力',
                   'numerical_failure_rejected': '数值失败，验收不通过'}[grade['verdict']]
        lines.append(f"| {name} | {normal['balance']}/{source['num_envs']} | {ground['balance']}/{source['num_envs']} | "
                     f"{ground['lost_source_samples']} | {ground['gained_source_samples']} | {verdict} | "
                     f"{'通过' if grade['all_samples_height_pass'] else '未通过'} |")
    for name, grade in results.items():
        for pose, row in grade['poses'].items():
            for sample in row['lost_sample_diagnostics']:
                lines += ['', f"## {name} / {pose} / 样本{sample['sample']}退化",
                    f"最后5秒平均轮支撑：{sample['wheel_n']} N；非轮支撑：{sample['nonwheel_n']:.2f} N。",
                    f"未持续满足的条件：{', '.join(sample['failed_checks']) or '组合条件未同时持续满足'}。"]
    (args.output / 'acceptance.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({name: g['verdict'] for name, g in results.items()}), flush=True)


if __name__ == '__main__':
    main()
