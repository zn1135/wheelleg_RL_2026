#!/usr/bin/env python3
"""配对比较两次中途击倒评估：默认用结束前连续 1 秒的最终平衡（v3）。

旧产物不含 final_balanced 时拒绝按新口径配对；--criterion legacy-v2 可显式读取历史口径。
以下持续平衡的末 5 秒说明仅适用于 legacy-v2。

用法（仓库根目录）：

  mjlab_training/.venv/bin/python scripts/agent/compare_multimode_knockdown.py \
    --baseline  logs/mjlab/evaluation/<旧>/result.json \
    --candidate logs/mjlab/evaluation/<新>/result.json \
    [--exclude-deep-tilt-deg 150]

`--exclude-deep-tilt-deg` 把**任一侧**末态倾角 ≥ 该值的样本排除在能力人口之外，用于
「需求不含深倒扣」时的重算；被排除的样本单独列出，其内部的退化／增益不计入结论。

两次评估必须用同一套参数（seed、强度档、击倒时刻、环境数、窗口、判据），否则逐样本
配对没有意义；脚本先核对协议，不一致直接拒绝（退出码 2）。

**能力人口 = 两次冲量后都确实倒地的样本。** 只有两次都被弄倒，才谈得上"恢复得好不好"。
若候选这次压根没被弄倒（它扛住了冲量），那是扰动没打翻，不是恢复失败，单独计数。
先前版本把 `sustained(new) == 'no_fall'` 当成"未达标"，于是 33 个"扛住冲量"的样本被
错误地算进了退化名单，报出"退化 42 个"这种混了假货的结论。

两个口径分开报（验收 v2 要求三类结果分开）：

* **曾恢复** = `recovered`，冲量后连续 0.5 秒满足组合条件。验收 v2 明确写着"曾连续
  0.5 秒成功仅表示曾经恢复，不能替代最终稳定平衡"，所以它是能力标记，不是验收通过。
* **持续平衡** = `last5_pure_v2_fraction >= 0.95`（纯 v2，不含高度 ±3 cm）。

退化否决按文档：原成功样本（持续平衡达标）在候选里失效即判**退化**，新增成功不能抵消
丢失。末 5 秒未被观测到的样本单独计数，不当作失败。

这是有限测量，不训练，不改冻结配方。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUSTAINED_MIN = .95
TERRAIN = {0: '平地', 1: '斜坡', 2: '台阶', 3: '楼梯'}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--criterion', choices=('final', 'legacy-v2'), default='final',
                   help='默认最终连续 1 秒；legacy-v2 只用于复查历史产物')
    p.add_argument('--json', type=Path, default=None, help='可选的机器可读输出')
    p.add_argument('--exclude-deep-tilt-deg', type=float, default=None,
                   help='把任一侧末态倾角 >= 该值的样本排除在能力人口之外（需求不含深倒扣时用）；'
                        '被排除的样本单独列出，其内部的退化/增益不计入结论')
    return p.parse_args()


def load(path):
    return json.load(open(path))


def check_protocol(old, new):
    """配对前提：两次评估必须是同一套参数，否则键仍对得上却会配出错结论。"""
    problems = []
    if old.get('task') != new.get('task'):
        problems.append(f'task: {old.get("task")!r} != {new.get("task")!r}')
    oc, nc = old.get('config') or {}, new.get('config') or {}
    for k in sorted(set(oc) | set(nc)):
        if oc.get(k) != nc.get(k):
            problems.append(f'config.{k}: {oc.get(k)!r} != {nc.get(k)!r}')
    if old.get('criteria') != new.get('criteria'):
        problems.append('criteria 不一致（恢复判据本身变了，不能配对）')
    return problems


def key(r):
    return (r['seed'], r['magnitude'], r['knockdown_at_s'], r['env'])


def fell(r):
    return bool(r.get('fell'))


def sustained(r, criterion):
    """倒地后按所选判据比较；None 仅表示历史末5秒没有观测。"""
    if not fell(r):
        return 'no_fall'
    if criterion == 'final':
        return r['final_balanced']
    f = r.get('last5_pure_v2_fraction')
    return None if f is None else f >= SUSTAINED_MIN


def table(rows, label, criterion):
    success_label = '最终平衡 旧→新' if criterion == 'final' else '持续平衡 旧→新'
    print(f'--- {label}   （能力口径的分母 = 两次都倒地的样本数）')
    print(f'{"分组":<10}{"样本":>5}{"两次都倒地":>11}'
          f'{"曾恢复 旧→新":>20}{success_label:>20}')
    groups = [('全部', rows)]
    groups += [(f'{m:.0f} rad/s', [p for p in rows if p['old']['magnitude'] == m])
               for m in sorted({p['old']['magnitude'] for p in rows})]
    ters = sorted({p['old'].get('terrain_kind') for p in rows
                   if p['old'].get('terrain_kind') is not None})
    groups += [(TERRAIN.get(t, str(t)), [p for p in rows if p['old'].get('terrain_kind') == t])
               for t in ters]
    for name, sub in groups:
        if not sub:
            continue
        pop = [p for p in sub if fell(p['old']) and fell(p['new'])]
        o_rec = sum(1 for p in pop if p['old']['recovered'])
        n_rec = sum(1 for p in pop if p['new']['recovered'])
        o_sus = sum(1 for p in pop if sustained(p['old'], criterion) is True)
        n_sus = sum(1 for p in pop if sustained(p['new'], criterion) is True)
        print(f'{name:<10}{len(sub):>5}{len(pop):>11}'
              f'{f"{o_rec}/{len(pop)} → {n_rec}/{len(pop)}":>20}'
              f'{f"{o_sus}/{len(pop)} → {n_sus}/{len(pop)}":>20}')


def main():
    args = parse_args()
    criterion = args.criterion
    success_label = '最终平衡' if criterion == 'final' else '历史持续平衡'
    old_d, new_d = load(args.baseline), load(args.candidate)
    problems = check_protocol(old_d, new_d)
    if criterion == 'final':
        for label, report in [('基线', old_d), ('候选', new_d)]:
            if not (report.get('criteria') or {}).get('final_balance') or any(
                    'final_balanced' not in r for r in report['episodes']):
                problems.append(f'{label}缺少最终平衡数据，需按新口径重新评估；历史复查用 --criterion legacy-v2')
    if problems:
        print('拒绝配对：')
        for p in problems:
            print(f'  - {p}')
        raise SystemExit(2)
    O = {key(r): r for r in old_d['episodes']}
    N = {key(r): r for r in new_d['episodes']}
    common = sorted(set(O) & set(N))
    both_hit = [k for k in common if O[k]['hit'] and N[k]['hit']]
    rows = [{'key': k, 'old': O[k], 'new': N[k]} for k in both_hit]
    print('当前判据：' + ('最终连续 1 秒平衡（v3）' if criterion == 'final' else '历史末 5 秒 ≥95%（v2）'))
    print(f'基线 {args.baseline}')
    print(f'候选 {args.candidate}')
    print(f'样本：基线 {len(O)}、候选 {len(N)}、共有 {len(common)}、'
          f'两边都送达冲量 {len(both_hit)}')

    deep = []
    if args.exclude_deep_tilt_deg is not None:
        thr = args.exclude_deep_tilt_deg
        deep = [p for p in rows
                if max(p['old'].get('final_tilt_deg') or 0.,
                       p['new'].get('final_tilt_deg') or 0.) >= thr]
        deep_keys = {p['key'] for p in deep}
        rows = [p for p in rows if p['key'] not in deep_keys]
        print(f'按末态倾角 >= {thr:.0f}° 排除 {len(deep)} 个样本（深倒扣不在需求内）')
        if deep:
            per_mag = {}
            for p in deep:
                per_mag[p['old']['magnitude']] = per_mag.get(p['old']['magnitude'], 0) + 1
            print('  被排除样本的强度分布：'
                  + '、'.join(f'{m:.0f} 为 {c}' for m, c in sorted(per_mag.items())))
            o_sus_deep = sum(1 for p in deep if sustained(p['old'], criterion) is True)
            n_sus_deep = sum(1 for p in deep if sustained(p['new'], criterion) is True)
            print(f'  被排除样本里：持续平衡 旧 {o_sus_deep} → 新 {n_sus_deep}')
            for p in sorted(deep, key=lambda p: p['old']['magnitude']):
                k = p['key']
                print(f'      排除 {k[1]:.0f} rad/s env={k[3]:>3} '
                      f'{TERRAIN.get(p["old"].get("terrain_kind"), "?")}  '
                      f'旧末态倾角={p["old"].get("final_tilt_deg", float("nan")):.0f}° '
                      f'新末态倾角={p["new"].get("final_tilt_deg", float("nan")):.0f}°  '
                      f'旧持续={sustained(p["old"], criterion)} 新持续={sustained(p["new"], criterion)}')
        print()

    pop = [p for p in rows if fell(p['old']) and fell(p['new'])]
    resisted = [p for p in rows if fell(p['old']) and not fell(p['new'])]
    newly_fell = [p for p in rows if not fell(p['old']) and fell(p['new'])]
    neither = [p for p in rows if not fell(p['old']) and not fell(p['new'])]
    print(f'两次都倒地（能力人口）{len(pop)}；基线倒地而候选扛住 {len(resisted)}；'
          f'基线未倒而候选倒地 {len(newly_fell)}；两次都未倒 {len(neither)}')
    unobs = [p for p in pop if sustained(p['old'], criterion) is None or sustained(p['new'], criterion) is None]
    print(f'其中平衡结果未被观测到 {len(unobs)} 个（不计入持续平衡分母，也不算失败）')
    if resisted:
        per_mag = {}
        for p in resisted:
            per_mag[p['old']['magnitude']] = per_mag.get(p['old']['magnitude'], 0) + 1
        print('  候选扛住冲量的按强度分布：'
              + '、'.join(f'{m:.0f} 为 {c}' for m, c in sorted(per_mag.items())))
    print()

    table(rows, '最终平衡对照' if criterion == 'final' else '历史持续平衡对照', criterion)
    print()

    lost = [p for p in pop if sustained(p['old'], criterion) is True and sustained(p['new'], criterion) is False]
    gain = [p for p in pop if sustained(p['new'], criterion) is True and sustained(p['old'], criterion) is False]
    lost_rec = [p for p in pop if p['old']['recovered'] and not p['new']['recovered']]
    gain_rec = [p for p in pop if p['new']['recovered'] and not p['old']['recovered']]
    print('=== 退化否决（按当前所选判据：原成功样本任一失效即判退化，新增不能抵消）===')
    print(f'{success_label}达标 → 候选失效：{len(lost)} 个（两个都倒地才计入）')
    for p in lost:
        k = p['key']
        t = TERRAIN.get(p['old'].get('terrain_kind'), '?')
        metric = (f"旧末段连续={p['old']['final_balance_hold_s']:.2f}s → 新末段连续={p['new']['final_balance_hold_s']:.2f}s"
                  if criterion == 'final' else
                  f"旧末5秒={p['old']['last5_pure_v2_fraction']:.3f} → 新末5秒={p['new']['last5_pure_v2_fraction']:.3f}")
        print(f'  seed{k[0]} {k[1]:.0f} rad/s t={k[2]:.0f}s env={k[3]:>3} {t}  '
              f'{metric}  '
              f'新末态倾角={p["new"].get("final_tilt_deg", float("nan")):.0f}°')
    print(f'历史曾恢复 → 候选失效：{len(lost_rec)} 个（v3 不作否决）；新增平衡 {len(gain)} 个、'
          f'新增曾恢复 {len(gain_rec)} 个（按文档都不能抵消上面的丢失）')
    print()
    o_sus = sum(1 for p in pop if sustained(p['old'], criterion) is True)
    n_sus = sum(1 for p in pop if sustained(p['new'], criterion) is True)
    o_rec = sum(1 for p in pop if p['old']['recovered'])
    n_rec = sum(1 for p in pop if p['new']['recovered'])
    print(f'能力人口 {len(pop)}：曾恢复 {o_rec} → {n_rec}；{success_label} {o_sus} → {n_sus}')

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'criterion': criterion, 'paired': len(rows), 'population_fell_both': len(pop),
                   'baseline_fell_candidate_resisted': len(resisted),
                   'candidate_fell_baseline_did_not': len(newly_fell),
                   'unobserved_last5': len(unobs),
                   'recovered_old': o_rec, 'recovered_new': n_rec,
                   'sustained_old': o_sus, 'sustained_new': n_sus,
                   'regressed_sustained': [list(p['key']) for p in lost],
                   'regressed_recovered': [list(p['key']) for p in lost_rec],
                   'gained_sustained': [list(p['key']) for p in gain],
                   'resisted': [list(p['key']) for p in resisted]},
                  open(args.json, 'w'), ensure_ascii=False, indent=2)
        print(f'已写入 {args.json}')
    raise SystemExit(1 if (lost or (criterion == 'legacy-v2' and lost_rec)) else 0)


if __name__ == '__main__':
    main()
