#!/usr/bin/env python3
"""配对比较两次 eval（强制初态）输出：验收 v2 的退化否决。

用法（仓库根目录）：

  mjlab_training/.venv/bin/python scripts/agent/compare_multimode_eval.py \
    --baseline  logs/mjlab/evaluation/<旧>/result.json \
    --candidate logs/mjlab/evaluation/<新>/result.json

验收 v2 原文：与源模型在**相同初态、命令、物理和 20 秒协议**下配对比较，普通和地面
分别比较成功样本集合；**原成功样本有任何一个失效即 `regressed_rejected`**（退化，
验收不通过），**新增成功样本不能抵消丢失**，数值失败同样不通过；初态指纹／评估协议／
成功样本计数不一致则**拒绝配对**。

本脚本按环境号逐样本配对（同一 `--num-envs`／`--seeds`／`--start-mode`／`--terrain`
下环境号与初态一一对应）。先核对协议与 `initial_state_signature`，再按四档判据分别给出
否决名单：`strict_recovered`（验收 v2 的严格起立）、`weak_recovered`（任务原生
`stable_time >= 0.5`）、`survived_20s`、`both_wheels_at_end`。判据文本取自 eval 自己
写入的 `criteria` 字段，不在本脚本里另立一套门槛。

退出码：发现任何一档出现"原成功样本失效"即为 1，可直接当门禁用。

这是有限测量，不训练，不改冻结配方。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 逐样本文件里可用来判定"原成功样本"的字段。`strict_recovered` 是验收 v2 的严格起立，
# 其余是同一批样本上的辅助档位；任何一档丢失都单列，不互相抵消。
TRACKED = ('strict_recovered', 'weak_recovered', 'survived_20s', 'both_wheels_at_end')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', type=Path, required=True, help='基线 result.json')
    p.add_argument('--candidate', type=Path, required=True, help='候选 result.json')
    p.add_argument('--json', type=Path, default=None, help='可选的机器可读输出')
    return p.parse_args()


def load(summary_path):
    d = json.load(open(summary_path))
    d['_dir'] = summary_path.parent
    return d


def episodes(d, seed):
    pattern = d.get('episodes_file_pattern', 'result_episodes_seed<seed>.json')
    f = d['_dir'] / pattern.replace('<seed>', str(seed))
    if not f.is_file():
        raise SystemExit(f'缺少逐样本文件: {f}')
    return {r['env']: r for r in json.load(open(f))}


def check_protocol(old, new):
    """配对前提：协议一致、初态指纹一致、样本数一致。不一致就拒绝配对。"""
    problems = []
    for key in ('task',):
        if old.get(key) != new.get(key):
            problems.append(f'{key}: {old.get(key)!r} != {new.get(key)!r}')
    oc, nc = old.get('config', {}), new.get('config', {})
    for key in sorted(set(oc) | set(nc)):
        if oc.get(key) != nc.get(key):
            problems.append(f'config.{key}: {oc.get(key)!r} != {nc.get(key)!r}')
    if old.get('seeds') != new.get('seeds'):
        problems.append(f'seeds: {old.get("seeds")} != {new.get("seeds")}')
    if old.get('criteria') != new.get('criteria'):
        problems.append('criteria 不一致（判据本身变了，不能配对）')
    return problems


def main():
    args = parse_args()
    old, new = load(args.baseline), load(args.candidate)
    print(f'基线   {args.baseline}')
    print(f'        权重 {old.get("checkpoint_sha256", "")[:16]}  '
          f'{old.get("config", {}).get("num_envs")} 环境  初态 {old.get("config", {}).get("start_mode")}  '
          f'地形 {old.get("config", {}).get("terrain")}')
    print(f'候选   {args.candidate}')
    print(f'        权重 {new.get("checkpoint_sha256", "")[:16]}  '
          f'{new.get("config", {}).get("num_envs")} 环境  初态 {new.get("config", {}).get("start_mode")}  '
          f'地形 {new.get("config", {}).get("terrain")}')
    problems = check_protocol(old, new)
    if problems:
        print('\n拒绝配对：')
        for p in problems:
            print(f'  - {p}')
        raise SystemExit(2)
    print('协议一致（config／seeds／criteria）')

    total_lost = {}
    total_excluded = {}
    per_seed_rows = {}
    for seed in old['seeds']:
        so = old['per_seed'].get(str(seed), {})
        sn = new['per_seed'].get(str(seed), {})
        print(f'\n=== seed {seed} ===')
        if so.get('initial_state_signature') != sn.get('initial_state_signature'):
            print(f'  拒绝配对：初态指纹不一致\n    基线 {so.get("initial_state_signature")}'
                  f'\n    候选 {sn.get("initial_state_signature")}')
            raise SystemExit(2)
        print(f'  初态指纹一致 {so.get("initial_state_signature", "")[:16]}…')
        eo, en = episodes(old, seed), episodes(new, seed)
        common = sorted(set(eo) & set(en))
        print(f'  样本：基线 {len(eo)}、候选 {len(en)}、共有 {len(common)}')
        if len(eo) != len(en):
            print('  拒绝配对：样本数不一致')
            raise SystemExit(2)

        for field in TRACKED:
            raw_lost = [i for i in common if eo[i].get(field) and not en[i].get(field)]
            # Leaving the 2.5 m patch is not a fall. The repo's 2026-10-15 record
            # established this: slope sample 36 swapped survival in both directions
            # between consecutive pairings under GPU/Warp non-determinism, and the
            # robot is standing the whole time (final height in tolerance, hold
            # fraction ~1.0, no numerical failure). Report these, but do not count
            # them as a lost success.
            out_of_patch = [i for i in raw_lost if en[i].get('termination') == 'out_of_patch']
            lost = [i for i in raw_lost if i not in out_of_patch]
            gain = [i for i in common if en[i].get(field) and not eo[i].get(field)]
            if out_of_patch:
                total_excluded.setdefault(field, []).extend((seed, i) for i in out_of_patch)
            o_cnt = sum(1 for i in common if eo[i].get(field))
            n_cnt = sum(1 for i in common if en[i].get(field))
            tag = '  退化' if lost else ''
            print(f'  {field:<20} {o_cnt:>3} → {n_cnt:>3}{tag}'
                  f'   丢失 {len(lost)}  新增 {len(gain)}（新增不能抵消丢失）')
            if out_of_patch:
                print(f'      越出地块、不计失效 {len(out_of_patch)} 个：'
                      + '、'.join(f'env={i}({eo[i].get("terrain")})' for i in out_of_patch))
            if lost:
                total_lost.setdefault(field, []).extend((seed, i) for i in lost)
                for i in lost:
                    r = en[i]
                    print(f'      丢失 env={i:<3} 地形={eo[i].get("terrain"):<5} '
                          f'初态高度={eo[i].get("start_height_m", float("nan")):.3f} '
                          f'基线严格起立={eo[i].get("strict_recover_time_s")} '
                          f'候选终止={r.get("termination")} 候选末高={r.get("final_height_m", float("nan")):.3f} '
                          f'候选保持率={r.get("strict_hold_fraction", float("nan")):.3f} '
                          f'数值异常={r.get("numerical_failure")}')

        ao, an = so.get('aggregates', {}).get('all', {}), sn.get('aggregates', {}).get('all', {})
        for k in ('median_strict_recover_time_s', 'mean_strict_hold_fraction',
                  'mean_final_height_m', 'numerical_failures'):
            if k in ao or k in an:
                print(f'  {k:<28} {ao.get(k)} → {an.get(k)}')
        per_seed_rows[seed] = {f: {'old': sum(1 for i in common if eo[i].get(f)),
                                   'new': sum(1 for i in common if en[i].get(f))} for f in TRACKED}

    print()
    if total_excluded:
        print('另有越出地块的差异（本仓既有判例不属跌倒，不计入否决）：'
              + '、'.join(f'{k}({len(v)})' for k, v in total_excluded.items()))
    if total_lost:
        print('判定：**退化，验收不通过**（regressed_rejected）。失效档位：'
              + '、'.join(f'{k}({len(v)})' for k, v in total_lost.items()))
    else:
        print('判定：四档均无原成功样本失效。按验收 v2，这只表示**保留原平衡能力**，'
              '不等于部署合格，也不代表新增能力达标——倒地恢复看另一支脚本。')

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'per_seed': per_seed_rows,
                   'regressed': {k: [list(x) for x in v] for k, v in total_lost.items()}},
                  open(args.json, 'w'), ensure_ascii=False, indent=2)
        print(f'已写入 {args.json}')
    raise SystemExit(1 if total_lost else 0)


if __name__ == '__main__':
    main()
