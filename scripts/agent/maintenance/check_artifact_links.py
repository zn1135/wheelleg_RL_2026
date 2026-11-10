"""检查项目文档的本地产物链接，分别统计已分类和未归类目标。"""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CATEGORIES = {
    'supermimic', 'fudan_ppo_walk', 'real2sim', 'imitation', 'maintenance',
    'isaac_legacy', 'legacy_mjlab', 'shared_validation', 'two_policy',
}


def main():
    checked = 0
    missing = []
    other_checked = 0
    other_missing = 0
    for document in sorted((ROOT / 'docs').rglob('*.md')):
        for line_number, line in enumerate(document.read_text().splitlines(), 1):
            for link in re.findall(r'\]\(([^)]+)\)', line):
                target = link.split('#', 1)[0].split('?', 1)[0]
                if 'artifacts/' not in target or '://' in target:
                    continue
                if not any(f'artifacts/{category}/' in target for category in CATEGORIES):
                    other_checked += 1
                    if not (document.parent / target).exists():
                        other_missing += 1
                    continue
                checked += 1
                if not (document.parent / target).exists():
                    missing.append((document, line_number, target))
    for document, line_number, target in missing:
        print(f'{document.relative_to(ROOT)}:{line_number}: {target}')
    print(f'已分类产物链接：{checked} 个，缺失 {len(missing)} 个')
    print(f'其他产物链接：{other_checked} 个，缺失 {other_missing} 个')
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
