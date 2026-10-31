"""Plot recorded ablation height trajectories; needs no simulation runtime."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.report.read_text(encoding='utf-8'))
    names = {'baseline': 'A baseline', 'wide_height': 'B wide height',
             'mode_reward': 'C mode reward', 'recover_timeout': 'D recover timeout'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, pose in zip(axes, ('normal', 'ground')):
        axis.axhspan(record['command']['height'] - .03, record['command']['height'] + .03,
                     color='green', alpha=.09, label='Height tolerance +/-3 cm')
        axis.axhline(record['command']['height'], color='green', linestyle='--', linewidth=1)
        for group, results in record['groups'].items():
            rows = results['poses'][pose]['trajectory']
            axis.plot([r['time_s'] for r in rows], [r['mean_height_m'] for r in rows],
                      label=names[group], linewidth=1.5)
        axis.set(title=f'{pose.capitalize()} starts (32 paired samples)',
                 xlabel='Time (s)', xlim=(0, 20), ylim=(.10, .40))
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Mean root height above ground (m)')
    axes[1].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Final checkpoint model_{record["iteration_index"]}: zero velocity, target 0.30 m')
    fig.tight_layout()
    if args.output.exists():
        raise FileExistsError(args.output)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
