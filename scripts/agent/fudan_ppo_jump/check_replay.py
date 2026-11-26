"""以从零初始化的临时网络验证采集、切模、视频、离线验收，不作为策略成绩。"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from unittest.mock import patch
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train import JumpRunner
from ct_mjlab.scripts.train import main as train_main
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import TASK, HEIGHTS
from ct_mjlab.utils.task_registry import ROOT
from .evaluate import collect, assess_recording, case_id


def check(output, precheck):
    output.mkdir(parents=True,exist_ok=False)
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def stop_before_learning(runner, deadline, max_iterations):
            assert runner.iteration == 0 and max_iterations == 1
            assert not runner.alg.optimizer.state and not runner.alg.extra_optimizer.state
            runner.save('scratch.pt')
            runner.writer.close()
            return dict(kind='dispatch_only', optimizer_updates=0)
        # 运行真实 CLI、预算和保存路径，在第一轮学习开始前终止。
        with patch.object(JumpRunner, 'learn_until', stop_before_learning):
            train_main(['--task', TASK, '--iterations', '1', '--num-envs', '20',
                        '--precheck', str(precheck), '--log-dir', str(tmp/'model')])
        model = tmp/'model/scratch.pt'
        recording = tmp/'gpu'
        collect(model,recording,'cuda:0')
        subprocess.run([str(ROOT/'scripts/mjlab.sh'),'accept','--task',TASK,'--recording',
                        str(recording),'--output',str(tmp/'accept')],check=True,cwd=ROOT)
        report = json.loads((tmp/'accept/acceptance.json').read_text())
        assert report['total']==40 and not report['technical_thresholds_met']
        checks = {}
        for height in HEIGHTS:
            switched, hits, finite = 0,0,0
            for i in range(20):
                with np.load(recording/(case_id(height,i)+'.npz')) as t:
                    switched += int(t['mode'].sum()==48 and not t['mode'][-1])
                    hits += int(t['trace'][:,20].any())
                    finite += int(np.isfinite(t['trace']).all())
            assert switched and hits and finite == 20
            checks[str(height)] = dict(complete_48_tick_switches=switched, bar_contact_samples=hits,
                                        finite_trajectories=finite)
        key=case_id(HEIGHTS[0],0)
        (recording/(key+'.npz')).unlink()  # 临时测试目录：模拟缺失样本。
        missing=assess_recording(recording)
        assert missing['total']==40 and missing['samples'][0]['reason']=='missing_recording'
        video=next(recording.glob('*.mp4'))
        shutil.copyfile(video,output/'routing-diagnostic.mp4')
        (output/'routing.json').write_text(json.dumps(dict(task=TASK,kind='untrained_gpu_routing_check',
            by_height=checks, total=40, paired_initial_conditions=report['cohort_sha256'],
            missing_sample_kept=True, optimizer_updates=0, behavior_assessed=False),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precheck',type=Path,required=True)
    args=parser.parse_args()
    check(args.output,args.precheck)
