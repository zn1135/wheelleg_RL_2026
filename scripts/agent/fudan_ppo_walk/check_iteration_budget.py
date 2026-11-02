"""Exercise the real runner stop boundary with fake collection/update, no physics."""
import argparse
from contextlib import ExitStack,redirect_stdout
import io,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import torch
from ct_mjlab.fudan_ppo_walk import runner as module
from ct_mjlab.fudan_ppo_walk import __main__ as cli


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    def forbidden(*args,**kwargs):raise AssertionError('Real physics/Adam step is forbidden')
    results=[]
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        for name in ('mj_step','mj_step1','mj_step2'):stack.enter_context(patch.object(mujoco,name,forbidden))
        stack.enter_context(patch.object(torch.optim.Adam,'step',forbidden))
        obs=torch.zeros(4,25);critic=torch.zeros(4,141);history=torch.zeros(4,125)
        reward=torch.ones(4);done=torch.zeros(4,dtype=torch.bool);actions=torch.zeros(4,6)
        info={'terms':{name:(reward if name=='tracking_lin_vel' else torch.zeros_like(reward)) for name in module.WEIGHTS}}
        for limit in (1,3,3000,None):
            r=object.__new__(module.FudanRunner);r.log_dir=Path(temporary)/str(limit);r.log_dir.mkdir()
            r.iteration=r.transitions=r.primary_updates=r.encoder_updates=0;r.partial_update=False
            r.resume_from=None
            r.env=SimpleNamespace(num_envs=4,speed_limit=1.,step=lambda action:(obs,critic,reward,done,info,history))
            counts=dict(act=0,update=0)
            def act(*args):counts['act']+=1;return actions
            def update():
                counts['update']+=1;r.primary_updates+=20;r.encoder_updates+=20
                return [0.,0.,0.,0.]
            r.alg=SimpleNamespace(act=act,process_env_step=lambda *args:None,compute_returns=lambda *args:None,
                update=update,learning_rate=.001)
            r.policy=SimpleNamespace(encode=lambda value:torch.zeros(4,3))
            r.writer=SimpleNamespace(add_scalar=lambda *args:None,flush=lambda:None,close=lambda:None)
            def save(name):
                path=r.log_dir/name;path.write_text(str(r.iteration));return path
            r.save=save
            with patch.object(module,'initialize_rollout',return_value=(obs,critic,history)),redirect_stdout(io.StringIO()):
                report=r.learn_until(float('inf') if limit is not None else time.time()+1,max_iterations=limit)
            expected=limit or 0
            assert report['iterations']==counts['update']==expected
            assert counts['act']==expected*48 and report['transitions']==expected*48*4
            assert report['primary_updates']==report['encoder_updates']==20*expected
            assert not report['partial_update'] and (r.log_dir/'model_final.pt').read_text()==str(expected)
            assert report['stopped_by']==('iteration_budget' if limit else 'fixed_wall_clock_budget')
            results.append(dict(iteration_budget=limit,loop_iterations=expected,collection_calls=counts['act'],passed=True))
        recorded={}
        class EntryRunner:
            def __init__(self,*args,**kwargs):self.iteration=0
            def learn_until(self,deadline,max_iterations=None):
                recorded.update(deadline=deadline,max_iterations=max_iterations);return {}
        fake_env=SimpleNamespace(close=lambda:None)
        with patch.object(cli,'FudanEnv',return_value=fake_env),patch.object(cli,'FudanRunner',EntryRunner),patch.object(cli,'snapshot'),redirect_stdout(io.StringIO()):
            cli.main(['train','--num-envs','512','--log-dir',str(Path(temporary)/'cli'),
                '--precheck','artifacts/fudan-reset-v2-20261101/precheck-v2-all-inputs/report.json','--iterations','3000'])
        assert recorded['deadline']==float('inf') and recorded['max_iterations']==3000
    report=dict(passed=True,physics_steps=0,optimizer_updates=0,network_forwards=0,
        scope='Real loop/CLI using fake collection and update; not behavioral validation',cases=results,
        cli_iteration_budget=3000,legacy_deadline_supported=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
