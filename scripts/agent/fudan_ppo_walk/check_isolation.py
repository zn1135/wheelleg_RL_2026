"""Check the new task boundary without blessing unrelated historical hash differences."""
import hashlib,json,subprocess
from pathlib import Path
from ct_mjlab.envs import task_registry


def main():
    root=Path(__file__).resolve().parents[3]
    protection=json.loads((root/'docs/maintenance/20261101-sim2sim-protection.json').read_text())
    for name,digest in protection['protected_files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    unchanged=[]
    for base in ('mjlab_training/src/ct_mjlab/envs','resources','deployment'):
        paths=subprocess.check_output(['git','ls-files','-z',base],cwd=root,text=True).split('\0')
        for name in paths:
            if not name or name.endswith('envs/__init__.py'):continue
            old=subprocess.check_output(['git','show','HEAD:'+name],cwd=root)
            if (root/name).is_symlink():continue
            assert (root/name).read_bytes()==old,name
            unchanged.append(name)
    old_tasks=('chuanliantui','chuanliantui_stand','chuanliantui_stand_v2','chuanliantui_getup','chuanliantui_walk')
    assert task_registry.names()==('smp','supermimic','fudan_ppo_walk')
    for name in old_tasks:
        try:task_registry.get_cfgs(name)
        except ValueError:pass
        else:raise AssertionError(name)
    p='mjlab_training/src/ct_mjlab/supermimic/__main__.py'
    actual=hashlib.sha256((root/p).read_bytes()).hexdigest()
    head=hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=root)).hexdigest()
    migration=json.loads((root/'docs/maintenance/20261101-source-migration.json').read_text())
    approved_gpu_entry=migration['changed']['supermimic/__main__.py']['after']==actual
    assert actual==head or approved_gpu_entry
    print(json.dumps(dict(new_task_isolated=True,unchanged_tracked_physics_and_deployment=len(unchanged),
        protected_sim2sim_files=len(protection['protected_files']),old_tasks_rejected=list(old_tasks),
        unrelated_maintenance_hash_failure=dict(file=p,current_equals_head=actual==head,
            approved_global_gpu_entry=approved_gpu_entry,sha256=actual),
        physics_steps=0,optimizer_updates=0),indent=2))

if __name__=='__main__':main()
