"""Direct mjlab task factories used by ct-mjlab (separate from Isaac tasks)."""
TASKS = {'chuanliantui': 'ct_mjlab.reference_training.entry:RecoverVecEnv'}

def make_task(name, *, legacy=False, **kwargs):
    import importlib
    module, cls = ('ct_mjlab.environment:ChuanliantuiEnv' if legacy else TASKS[name]).split(':')
    return getattr(importlib.import_module(module), cls)(**kwargs)
