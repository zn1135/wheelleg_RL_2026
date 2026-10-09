"""Direct mjlab task factories used by ct-mjlab (separate from Isaac tasks)."""
TASKS = {'chuanliantui': 'ct_mjlab.environment:ChuanliantuiEnv'}

def make_task(name, **kwargs):
    import importlib
    module, cls = TASKS[name].split(':')
    return getattr(importlib.import_module(module), cls)(**kwargs)
