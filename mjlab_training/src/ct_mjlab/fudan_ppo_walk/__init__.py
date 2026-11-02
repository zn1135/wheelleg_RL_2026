"""Lazy registration avoids loading the simulator through shared physical imports."""


def make_env_cfg(*,play=False):
    from .env import make_env_cfg as factory
    return factory(play=play)


def make_train_cfg():
    from .contract import make_train_cfg as factory
    return factory()


def make_runner(*args,**kwargs):
    from .runner import FudanRunner
    return FudanRunner(*args,**kwargs)
