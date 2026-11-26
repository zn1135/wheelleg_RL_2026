"""Resolve historical environment module names to their canonical robot package."""
import importlib
import importlib.abc
import importlib.util
import sys

_PACKAGES = (
    "chuanliantui_split", "chuanliantui_stand", "chuanliantui_stand_v2",
    "fudan_ppo_walk", "fudan_ppo_step", "fudan_ppo_jump",
)


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        return importlib.import_module(self.canonical)

    def exec_module(self, module):
        module.__spec__ = importlib.util.find_spec(self.canonical)


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for package in _PACKAGES:
            old = f"ct_mjlab.envs.{package}"
            if fullname == old or fullname.startswith(old + "."):
                canonical = "ct_mjlab.envs.chuanliantui." + fullname.removeprefix("ct_mjlab.envs.")
                spec = importlib.util.find_spec(canonical)
                if spec is None:
                    return None
                return importlib.util.spec_from_loader(
                    fullname, _LegacyLoader(canonical),
                    is_package=spec.submodule_search_locations is not None,
                )
        return None


sys.meta_path.insert(0, _LegacyFinder())
