"""Register robot-specific mjlab training tasks."""
from . import _legacy_imports
from .chuanliantui.tasks import task_registry
from importlib import import_module

import_module(".2027_wheelleg", __name__)

__all__ = ["task_registry"]
