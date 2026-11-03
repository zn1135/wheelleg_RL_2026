"""Read-only contact metrics; never feed observations, rewards or mode transitions."""

from ..chuanliantui.standup_mdp import base_contact_force
from .mdp import normal_wheel_support, wheel_load


BASE_CONTACT_THRESHOLD_N = 1.


def base_contact_fraction(env):
    return (base_contact_force(env) > BASE_CONTACT_THRESHOLD_N).float()


def normal_base_contact_fraction(env):
    state = env.command_manager.get_term("standup")
    return ((base_contact_force(env) > BASE_CONTACT_THRESHOLD_N) & ~state.recover).float()


def wheel_support_force(env, side):
    forces, _ = wheel_load(env)
    return forces[:, side]


def wheel_support_weight_ratio(env):
    forces, weight = wheel_load(env)
    return forces.sum(-1) / weight


def normal_wheel_support_fraction(env):
    return normal_wheel_support(env).float()
