"""Wheel-to-ground load bearing, independent of reset labels and leg angles."""
import math
import torch
from .recovery_goals import Curriculum, STAGES

PARAMETERS = dict(tilt_deg=15., wheel_contact_n=5., minimum_each_weight_fraction=.1,
    minimum_total_weight_fraction=.8, maximum_total_weight_fraction=1.2,
    maximum_nonwheel_weight_fraction=.05, base_contact_n=1., linear_speed_m_s=.15,
    angular_speed_rad_s=.5, height_tolerance_m=.03, success_hold_s=.5,
    ready_fraction=2 / 3, curriculum_hold_s=2., minimum_updates=100)


def predicates(*, gravity, wheel_support, wheel_contact, nonwheel_support, robot_weight,
               base_contact, linear_velocity, angular_velocity, height, target_height, valid):
    p = PARAMETERS
    finite = (torch.isfinite(gravity).all(-1) & torch.isfinite(wheel_support).all(-1)
              & torch.isfinite(base_contact) & torch.isfinite(nonwheel_support)
              & torch.isfinite(robot_weight) & (robot_weight > 0.)
              & torch.isfinite(linear_velocity).all(-1) & torch.isfinite(angular_velocity).all(-1)
              & torch.isfinite(height) & torch.isfinite(target_height) & valid)
    righted = finite & (-gravity[:, 2] >= math.cos(math.radians(p['tilt_deg'])))
    minimum_each = (robot_weight * p['minimum_each_weight_fraction']).clamp_min(p['wheel_contact_n'])
    total = wheel_support.sum(-1)
    loaded = (wheel_contact.all(-1) & (wheel_support >= minimum_each[:, None]).all(-1)
              & (total >= robot_weight * p['minimum_total_weight_fraction'])
              & (total <= robot_weight * p['maximum_total_weight_fraction'])
              & (nonwheel_support <= robot_weight * p['maximum_nonwheel_weight_fraction']))
    supported = righted & loaded & (base_contact <= p['base_contact_n'])
    balanced = (supported & (linear_velocity.norm(dim=-1) <= p['linear_speed_m_s'])
                & (angular_velocity.norm(dim=-1) <= p['angular_speed_rad_s']))
    tracking = balanced & ((height - target_height).abs() <= p['height_tolerance_m'])
    return dict(righting=righted, wheel_balance=balanced, height_tracking=tracking,
                supported_upright=supported, wheel_load_bearing=finite & loaded)
