"""Replay-only wheel balance diagnostics and paired regression acceptance.

This module never changes training rewards, curriculum, or checkpoint manifests.
"""
import math

STANDARD = 'wheel-balance-paired-acceptance-v2'


def components(*, gravity, wheel_support, wheel_contact, nonwheel_support,
               robot_weight, base_contact, linear_velocity, angular_velocity,
               height, target_height, valid, parameters):
    import torch
    p = parameters
    finite = (torch.isfinite(gravity).all(-1) & torch.isfinite(wheel_support).all(-1)
              & torch.isfinite(nonwheel_support) & torch.isfinite(robot_weight)
              & (robot_weight > 0.) & torch.isfinite(base_contact)
              & torch.isfinite(linear_velocity).all(-1) & torch.isfinite(angular_velocity).all(-1)
              & torch.isfinite(height) & torch.isfinite(target_height) & valid)
    total = wheel_support.sum(-1)
    parts = dict(numerically_valid=finite,
        near_level=-gravity[:, 2] >= math.cos(math.radians(p['tilt_deg'])),
        both_wheels_touch_ground=wheel_contact.all(-1),
        each_wheel_bears_weight=(wheel_support >= (robot_weight * p['minimum_each_weight_fraction'])
            .clamp_min(p['wheel_contact_n'])[:, None]).all(-1),
        total_wheel_load=(total >= robot_weight * p['minimum_total_weight_fraction'])
            & (total <= robot_weight * p['maximum_total_weight_fraction']),
        no_nonwheel_prop=nonwheel_support <= robot_weight * p['maximum_nonwheel_weight_fraction'],
        no_base_contact=base_contact <= p['base_contact_n'],
        low_linear_speed=linear_velocity.norm(dim=-1) <= p['linear_speed_m_s'],
        low_angular_speed=angular_velocity.norm(dim=-1) <= p['angular_speed_rad_s'],
        target_height=(height - target_height).abs() <= p['height_tolerance_m'])
    bearing = (finite & parts['both_wheels_touch_ground'] & parts['each_wheel_bears_weight']
               & parts['total_wheel_load'] & parts['no_nonwheel_prop'])
    balance = (bearing & parts['near_level'] & parts['no_base_contact']
               & parts['low_linear_speed'] & parts['low_angular_speed'])
    return dict(**parts, wheel_load_bearing=bearing, wheel_balance=balance,
                height_tracking=balance & parts['target_height'])


def paired_acceptance(result, source):
    """New successes cannot compensate for losing any previously balanced state."""
    if (result['num_envs'] != source['num_envs'] or result['criteria'] != source['criteria']
            or result['command'] != source['command'] or result['seconds'] != source['seconds']
            or result.get('evaluation_standard') != source.get('evaluation_standard')
            or result['learning'] or source['learning']
            or result['noise_and_delay'] != source['noise_and_delay']
            or result['nominal_physics'] != source['nominal_physics']):
        raise ValueError('Paired evaluation protocol mismatch')
    if set(result['poses']) != set(source['poses']):
        raise ValueError('Paired pose populations differ')
    poses = {}
    for pose, current in result['poses'].items():
        original = source['poses'][pose]
        if (current['initial_state_sha256'] != original['initial_state_sha256']
                or not current['initial_states_restored'] or not original['initial_states_restored']):
            raise ValueError('Paired initial states mismatch')
        for item in (original, current):
            ids = item['balanced_samples']
            if (len(ids) != len(set(ids)) or any(not isinstance(i, int) or not 0 <= i < result['num_envs'] for i in ids)
                    or len(ids) != item['sustained_last5s']['wheel_balance']):
                raise ValueError('Invalid balance sample accounting')
        old, new = set(original['balanced_samples']), set(current['balanced_samples'])
        lost, gained = sorted(old - new), sorted(new - old)
        poses[pose] = dict(lost_source_samples=lost, gained_source_samples=gained,
            source_balance=len(old), balance=len(new), regressed=bool(lost),
            numerical_failures=current['numerical_failures'],
            balance_retention_pass=not lost and current['numerical_failures'] == 0,
            all_samples_balance_pass=len(new) == result['num_envs'] and current['numerical_failures'] == 0,
            height_tracking=current['sustained_last5s']['height_tracking'])
    regressed = any(p['regressed'] for p in poses.values())
    retention = all(p['balance_retention_pass'] for p in poses.values())
    return dict(standard=STANDARD, poses=poses, regressed=regressed,
        balance_retention_pass=retention,
        verdict='regressed_rejected' if regressed else ('retained' if retention else 'numerical_failure_rejected'),
        all_samples_balance_pass=all(p['all_samples_balance_pass'] for p in poses.values()),
        all_samples_height_pass=all(p['height_tracking'] == result['num_envs'] and p['numerical_failures'] == 0
                                   for p in poses.values()))
