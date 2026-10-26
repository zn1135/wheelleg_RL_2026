"""Actor39 contract with two reset-relative wheel-angle slots."""

OBSERVATION_CONTRACT = 'chuanliantui-huananhu-wheel-angle-39-82-v1'
PAIR_SCHEMA = 'chuanliantui_actor39_pd6_prestep_wheel_angle_v1'
FORMAT = 'ct_bc_actor39_pd6_wheel_angle_v1'
WHEEL_ANGLES = dict(schema='reset_relative_native_wheel_angles_v1',
    actor_indices=[14,15],critic_indices=[14,15],joints=['lfwheel','rfwheel'],
    units='radians',scale=1.,clip=[-100.,100.],origin='episode_reset',
    signs='native_joint_axes_same_as_wheel_velocity',wrap=False,
    sensor_group='joint_positions',noise_bound_rad=.025)

HANDOVER_OBSERVATION_CONTRACT = 'chuanliantui-huananhu-wheel-angle-39-82-handover-v2'
HANDOVER_PAIR_SCHEMA = 'chuanliantui_actor39_pd6_prestep_wheel_angle_handover_v2'
HANDOVER_FORMAT = 'ct_bc_actor39_pd6_wheel_angle_handover_v2'
HANDOVER_WHEEL_ANGLES = dict(WHEEL_ANGLES, schema='first_recovery_relative_native_wheel_angles_v2',
    origin='episode_reset_then_first_ground_start_recovered',
    trigger='ground_start_and_recovered_first_true', rearm='episode_reset_only',
    delay_history='clear_wheel_position_columns_only_at_handover')
PROFILES = {
    OBSERVATION_CONTRACT: dict(pair_schema=PAIR_SCHEMA, model_format=FORMAT, wheel_angles=WHEEL_ANGLES),
    HANDOVER_OBSERVATION_CONTRACT: dict(pair_schema=HANDOVER_PAIR_SCHEMA, model_format=HANDOVER_FORMAT,
                                     wheel_angles=HANDOVER_WHEEL_ANGLES),
}
