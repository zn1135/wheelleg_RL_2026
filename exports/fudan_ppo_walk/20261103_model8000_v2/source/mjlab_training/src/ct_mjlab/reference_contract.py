"""Pure-data contract for shared, observable LQR position-reference feedback."""

OBSERVATION_CONTRACT = 'chuanliantui-huananhu-reference-41-84-v1'
PAIR_SCHEMA = 'chuanliantui_actor41_pd6_prestep_reference_v1'
FORMAT = 'ct_bc_actor41_pd6_reference_v1'
REFERENCE = dict(schema='ct_bc_shared_reference_v2', actor_index=40, critic_index=83,
    units='metres', sign='target_minus_estimated_position', scale=1.,
    observer='live_H7_LQR_estimator_dt0.01_hold0.2_velocity_integral',
    reset='episode;normal_handover;air_change;first_v31_support_1s_sync_only_when_not_DONE;explicit_teacher_takeover',
    labels='same_cached_reference_as_observation_for_live_and_shadow_teacher',
    wheel_travel_origin='episode_reset_independent_of_reference_resets')
