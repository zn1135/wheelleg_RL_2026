"""Encoder-only travel shared by live observations and offline BC pairs."""

WHEEL_OBSERVATION_CONTRACT = 'chuanliantui-huananhu-multimode-wheel-40-83-v1'
WHEEL_ODOMETRY = dict(schema='signed_mean_wheel_travel_v1', radius_m=.06,
    joints=['lfwheel', 'rfwheel'], forward_signs=[1, -1], origin='episode_reset',
    units='metres', scale=1., actor_index=39, critic_index=82,
    noise=False, delay=False, wrap=False)


def wheel_displacement(q, origin):
    # XML左右轮轴相反；连续编码器角不按2π回绕，前进为正。
    delta = q-origin
    return (.5*WHEEL_ODOMETRY['radius_m']*(delta[..., 0]-delta[..., 1]))[..., None]
