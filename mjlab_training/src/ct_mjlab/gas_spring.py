"""Physical hinge/slide spring law, matching wheelleg_RL's 90 mm curve.

The 10 mm preload and reference mass are model settings, not measurements.
Positive slide displacement extends the spring. Internal efforts only push.
"""
SPRING_JOINTS = ('left_spring2_joint', 'right_spring2_joint')
PRELOAD = .01
REFERENCE_COMPRESSION = .09
FORCE_LIMIT = 150.
DAMPING = 50.

def gas_spring_force(position, velocity):
    compression = (PRELOAD - position).clip(min=0.)
    elastic = (FORCE_LIMIT / REFERENCE_COMPRESSION * compression).clip(0., FORCE_LIMIT)
    return (elastic - DAMPING * velocity).clip(0., FORCE_LIMIT)
