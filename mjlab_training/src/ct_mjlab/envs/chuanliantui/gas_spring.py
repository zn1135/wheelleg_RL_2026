"""User-selected 0–150 N compression curve for the physical gas springs.

The slide zero is calibrated to this robot's CAD reference pose. Initial
compression is set to 10 mm by the user. Positive slide motion extends the spring.
"""

SPRING_INITIAL_COMPRESSION = 0.01
SPRING_FORCE_MIN = 0.0
SPRING_FORCE_MAX_REFERENCE = 150.0
SPRING_REFERENCE_STROKE = 0.09
SPRING_DAMPING = 50.0
SPRING_FORCE_LIMIT = 150.0


def gas_spring_force(position, velocity, force_bias=0.0):
    """Compute bounded slider effort for NumPy arrays or Torch tensors.

    Zero compression produces zero nominal thrust; 90 mm produces 150 N.
    Bound the elastic curve first, then bound the total after bias and damping.
    The spring only pushes: it cannot pull or exceed 150 N.
    """
    compression = (SPRING_INITIAL_COMPRESSION - position).clip(min=0.0)
    stiffness = (SPRING_FORCE_MAX_REFERENCE - SPRING_FORCE_MIN) / SPRING_REFERENCE_STROKE
    elastic_force = (SPRING_FORCE_MIN + stiffness * compression).clip(0.0, SPRING_FORCE_LIMIT)
    force = elastic_force + force_bias - SPRING_DAMPING * velocity
    return force.clip(0.0, SPRING_FORCE_LIMIT)
