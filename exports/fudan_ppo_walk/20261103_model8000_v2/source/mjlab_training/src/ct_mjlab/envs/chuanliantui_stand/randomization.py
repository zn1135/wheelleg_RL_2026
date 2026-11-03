"""Scale the existing randomization around the nominal model, only for standing."""
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from ..chuanliantui import huananhu_randomization

MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_solref",
                "dof_frictionloss", "dof_damping")


@requires_model_fields(*MODEL_FIELDS, recompute=RecomputeLevel.set_const)
def randomize_at_startup(env, env_ids, scale=.25):
    huananhu_randomization.randomize_at_startup(env, env_ids)
    if scale != 1.:
        for name in MODEL_FIELDS:
            value = getattr(env.sim.model, name)
            nominal = env.sim.get_default_field(name)
            value.copy_(nominal + scale * (value - nominal))
    state = huananhu_randomization.get_state(env)
    state.mass_scales = 1 + scale * (state.mass_scales - 1)
    state.com_offset *= scale
    state.training_randomization_scale = scale
