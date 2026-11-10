"""Shared trace schema and frozen evaluation inputs; GPU and CPU physics remain separate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.managers.event_manager import RecomputeLevel
from .contract import MOTORS, LEGS, RESET_CONTRACT, RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
FIELDS = ('body_mass', 'body_inertia', 'body_ipos', 'geom_friction', 'geom_solref')
COLUMNS = dict(position=slice(0, 3), linear_body=slice(3, 6), linear_world=slice(6, 9), angular_body=slice(9, 12), roll_pitch_deg=slice(12, 14), wheel_support=slice(14, 16), base_contact=16, weight=17)

def gpu_row(env):
    q = env.sim.data.qpos
    v = env.sim.data.qvel
    quat = q[:, 3:7]
    linear = quat_apply_inverse(quat, v[:, :3])
    gravity = quat_apply_inverse(quat, q.new_tensor([0.0, 0.0, -1.0]).expand(env.num_envs, -1))
    angles = torch.stack((torch.atan2(-gravity[:, 1], -gravity[:, 2]), torch.asin(gravity[:, 0].clamp(-1.0, 1.0))), -1) * 180 / np.pi
    support = -env.scene['wheel_contact'].data.force[:, :, 2]
    base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    mass = env.sim.model.body_mass
    weight = mass.sum(-1).expand(env.num_envs) * 9.81
    return torch.cat((q[:, :3] - env.scene.env_origins, linear, v[:, :3], v[:, 3:6], angles, support, base[:, None], weight[:, None]), -1)

def arrays(rows, qpos, terminated=None, numerical=None):
    rows = np.asarray(rows)
    n = len(rows)
    result = {key: rows[:, slot] for key, slot in COLUMNS.items()}
    result.update(time=np.arange(n) * 0.002, qpos=np.asarray(qpos), qpos_time=np.arange(len(qpos)) * 0.01, terminated=np.zeros(n, dtype=bool) if terminated is None else terminated, numerical_failure=np.zeros(n, dtype=bool) if numerical is None else numerical)
    return result

def configure_samples(wrapper, samples, out):
    """Independent seeded physical parameters; export exactly those inputs to CPU."""
    import mujoco
    env = wrapper.env
    model = env.sim.mj_model
    action = wrapper.action
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'model.mjb').exists():
        mujoco.mj_saveModel(model, str(out / 'model.mjb'))
    robot = env.scene['robot']
    ids = robot.indexing.body_ids.cpu().numpy()
    base = model.body('robot/base_link').id
    all_geoms = np.flatnonzero(np.isin(model.geom_bodyid, ids))
    snapshots = {field: np.repeat(np.asarray(getattr(model, field))[None], len(samples), axis=0) for field in FIELDS}
    kp = np.ones((len(samples), 6))
    kd = kp.copy()
    strength = kp.copy()
    offset = np.zeros_like(kp)
    qpos = env.sim.data.qpos.cpu().numpy().copy()
    qvel = env.sim.data.qvel.cpu().numpy().copy()
    qpos[:, :2] = env.scene.env_origins[:, :2].cpu().numpy()
    base_mass = np.zeros(len(samples))
    com = np.zeros((len(samples), 3))
    friction = np.full(len(samples), 0.5)
    restitution = np.full(len(samples), 0.5)
    for i, sample in enumerate(samples):
        case_hash = int(hashlib.sha256(sample['case']['id'].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(np.random.SeedSequence([sample['seed'], case_hash]))
        qvel[i] = 0.0
        qvel[i, :6] = rng.uniform(-0.5, 0.5, 6)
        if sample['randomized']:
            added = rng.uniform(-1.0, 2.0)
            scale = rng.uniform(0.9, 1.1, len(ids))
            com[i] = rng.uniform(-0.02, 0.02, 3)
            snapshots['body_mass'][i, base] += added
            base_mass[i] = snapshots['body_mass'][i, base]
            snapshots['body_mass'][i, ids] *= scale
            snapshots['body_inertia'][i, ids] *= scale[:, None]
            snapshots['body_ipos'][i, base] += com[i]
            buckets = rng.uniform(0.6, 1.4, 64)
            friction[i] = buckets[rng.integers(64)]
            restitution[i] = rng.uniform(0.6, 1.0)
            snapshots['geom_friction'][i, all_geoms, 0] = (friction[i] + 0.5) * 0.5
            log = np.log((restitution[i] + 0.5) * 0.5)
            zeta = -log / np.sqrt(np.pi ** 2 + log ** 2)
            snapshots['geom_solref'][i, all_geoms] = [-10000.0, -200 * zeta]
            kp[i] = rng.uniform(0.95, 1.05, 6)
            kd[i] = rng.uniform(0.95, 1.05, 6)
            strength[i] = rng.uniform(0.95, 1.05, 6)
            offset[i] = rng.uniform(-0.03, 0.03, 6)
        else:
            base_mass[i] = model.body_mass[base]
    for key, value in snapshots.items():
        getattr(env.sim.model, key).copy_(torch.as_tensor(value, device=env.device))
    env.sim.recompute_constants(RecomputeLevel.set_const)
    env._fudan_parameters = {k: torch.as_tensor(v, device=env.device, dtype=torch.float32) for k, v in dict(base_mass=base_mass, com=com, friction=friction, restitution=restitution).items()}
    nominal = action.nominal_positions - action.nominal_offset
    action.nominal_offset.copy_(torch.as_tensor(offset, device=env.device))
    action._nominal_pos.copy_(nominal + action.nominal_offset)
    for key, value in [('kp_scale', kp), ('kd_scale', kd), ('motor_strength', strength)]:
        getattr(action, key).copy_(torch.as_tensor(value, device=env.device))
    action.rebuild_reset_poses()
    qpos[:, :3] = env.scene.env_origins.cpu().numpy()
    qpos[:, 2] += action.reset_root_height.cpu().numpy()
    qpos[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    joint_q = robot.indexing.joint_q_adr.cpu().numpy()
    qpos[:, joint_q] = action.reset_joint_positions.cpu().numpy()
    action._raw_actions.zero_()
    action._control_actions.zero_()
    wrapper.previous.zero_()
    wrapper.previous_vel.zero_()
    wrapper.fail_ticks.zero_()
    wrapper.sums.zero_()
    wrapper.episode_steps.zero_()
    wrapper.command_mode.zero_()
    wrapper.command_elapsed.zero_()
    wrapper.stop_command_step.zero_()
    env.sim.data.qpos.copy_(torch.as_tensor(qpos, device=env.device))
    env.sim.data.qvel.copy_(torch.as_tensor(qvel, device=env.device))
    env.sim.forward()
    env.scene.update(dt=0.0)
    env.sim.sense()
    action.last_sample_q.copy_(env.sim.data.qpos[:, action._motor_q_adr])
    action.sample_velocity.zero_()
    disturbances = getattr(wrapper, 'disturbance_config', DisturbanceConfig())
    if hasattr(action, 'disturbances'):
        state = action.disturbances
        state.reset(slice(None), action.sensor_frame())
        wrapper.update_push_scale()
        for i, sample in enumerate(samples):
            case_hash = int(hashlib.sha256(sample['case']['id'].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng([sample['seed'], case_hash, 20261102])
            for enabled, bounds, target in ((disturbances.action_delay, disturbances.action_delay_ms, state.action_steps), (disturbances.sensor_delay, disturbances.sensor_delay_ms, state.sensor_steps)):
                if enabled:
                    target[i] = int(np.rint(rng.uniform(*bounds) / 2.0))
            state.push_seeds[i] = int(rng.integers(0, 2 ** 31)) if disturbances.pushes else 0
            for columns, bounds in ((slice(0, 3), disturbances.external_force_range), (slice(3, 6), disturbances.external_torque_range)):
                if bounds != (0., 0.):
                    state.episode_wrench[i, columns] = torch.as_tensor(rng.uniform(*bounds, 3), device=env.device)
    wrapper.previous_position.copy_(wrapper.robot.data.root_link_pos_w)
    wrapper._update_state()
    for i, sample in enumerate(samples):
        wrapper.commands[i] = torch.tensor(sample['case']['segments'][0]['command'], device=env.device)
    wrapper.noise_mask = torch.tensor([s['randomized'] for s in samples], device=env.device)
    clean = wrapper.frame().clone()
    obs, critic = wrapper.observations()
    wrapper.history = torch.cat((clean.repeat(1, 4), wrapper._history_frame), -1) if wrapper.reset_contract == RESET_V2_CONTRACT else obs.repeat(1, 5)
    wrapper._history_needs_fill.zero_()
    mapping = dict(reset_contract=wrapper.reset_contract, disturbances=disturbances.record(), motor_q=action._motor_q_adr.cpu().tolist(), motor_v=action._motor_v_adr.cpu().tolist(), motor_ids=[model.actuator('robot/' + n + '_motor').id for n in MOTORS], spring_ids=action._spring_ids.cpu().tolist(), spring_q=action._spring_q_adr.cpu().tolist(), spring_v=action._spring_v_adr.cpu().tolist(), base=base, wheels=[model.body('robot/' + n).id for n in ('lfwheel', 'rfwheel')], wheel_response=action._wheel_response.cpu().tolist(), reference=action.reference_positions[0].cpu().tolist())
    mapping['episode_wrench_frame'] = 'body_at_base_com'
    (out / 'mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    for i, sample in enumerate(samples):
        pose = qpos[i].copy()
        pose[:2] = 0.0
        disturbance_initial = {}
        if hasattr(action, 'disturbances'):
            disturbance_initial = dict(action_delay_steps=int(state.action_steps[i]), sensor_delay_steps=int(state.sensor_steps[i]), push_seed=int(state.push_seeds[i]), push_force_scale=float(action.push_force_scale[i]), episode_wrench_body=state.episode_wrench[i].cpu().numpy())
        np.savez_compressed(out / (sample['id'] + '_initial.npz'), qpos=pose, qvel=qvel[i], nominal=action.nominal_positions[i].cpu().numpy(), kp=kp[i], kd=kd[i], strength=strength[i], **disturbance_initial, **{k: v[i] for k, v in snapshots.items()})
    return (obs, critic)

def update_evaluation_commands(wrapper, commands, obs):
    wrapper.commands.copy_(torch.as_tensor(commands, device=wrapper.device, dtype=obs.dtype))
    scaled = wrapper.commands * obs.new_tensor(wrapper.cfg.normalization.command)
    if wrapper.reset_contract == RESET_CONTRACT:
        scaled = scaled.clamp(-wrapper.cfg.normalization.clip, wrapper.cfg.normalization.clip)
    obs[:, 6:9] = scaled
    wrapper._history_frame[:, 6:9] = scaled
    wrapper.history[:, -19:-16] = scaled
    return obs
