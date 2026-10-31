"""One physical path supplies observations, the unchanged reward and termination.

MuJoCo leaves derived kinematics one 2ms substep behind integrated q/qdot.
We keep the five physical samples, rather than learning a second reward state.
Contact wrenches and actual actuator forces remain learned physical quantities;
tree kinematics is not a contact or closed-chain dynamics solver.
"""

import math
import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

from .analytic_reward import differentiable_reward
from .kinematics import RobotKinematics


PHYS_DIM, DELTA_DIM, SUBSTEPS, SENSOR_SAMPLES = 64, 63, 5, 9
STATE_DIM, PARAM_DIM = 254, 75
PHYSICAL = slice(0,64)
SENSORS = slice(64,226)
NOISE = slice(226,244)
MEMORY = slice(244,254)
NOMINAL = slice(0,6)
KP, KD, STRENGTH = slice(6,12), slice(12,18), slice(18,24)
ACTION_LAGS, SENSOR_LAGS = slice(24,26), slice(26,30)
MASSES, BASE_IPOS, FRICTION = slice(30,50), slice(50,53), slice(53,59)
SPRING = slice(59,63)
BASE_INERTIA, SOLREF, WRENCH = slice(64,67), slice(67,69), slice(69,75)
ACTION_ORDER = (0,1,3,4,2,5)


def unit_quat(q):
    return q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)


def rotation_vector(q):
    q = unit_quat(q)
    q = torch.where(q[...,:1] < 0.,-q,q)
    length = q[...,1:].norm(dim=-1,keepdim=True)
    angle = 2.*torch.atan2(length,q[...,:1].clamp_min(1e-8))
    scale = torch.where(length > 1e-5,angle/length.clamp_min(1e-8),2.+length.square()/3.)
    return q[...,1:]*scale


def rotation_quat(vector):
    # sinc avoids the undefined derivative of sin(theta/2)/theta at zero.
    angle = vector.norm(dim=-1,keepdim=True)
    return torch.cat((torch.cos(angle*.5),vector*.5*torch.sinc(angle/(2.*math.pi))),-1)


def physical_delta(before, after):
    """Body-local translation/rotation/velocity deltas, then joints and forces."""
    q = unit_quat(before[...,3:7])
    inv = torch.cat((q[...,:1],-q[...,1:]),-1)
    return torch.cat((quat_apply_inverse(q,after[...,:3]-before[...,:3]),
        rotation_vector(quat_mul(inv,unit_quat(after[...,3:7]))),
        after[...,7:25]-before[...,7:25],
        quat_apply_inverse(q,after[...,25:28]-before[...,25:28]),
        after[...,28:]-before[...,28:]),-1)


def integrate_physical(before, delta):
    q = unit_quat(before[...,3:7])
    return torch.cat((before[...,:3]+quat_apply(q,delta[...,:3]),
        unit_quat(quat_mul(q,rotation_quat(delta[...,3:6]))),
        before[...,7:25]+delta[...,6:24],
        before[...,25:28]+quat_apply(q,delta[...,24:27]),
        before[...,28:]+delta[...,27:]),-1)


class StateAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kin = RobotKinematics()
        names = self.kin.joint_names
        # XML joints after the free joint are all scalar hinge/slide coordinates.
        from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
        motor = [names.index(name)-1 for name in ACTUATED_JOINTS]
        self.register_buffer('motor_q',torch.tensor([7+i for i in motor]))
        self.register_buffer('motor_v',torch.tensor([31+i for i in motor]))
        self.register_buffer('leg_q',self.motor_q[[0,1,3,4]].clone())
        self.register_buffer('leg_v',self.motor_v[[0,1,3,4]].clone())
        self.register_buffer('knee_q',torch.tensor([7+names.index(name)-1 for name in ('lf1','rf1')]))
        self.hips = [self.kin.body_names.index(name) for name in ('lf0','rf0')]
        self.wheels = [self.kin.body_names.index(name) for name in ('lfwheel','rfwheel')]
        self.register_buffer('noise_bounds',torch.tensor([.2]*3+[.05]*3+[.02]*4+[0.]*2+[1.5]*6))

    def raw_sensors(self, physical, parameters):
        q = unit_quat(physical[...,3:7])
        gravity = torch.zeros_like(physical[...,:3]); gravity[...,2] = -1.
        relative = physical[...,self.motor_q]-parameters[...,NOMINAL]
        relative = torch.cat((relative[...,(0,1,3,4)],torch.zeros_like(relative[...,:2])),-1)
        return torch.cat((physical[...,28:31],quat_apply_inverse(q,gravity),relative,
                          physical[...,self.motor_v][...,ACTION_ORDER]),-1)

    def actor_frame(self, state, parameters, command, action):
        ring = state[...,SENSORS].reshape(*state.shape[:-1],SENSOR_SAMPLES,18)
        lags = parameters[...,SENSOR_LAGS].long().clamp(0,SENSOR_SAMPLES-1)
        parts = []
        for group,(lo,hi) in enumerate(((0,3),(3,6),(6,12),(12,18))):
            index = (SENSOR_SAMPLES-1-lags[...,group])[...,None,None].expand(*lags.shape[:-1],1,hi-lo)
            parts.append(ring[...,lo:hi].gather(-2,index).squeeze(-2))
        sensor = torch.cat(parts,-1)+state[...,NOISE]
        # Same physical motor order/scales as StandV2ObservationState.actor().
        return torch.cat((sensor[...,:3]*.25,sensor[...,3:6],command,
            sensor[...,6:10],sensor[...,12:18][...,(0,1,4,2,3,5)]*.05,action),-1).clamp(-100.,100.)

    def reward_features(self, path):
        current, lagged = path[...,-1,:], path[...,-2,:]
        q = unit_quat(lagged[...,3:7])
        gravity = torch.zeros_like(lagged[...,:3]); gravity[...,2] = -1.
        position, rotation, linear, angular = self.kin(lagged[...,:25],lagged[...,25:49])
        relative = position[...,self.wheels,:]-position[...,self.hips,:]
        relative = quat_apply_inverse(q[...,None,:].expand(*relative.shape[:-1],4),relative)
        swing = torch.atan2(-relative[...,0],-relative[...,2])
        return torch.cat((quat_apply_inverse(q,linear[...,1,:]),
            quat_apply_inverse(q,angular[...,1,:]),quat_apply_inverse(q,gravity),
            lagged[...,2:3],swing,current[...,self.leg_q],current[...,self.knee_q],
            current[...,self.leg_v],current[...,49:55],current[...,55:58].norm(dim=-1,keepdim=True),
            (-current[...,60:64:3] > 1.).to(current.dtype)),-1)

    def advance_memory(self, state, parameters, path):
        """Exact hard walking failure rules, including their episode-local counters."""
        old = state[...,MEMORY]
        current, lagged = path[...,-1,:],path[...,-2,:]
        q, q_lag = unit_quat(current[...,3:7]),unit_quat(lagged[...,3:7])
        gravity = torch.zeros_like(current[...,:3]); gravity[...,2] = -1.
        g = quat_apply_inverse(q_lag,gravity)
        gz = quat_apply_inverse(q,gravity)[...,2]
        support = -current[...,60:64:3]
        weight = parameters[...,MASSES].sum(-1)*9.81
        loaded = ((support >= (.1*weight).clamp_min(5.)[...,None]).all(-1)
                  & (support.sum(-1) >= .8*weight))
        base = current[...,55:58].norm(dim=-1)
        finite = torch.isfinite(current[...,:49]).all(-1)
        finite_support = torch.isfinite(support).all(-1) & torch.isfinite(base)
        age = old[...,0]+1.
        landed = (old[...,1] > .5) | (support > 1.).any(-1)
        stable = landed & (lagged[...,2] >= .20) & (g[...,2] <= -math.cos(math.radians(15))) & (base <= .1) & loaded
        standing = torch.where(stable,old[...,3].round()+1.,0.)
        stood = (old[...,2] > .5) | (standing >= 50.)
        airborne = torch.where(landed & ~(support > 1.).any(-1),old[...,4]+1.,0.)
        fallen = landed & stood & ((base > 10.) | (g[...,2] > -.1))
        fallen_ticks = torch.where(fallen,old[...,5]+1.,0.)
        old_failure = (landed & stood & ((g[...,2] > 0.) | (airborne >= 20.) | (fallen_ticks > 100.))) | ~finite
        down = (~finite | ~finite_support | ~loaded | (base > 10.) | (current[...,2] <= .18)
                | (gz > -math.cos(math.radians(40))))
        down_ticks = torch.where(down & (age >= 100.),old[...,6]+1.,0.)
        failure = old_failure | ~finite | ~finite_support | (gz > 0.) | (down_ticks >= 10.)
        roll = torch.atan2(-g[...,1],-g[...,2]); pitch = torch.asin(g[...,0].clamp(-1.,1.))
        observation_down = ((roll.abs() > math.radians(40)) | (pitch.abs() > math.radians(40))
                            | (lagged[...,2] <= .20) | ~loaded)
        obs_ticks = torch.where(observation_down,0.,old[...,7]+1.)
        recovered = ~observation_down & ((old[...,8] > .5) | (obs_ticks >= 50.))
        landed_ticks = torch.where((old[...,1] < .5) & landed,0.,old[...,9]+landed.float())
        memory = torch.stack((age,landed.float(),stood.float(),standing,airborne,fallen_ticks,
                              down_ticks,obs_ticks,recovered.float(),landed_ticks),-1)
        return memory,failure,~finite

    def advance(self, state, parameters, path, command, action, previous, previous_previous, context, noise):
        sensors = self.raw_sensors(path,parameters[...,None,:])
        ring = state[...,SENSORS].reshape(*state.shape[:-1],SENSOR_SAMPLES,18)
        ring = torch.cat((ring[...,SUBSTEPS:,:],sensors),-2)
        memory,failure,numerical = self.advance_memory(state,parameters,path)
        next_state = torch.cat((path[...,-1,:],ring.flatten(-2),noise,memory),-1)
        frame = self.actor_frame(next_state,parameters,command,action)
        features = self.reward_features(path)
        # These masks follow the same task memory as failure; commands remain exogenous.
        active = ~((parameters[...,63] > .5) & (state[...,MEMORY][...,0] < 20.))
        reward_context = torch.cat((context[...,:5],numerical[...,None].float(),active[...,None].float(),context[...,7:]),-1)
        _,reward = differentiable_reward(features,action,previous,previous_previous,reward_context)
        return next_state,frame,reward,failure,features


@torch.no_grad()
def physical_snapshot(env):
    robot = env.scene['robot']; data = env.sim.data
    from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
    motors = robot.find_joints(ACTUATED_JOINTS,preserve_order=True)[0]
    q = data.qpos.clone(); q[...,:3] -= env.scene.env_origins
    return torch.cat((q,data.qvel.clone(),data.qfrc_actuator[:,robot.indexing.joint_v_adr[motors]],
        env.scene['base_contact'].data.force[:,0],env.scene['wheel_contact'].data.force.flatten(-2)),-1).clone()


@torch.no_grad()
def capture_state(env):
    observer = env._multimode_observations
    command = env.command_manager.get_term('standup')
    ring = observer.delay.history
    order = (observer.delay.head+1+torch.arange(SENSOR_SAMPLES,device=env.device)) % SENSOR_SAMPLES
    memory = torch.stack((env.episode_length_buf,command.has_landed,command.has_stood,
        (command.standing_time/env.step_dt).round(),command.wheels_airborne_steps,
        command.isaac_fallen_steps,command.walk_down_ticks,command.observation_stable_steps,
        command.observation_recovered,command.landed_steps),-1).float()
    noise = observer.noise if observer.cfg.noise else torch.zeros_like(observer.noise)
    return torch.cat((physical_snapshot(env),ring[order].transpose(0,1).flatten(-2),noise,memory),-1).clone()


@torch.no_grad()
def capture_parameters(env):
    observer = env._multimode_observations; action = env.action_manager.get_term('pd')
    masses = env.sim.model.body_mass
    robot_ids = env.scene['robot'].indexing.body_ids
    robot_masses = (masses[:,robot_ids] if masses.ndim == 2
                    else masses[robot_ids].expand(env.num_envs,-1))
    # Scene compilation inserts a terrain body; canonical IDs follow the robot XML.
    masses = torch.cat((torch.zeros(env.num_envs,1,device=env.device),robot_masses),-1)
    ipos = env.sim.model.body_ipos
    base_ipos = ipos[:,observer.base_id] if ipos.ndim == 3 else ipos[observer.base_id].expand(env.num_envs,-1)
    friction = env.sim.model.geom_friction
    material = (friction[:,observer.wheel_geom_ids] if friction.ndim == 3
                else friction[observer.wheel_geom_ids].expand(env.num_envs,-1,-1)).flatten(-2)
    flags = env.command_manager.get_term('standup').ground_start.float()[:,None]
    inertia = env.sim.model.body_inertia
    base_inertia = (inertia[:,observer.base_id] if inertia.ndim == 3
                    else inertia[observer.base_id].expand(env.num_envs,-1))
    solref = env.sim.model.geom_solref
    geom = int(observer.wheel_geom_ids[0])
    contact_solref = solref[:,geom] if solref.ndim == 3 else solref[geom].expand(env.num_envs,-1)
    return torch.cat((action.nominal_positions,action.position_gains,action.velocity_gains,
        action.motor_strength,action.action_delay_steps[:,None],action.wheel_action_delay_steps[:,None],
        observer.delay.lags,masses,base_ipos,material,action._spring_force_bias,action.spring_damping,flags,
        base_inertia,contact_solref,env.sim.data.xfrc_applied[:,observer.base_id]),-1).float().clone()
