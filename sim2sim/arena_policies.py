"""Four frozen Fudan controllers sharing a single RMUC robot."""
from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train import load_jump_policy
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from scripts.agent.fudan_ppo_jump.handoff import LandingHandoff
from sim2sim.step_playground import ManualStepPolicies

ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / 'logs/mjlab/fudan_ppo_step/20261119_restart_v11_4096_resume5200_to6200_01/model_final.pt'
PARK = ROOT / 'logs/mjlab/fudan_ppo_step/20261118_fixed_speed2_L40_resume4847_to5200_01/model_final.pt'
JUMP = ROOT / 'logs/mjlab/fudan_ppo_jump/20261119_preload4200_to5200_4096_01/model_final.pt'


class ArenaPolicies(ManualStepPolicies):
    def __init__(self, walk):
        step, saved = load_policy(STEP)
        super().__init__(walk, STEP, step, saved, parking_checkpoint=PARK)
        jump, meta = load_jump_policy(JUMP)
        for key in ('reset_contract', 'control', 'control_timing', 'reference', 'physical_leg_load',
                    'physical_wheel_model', 'wheel_response'):
            if json.dumps(meta[key], sort_keys=True) != json.dumps(saved[key], sort_keys=True):
                raise ValueError(f'Jump physical interface differs: {key}')
        if meta['disturbances']['action_delay'] or meta['disturbances']['sensor_delay']:
            raise ValueError('Expected Jump with direct actions and sensors')
        self.policies['jump'] = jump
        self.metadata['jump'] = meta
        self.jump_scale = np.asarray(meta['command_scale'], dtype=np.float32)
        self.training_disturbances['jump'] = meta['disturbances']
        self.checkpoints['jump'] = dict(checkpoint=str(JUMP), sha256=hashlib.sha256(JUMP.read_bytes()).hexdigest(),
                                        iterations=meta['iterations'])
        self.gate = None

    def reset(self, env):
        self.mode = 'walk'
        self.walk_height = float(env.command[2])
        self.reset_parking()
        self.gate = None
        self.jump_history = np.tile(env.direct_observation(self.jump_scale), 5)

    def start_jump(self, env):
        if self.mode != 'walk' or self.gate is not None:
            return False
        self.gate = LandingHandoff(torch.tensor([env.steps]), buffer_ticks=8)
        self.transitions.append(dict(time_s=float(env.data.time), before='walk', after='jump', reason='jump_request'))
        return True

    def inputs(self, env):
        # 延迟帧与即时帧取自同一物理状态，分别保存历史；同拍更新最新指令。
        obs = env.observation()
        env.history[-19:-16] = obs[6:9]
        direct = env.direct_observation(self.jump_scale)
        self.jump_history[-25:] = direct
        return (direct, self.jump_history) if self.gate is not None else (obs, env.history)

    def act_inference(self, observation, history):
        if self.gate is None:
            return super().act_inference(observation, history)
        self.active_role = 'jump'
        return self.policies['jump'].act_inference(observation, history)

    def after_step(self, env, samples):
        self.jump_history = np.concatenate((self.jump_history[25:], env.direct_observation(self.jump_scale)))
        if self.gate is None:
            return
        rows = np.asarray(samples)
        tick = env.steps - 1
        self.gate.observe_signals(tick, torch.tensor([(np.abs(rows[:, :2]) > 1.).any()]),
                                  torch.tensor([(rows[:, 3] > .15).any()]), self.gate.active(tick))
        if not self.gate.active(env.steps).item():
            landed = self.gate.touchdown.item()
            self.transitions.append(dict(time_s=float(env.data.time), before='jump', after='walk',
                reason='landed' if landed >= 0 else 'timeout', touchdown_tick=landed,
                start_tick=int(self.gate.started.item()), return_tick=env.steps))
            self.gate = None


class FieldContacts:
    """Net vertical wheel forces and the largest non-wheel ground force after each integration."""
    def __init__(self, model):
        self.body = model.geom_bodyid
        self.fixed = model.body_weldid == 0
        self.wheels = {model.body(name).id: i for i, name in enumerate(('lfwheel', 'rfwheel'))}
        self.force = np.empty(6)

    def sample(self, model, data):
        net = {}
        for i, contact in enumerate(data.contact):
            a, b = self.body[contact.geom]
            if self.fixed[a] == self.fixed[b] or contact.efc_address < 0:
                continue
            body = b if self.fixed[a] else a
            mujoco.mj_contactForce(model, data, i, self.force)
            world = contact.frame.reshape(3, 3).T @ self.force[:3]
            net[body] = net.get(body, 0.) + (world if self.fixed[a] else -world)
        loads = [0., 0.]
        other = 0.
        for body, force in net.items():
            if body in self.wheels:
                loads[self.wheels[body]] = force[2]
            else:
                other = max(other, np.linalg.norm(force))
        return (*loads, other, data.qvel[2])
