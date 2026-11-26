"""GPU playback of the independent 2027 policy in a browser viewer."""
import time

import torch

from ct_mjlab.utils.task_registry import task_registry
from ct_mjlab.fudan_ppo_walk.contract import make_network
from .config import TASK
from .runner import CONTRACT, robot_record


def play(args):
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene

    cfg, ppo = task_registry.get_cfgs(TASK, play=True)
    policy = None
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if saved.get("contract") != CONTRACT or saved.get("robot") != robot_record():
            raise ValueError("Playback checkpoint does not match the 2027 robot")
        policy = make_network(ppo).to(args.device)
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        policy.eval()
    env, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=ppo, play=True,
                                    num_envs=args.num_envs or 1, device=args.device)
    server = None
    try:
        server = viser.ViserServer(host="127.0.0.1", port=args.viser_port or 8097, label=TASK)
        scene = MjlabViserScene(server, env.env.sim.mj_model, env.num_envs,
                               sim_model=env.env.sim.model, expanded_fields=env.env.sim.expanded_fields)
        scene.create_scene_gui(camera_distance=2., show_debug_viz_control=False)
        obs, critic, history = env.reset()
        print(f"GPU 2027 viewer http://127.0.0.1:{server.get_port()}", flush=True)
        while True:
            started = time.monotonic()
            with torch.inference_mode():
                actions = torch.zeros(env.num_envs, 6, device=env.device) if policy is None else policy.act_inference(obs, history)[0]
                obs, critic, reward, done, info, history = env.step(actions)
            scene.update(env.env.sim.data)
            time.sleep(max(0., env.dt - (time.monotonic() - started)))
    finally:
        if server is not None:
            server.stop()
        env.close()
