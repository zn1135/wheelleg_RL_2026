"""Check the 2027 registered GPU environment with 20 environments and no learning."""
import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ct_mjlab.envs import task_registry

TASK = "2027_wheelleg_walk"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer", choices=("none", "viser"), default="none")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    if args.steps < 200:
        raise ValueError("Use at least 200 policy steps for this finite-state precheck")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    runner_module = importlib.import_module("ct_mjlab.envs.2027_wheelleg.runner")
    cfg, ppo = task_registry.get_cfgs(TASK)
    config_record = importlib.import_module("ct_mjlab.envs.2027_wheelleg.config").config_record
    report = dict(task=TASK, robot=runner_module.robot_record(), num_envs=20,
                  passed=False, physics_steps=0, optimizer_updates=0, configuration=config_record(cfg),
                  scope="geometry, finite GPU dynamics, interface and checkpoint save/load; behavior acceptance pending")
    env = None
    server = None
    started = time.monotonic()
    try:
        env, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=ppo,
                                        num_envs=20, device="cuda:0", seed=ppo.seed)
        obs, critic, history = env.reset()
        assert (obs.shape, critic.shape, history.shape) == ((20, 25), (20, 141), (20, 125))
        model = env.env.sim.mj_model
        assert model.neq == 6
        mass = float(model.body_mass[env.robot.indexing.body_ids.cpu().numpy()].sum())
        asset_model = importlib.import_module("ct_mjlab.envs.2027_wheelleg.model").load_model()
        assert abs(mass - float(asset_model.body_mass.sum())) < 1e-9, mass
        spring_ids = [model.actuator("robot/" + name + "_gas_spring_motor").id for name in ("left", "right")]
        left = [model.site(f"robot/{side}_{name}").id for side in ("left", "right")
                for name in ("rear_pin1", "rear_pin2", "spring_tip")]
        right = [model.site(f"robot/{side}_{name}").id for side in ("left", "right")
                 for name in ("front_pin1", "front_pin2", "gas_spring_lower")]
        if args.viewer == "viser":
            import viser
            from mjlab.viewer.viser.scene import MjlabViserScene
            server = viser.ViserServer(host="127.0.0.1", port=args.port, label=TASK)
            scene = MjlabViserScene(server, model, 20, sim_model=env.env.sim.model,
                                   expanded_fields=env.env.sim.expanded_fields)
            scene.show_only_selected = False
            scene.create_scene_gui(camera_distance=6., show_debug_viz_control=False)
            print(f"20-environment GPU precheck http://127.0.0.1:{server.get_port()}", flush=True)
        peak_gap = 0.
        completed = 0
        for step in range(args.steps):
            tick = time.monotonic()
            actions = torch.zeros(20, 6, device=env.device)
            obs, critic, reward, done, info, history = env.step(actions)
            for value in (obs, critic, reward, history, env.env.sim.data.qpos, env.env.sim.data.qvel):
                assert torch.isfinite(value).all()
            torch.testing.assert_close(env.env.sim.data.ctrl[:, spring_ids], torch.full((20, 2), 400., device=env.device))
            gap = (env.env.sim.data.site_xpos[:, left] - env.env.sim.data.site_xpos[:, right]).norm(dim=-1)
            peak_gap = max(peak_gap, float(gap.max()))
            completed += int(done.sum())
            report["physics_steps"] += cfg.control.decimation
            if server is not None:
                scene.update(env.env.sim.data)
                time.sleep(max(0., env.dt - (time.monotonic() - tick)))
        runner, _ = task_registry.make_alg_runner(env, TASK, train_cfg=ppo,
                                                log_dir=args.output / "checkpoint", device="cuda:0")
        saved = runner.save("random_check.pt")
        saved_data = torch.load(saved, map_location="cpu", weights_only=False)
        assert saved_data["task"] == TASK and saved_data["robot"] == runner_module.robot_record()
        assert saved_data["contract"] == runner_module.CONTRACT
        # Zero-update fixtures can restore network tensors, while the normal
        # resume entry correctly requires a completed training iteration.
        from ct_mjlab.fudan_ppo_walk.contract import make_network
        restored_policy = make_network(ppo).to(env.device)
        restored_policy.load_state_dict(saved_data["model_state_dict"], strict=True)
        for name, value in runner.policy.state_dict().items():
            torch.testing.assert_close(restored_policy.state_dict()[name], value, rtol=0, atol=0)
        try:
            runner.load(saved)
        except ValueError as error:
            assert "update counters" in str(error)
        else:
            raise AssertionError("An untrained checkpoint must not be accepted as a trained resume")
        assert runner.primary_updates == runner.encoder_updates == 0
        with torch.inference_mode():
            action, velocity = runner.policy.act_inference(obs, history)
        assert action.shape == (20, 6) and velocity.shape == (20, 3)
        assert torch.isfinite(action).all() and torch.isfinite(velocity).all()
        runner.writer.close()
        report.update(passed=True, episode_completions=completed, max_pin_gap_m=peak_gap,
                      body_mass_kg=mass, checkpoint_tensor_roundtrip=True, zero_update_resume_rejected=True, normal_entry_point=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["wall_s"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if server is not None:
            server.stop()
        if env is not None:
            env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
