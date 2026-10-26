#!/usr/bin/env python3
"""Check new40D BC data, finite gradients and legacy replay/fail-closed training."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.bc import ActionDataset, BRAKING_SAMPLING, build_actor, load_actor
from ct_mjlab.scripts.train_bc import train
from check_lqr_student_closed_loop import digest


def check(args):
    torch.set_num_threads(1)
    checkpoint_sha=digest(args.checkpoint)
    actor,saved=load_actor(args.checkpoint)
    old=ActionDataset([args.legacy_dataset],args.legacy_corrections,action_sampling=BRAKING_SAMPLING)
    assert old.description['fingerprint']==saved['dataset']['fingerprint']
    assert actor.mlp[0].in_features==39
    legacy_inputs,_=old.sample('validation',32,torch.Generator().manual_seed(714))
    assert torch.isfinite(actor.mlp(legacy_inputs)).all()
    data=ActionDataset([args.run/'paired/dataset.json'],[args.run/'braking69',args.run/'braking121'],
                       action_sampling=BRAKING_SAMPLING,require_reference=True)
    network=build_actor(observation_dim=40)
    before={k:v.clone() for k,v in network.state_dict().items()}
    inputs,targets=data.sample('train',512,torch.Generator().manual_seed(714))
    loss=(network.mlp(inputs)-targets).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in network.mlp.parameters())
    for k,v in before.items():torch.testing.assert_close(network.state_dict()[k],v,atol=0,rtol=0)
    # 禁止在核对失败时意外开始更新；实际调用训练入口，Adam一旦创建即失败。
    with tempfile.TemporaryDirectory() as name:
        output=Path(name)/'must-not-exist'
        cfg=SimpleNamespace(iterations=1,batch_size=512,save_every=1,seed=42,device='cpu',
            dataset=[args.legacy_dataset],corrections=args.legacy_corrections,
            action_sampling=BRAKING_SAMPLING,init=None,output=output)
        with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer permitted in preflight')):
            try:
                train(cfg)
            except ValueError as error:
                assert 'unified label_reference' in str(error)
            else:
                raise AssertionError('Unversioned training accepted')
            cfg.dataset=[args.run/'paired/dataset.json'];cfg.corrections=[args.run/'braking69',args.run/'braking121']
            cfg.init=args.checkpoint
            try:
                train(cfg)
            except ValueError as error:
                assert 'observation dimensions differ' in str(error)
            else:
                raise AssertionError('39D actor silently initialized a 40D network')
        assert not output.exists()
    try:
        ActionDataset([args.run/'paired/dataset.json'],[args.legacy_corrections[0]],action_sampling=BRAKING_SAMPLING)
    except ValueError as error:
        assert 'parking label references' in str(error)
    else:
        raise AssertionError('Mixed shadow query reference accepted')
    # 与XML中实际轮轴和保存的回合初态交叉核对，不只用导出器自己的首帧定义。
    import mujoco
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    model=mujoco.MjModel.from_xml_path(str(MODEL_XML))
    joints=[model.joint(n).id for n in ('lfwheel','rfwheel')]
    np.testing.assert_array_equal(model.jnt_axis[joints],[[0,1,0],[0,-1,0]])
    adr=model.jnt_qposadr[joints]
    for episode in data.motion.episodes:
        with np.load(episode['path']) as d:
            np.testing.assert_array_equal(d['source_motor_q'][0,[2,5]],d['initial_qpos'][adr])
            expected=.03*((d['source_motor_q'][:,2]-d['initial_qpos'][adr[0]])-
                          (d['source_motor_q'][:,5]-d['initial_qpos'][adr[1]]))
            np.testing.assert_allclose(d['observation'][:,39],expected,atol=3e-7,rtol=2e-6)
    for directory in (args.run/'braking69',args.run/'braking121'):
        with np.load(directory/'trace.npz') as d:
            q=d['source_q']
            np.testing.assert_array_equal(q[0][:,[2,5]],d['initial_qpos'][:,adr])
            expected=.03*((q[:,:,2]-d['initial_qpos'][None,:,adr[0]])-
                          (q[:,:,5]-d['initial_qpos'][None,:,adr[1]]))
            np.testing.assert_allclose(d['observation'][:,:,39],expected,atol=3e-7,rtol=2e-6)
    assert digest(args.checkpoint)==checkpoint_sha
    result=dict(passed=True,optimizer_updates=0,finite_gradient_loss=float(loss.detach()),
        actor40_shape=[40,256,128,64,6],old39_checkpoint_replay_compatible=True,
        old_dataset_fingerprint_preserved=old.description['fingerprint'],checkpoint_sha256=checkpoint_sha,
        new_dataset_fingerprint=data.description['fingerprint'],class_frame_counts=data.description['class_frame_counts'],
        unversioned_training_rejected_before_optimizer=True,old39_actor_init_into40_rejected_before_optimizer=True,
        shadow_queries_mixed_with_new_data_rejected=True,xml_wheel_axes_and_saved_initial_encoder_origins_checked=True,
        actor_parameters_unchanged=True,checker_sha256=digest(__file__),
        scope='Data, actor forward/backward and negative training preflight only; no optimizer or policy acceptance')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--legacy-dataset',type=Path,required=True);p.add_argument('--legacy-corrections',type=Path,action='append',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);check(p.parse_args())
