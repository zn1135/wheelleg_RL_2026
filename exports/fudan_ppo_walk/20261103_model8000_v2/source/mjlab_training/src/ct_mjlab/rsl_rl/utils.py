#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import git
import os
import pathlib
import torch

def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    padded_trajectories = padded_trajectories[:, :-1]
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return (padded_trajectories, trajectory_masks)

def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, 'git')
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f'Could not find git repository in {repository_file_path}. Skipping.')
            continue
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        diff_file_name = os.path.join(git_log_dir, f'{repo_name}.diff')
        if os.path.isfile(diff_file_name):
            continue
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, 'x') as f:
            content = f'--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}'
            f.write(content)
        file_paths.append(diff_file_name)
    return file_paths
