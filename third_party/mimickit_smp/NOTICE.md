# SMP source and license

Algorithm: Yuxuan Mu et al., *SMP: Reusable Score-Matching Motion Priors for
Physics-Based Character Control*, https://arxiv.org/abs/2512.03028v3.

Reference implementation: https://github.com/xbpeng/MimicKit,
commit **fbe1a36e878f036d6bf5521defb8eb3b8ab4170d** (2026-10-17).
Copyright 2025 Xue Bin Peng. Apache License 2.0, supplied in `LICENSE`.
MimicKit's activation implementation also carries Copyright 2024 HuggingFace Inc.
under Apache License 2.0.

Adapted files in `mjlab_training/src/ct_mjlab/smp/`:

- `dit.py`: unconditional TinyStableMotionDiTModel, StableMotionDiTBlock,
  SwiGLU, sinusoidal position/time embedding. Replaces the diffusers attention
  and embedding components with equivalent PyTorch operations, removes unused
  conditional/generation paths and an unused output modulation parameter.
- `prior.py`: epsilon/L1 diffusion training, cosine beta schedule, EMA updates,
  ESM_SDS_loss and DiffNormalizer. No generated state initialization.
- Feature layout, physical data collection, episode qualification, dataset
  splits and training/checkpoint entry points are robot-specific additions.

The original `tinymdm_model.py`, `arch.py`, `modules.py`, `EMA.py` and
`diff_normalizer.py` are retained in `reference/` for numerical checks (only
trailing newlines normalized), never imported by runtime code.
The formal check extracts the original score/training functions and supplies
fixed inputs, noise and an independent epsilon/DDIM scheduler adapter.
It also executes the original DiT blocks and wiring with copied weights and an
independent PyTorch attention backend. This is not binary compatibility with
humanoid weights.

Upstream preset: 2 DiT layers, 4 heads of dimension 64, 50 diffusion steps,
`squaredcos_cap_v2`, L1 epsilon prediction, feature std floor0.2, AdamW2e-4,
EMA0.995 every10 steps after5000, scoring steps `[22,15,8]`, reward
`exp(-6 * mean(error / running_mean_abs_error))`.

The robot uses 100Hz and 10x30 motion features, instead of the humanoid's feature
definition and 30Hz control frequency. Existing MimicKit humanoid checkpoints
are incompatible. Third-stage PPO integration is outside this change.
