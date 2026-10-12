# VLA-Reasoner

Official implementation of **VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte
Carlo Tree Search** (ICRA 2026).

- Paper: https://arxiv.org/abs/2509.22643
- Project page: https://wkguo.github.io/VLA-Reasoner/

## Method in brief

At test time a frozen VLA proposes an action `a_t^VLA = π_θ(s_t, l)`. VLA-Reasoner searches for a better action with
a lightweight MCTS. The search runs inside a learned, action-conditioned video world model `W`, so the real
execution is never touched. The search result is injected into the VLA action:

```
a_t = α · a_t^VLA + (1 − α) · a_t^Reasoner          (Eq. 1, default α = 0.6; α = 1 is the vanilla VLA)
```

| Component | Paper | Code |
|---|---|---|
| Tree search: Top-k expansion (Eq. 2), simulation (Eq. 3), backpropagation (Eq. 4), UCB selection (Eq. 5), Algorithm 1 | Sec. III-B | `vla_reasoner/mcts.py` |
| Visit counts from KDE densities, `N(a) ∝ p_KDE(a)`, summed into `N(o_i)` (Eq. 4) | Sec. III-B/C | `mcts.pseudo_counts` |
| Gaussian KDE action prior (Eq. 6) + Top-k sampling (Eq. 2) | Sec. III-B/C | `vla_reasoner/kde.py`, `vla_reasoner/sampling.py` |
| Image value: frozen ImageNet ResNet-34 + 2-layer MLP, MSE on linear progress labels (Eq. 7) | Sec. III-D | `vla_reasoner/value/` |
| Action-conditioned world model (iVideoGPT: 114M tokenizer + 436M transformer) | Sec. III-B(b), IV-A | `vla_reasoner/world_model/ivideogpt.py` |
| Propose → search → inject, sparse search schedule, action chunks | Sec. III-A/B/C | `vla_reasoner/reasoner.py` |

Algorithm 1:

```
o(0): s(0) = s_t, a(0) = a_t^VLA
for d = 0 .. MaxDepth:
    A_TopK ← TopK(π_KDE, a(d))            # Eq. 2: N samples from the KDE, keep the k nearest to a(d)
    for a in A_TopK:  ŝ ← W(s(d), a);  v̂ ← MLP(ŝ);  Update(o(d), v̂) by Eq. 4
    o(d) ← SelectUCB(o(d)) by Eq. 5;  a(d) ← a(o(d))
a^Reasoner ← argmax over the root's children of Q(o_i)
a_t ← α · a_t^VLA + (1 − α) · a^Reasoner
```

## Repository layout

```
vla_reasoner/
  kde.py            Gaussian KDE over (flattened) actions, Top-k sampling           (numpy only)
  sampling.py       KDE Top-k sampler; Gaussian-noise sampler (ablation)          (numpy only)
  mcts.py           Node, pseudo-counts, Eq. 4, Eq. 5, Algorithm 1                (numpy only)
  reasoner.py       VLAReasoner: propose -> search -> inject, schedule, chunks    (numpy only)
  deploy.py         robot-agnostic real-world loop + Robot interface              (numpy only)
  value/            labels.py (down-sampling, i/(L-1) labels), model.py (ResNet-34 + MLP), dataset.py, train.py
  world_model/      base.py (interface), ivideogpt.py (adapter), oracle.py (DEBUG-only simulator oracle)
  policies/         base.py, octo_simpler.py, spatialvla_simpler.py, openvla.py, openpi_client.py
  data/             episode.py (npz format, chunking), libero_hdf5.py, bridge_rlds.py
experiments/
  fit_kde.py  train_value.py  export_world_model_data.py  ivideogpt_launch.py
  eval_libero.py  eval_simpler.py  deploy_real.py  common.py  configs/*.flags
scripts/            collect_failures_*.sh, prepare_*.sh, finetune_world_model.sh, eval_*.sh, ablation_*.sh, deploy_real_pi0fast.sh
examples/toy_reach.py   2-D sanity check of the search machinery
tests/              CPU unit tests
```

The core (`kde.py`, `sampling.py`, `mcts.py`, `reasoner.py`, the interfaces, the data containers) depends only on numpy.
torch and torchvision are used by `value/` and `world_model/ivideogpt.py`. Every benchmark or model dependency
(simpler_env, octo/jax, tensorflow, transformers, libero, openpi_client, iVideoGPT) is imported lazily inside its
adapter.

## Installation

The core, the value function and the tests:

```bash
git clone https://github.com/wkguo/VLA-Reasoner.git && cd VLA-Reasoner
pip install -e ".[torch,scripts,libero,dev]"      # or: pip install -r requirements/core.txt
pytest
```

Each benchmark has its own (mutually incompatible) software stack. Install VLA-Reasoner (`pip install -e ".[torch,scripts]"`)
into each environment below. No git submodules are used; clone the dependencies yourself at the commits below.

| Dependency | Repository | Commit | Used for |
|---|---|---|---|
| SimplerEnv (Octo) | nakamotoo/SimplerEnv | `f343c82` (V-GPS submodule pin) | WidowX tasks, `OctoInference` |
| Octo | nakamotoo/octo | `38b9b76` (V-GPS submodule pin) | Octo-Small |
| SimplerEnv (SpatialVLA) | DelinQu/SimplerEnv-OpenVLA | `ccfe380` | `SpatialVLAInference` |
| OpenVLA | openvla/openvla | `c8f03f4` | LIBERO evaluation protocol |
| LIBERO | Lifelong-Robot-Learning/LIBERO | `8f1084e` | LIBERO simulation |
| iVideoGPT | thuml/iVideoGPT | `d601d5c` | world model |
| openpi | Physical-Intelligence/openpi | `215abfb` | π0-FAST server / client |

- **LIBERO + OpenVLA**: see `requirements/libero_openvla.txt` (LIBERO from source; `transformers==4.40.1`, optional flash-attn;
  without flash-attn pass `--attn_implementation=` to `eval_libero.py`).
- **SimplerEnv + Octo** (the V-GPS environment): see `requirements/simpler_octo.txt`.
- **SimplerEnv + SpatialVLA**: see `requirements/simpler_spatialvla.txt` (`transformers==4.47.0`).
- **iVideoGPT** (world model): clone iVideoGPT at the commit above (it is used as an external dependency, not
  vendored) and download the pretrained `thuml/ivideogpt-oxe-64-act-free` and `thuml/ivideogpt-oxe-64-act-free-medium`
  checkpoints from Hugging Face. World-model training runs in iVideoGPT's own environment
  (`requirements/ivideogpt.txt`). At evaluation time the adapter adds `--ivideogpt_root` to `sys.path` and imports
  iVideoGPT's model code (`ivideogpt.vq_model`, `ivideogpt.transformer`) into the evaluation process, so the evaluation
  environments above additionally need `diffusers==0.27.0` (listed in their requirements files together with
  `huggingface_hub<0.26`, which that diffusers version needs).
- **Real world (π0-FAST)**: `requirements/real_openpi.txt` on the robot side; serve the fine-tuned π0-FAST with openpi.

## Data preparation

All components are trained on the dataset that was used to fine-tune the VLA. The world model additionally sees a
small set of failure rollouts of the frozen VLA (Sec. III-D, IV-A/B).

- **LIBERO**: the official LIBERO hdf5 files (one folder per suite: `libero_spatial`, `libero_object`, `libero_goal`,
  `libero_10`), or OpenVLA's regenerated no-op-free hdf5 files. The loader removes no-ops with OpenVLA's criterion,
  rotates images by 180° as OpenVLA's evaluation does, and maps actions into OpenVLA's output space
  (gripper `1 − clip(g, 0, 1)`).
- **SimplerEnv WidowX**: BridgeData V2 in TFDS format (`bridge_dataset/1.0.0`). Actions are transformed like each policy's
  training pipeline: Octo's Bridge transform for Octo, OpenVLA/SpatialVLA's `bridge_orig` transform for SpatialVLA.
- **Real world / custom**: one `.npz` per episode with `image (T,H,W,3) uint8`, `action (T,A)` (raw VLA action space),
  `instruction`, `success` (see `vla_reasoner/data/episode.py`). `deploy_real.py --save_dir` writes this format.

Rollouts of the vanilla VLA are saved in the same npz format by `eval_libero.py`, `eval_simpler.py` and `deploy_real.py`
via `--save_rollouts_dir` / `--save_dir`. `scripts/collect_failures_libero.sh` uses random initial placements
(`--random_init_states`), so the fixed LIBERO test states are not used for world-model training.

## Training (three components)

```bash
# 0) failure rollouts of the frozen VLA (for the world model)
bash scripts/collect_failures_libero.sh                          # or collect_failures_simpler.sh

# 1) KDE prior, 2) value function, 3a) world-model data -- per LIBERO suite
LIBERO_DATA=/path/to/libero/datasets bash scripts/prepare_libero.sh
#    (single commands:)
python experiments/fit_kde.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/libero_spatial \
    --chunk_size=1 --output=checkpoints/libero_spatial/kde.npz
python experiments/train_value.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/libero_spatial \
    --output=checkpoints/libero_spatial/value.pt
python experiments/export_world_model_data.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/libero_spatial \
    --failure_path=rollouts/vanilla/libero_spatial --chunk_size=1 --resolution=64 --output_dir=data/wm/libero_spatial

# 3b) fine-tune iVideoGPT (tokenizer, then the action-conditioned transformer) with the official scripts;
#     stage 2 needs iVideoGPT's I3D model (see below)
IVG=/path/to/iVideoGPT PRETRAINED=/path/to/hf_ckpts DATA=data/wm/libero_spatial ACTION_DIM=7 \
    bash scripts/finetune_world_model.sh                         # stage 1 (tokenizer)
TOKENIZER=<stage-1 run>/checkpoint-<step>/unwrapped_model IVG=... PRETRAINED=... DATA=... ACTION_DIM=7 \
    bash scripts/finetune_world_model.sh                         # stage 2 (transformer)
```

For SimplerEnv use `BRIDGE=/path/to/tfds bash scripts/prepare_bridge.sh`. For chunked policies, `--chunk_size` must equal
the number of actions executed per query (SpatialVLA 4, π0-FAST 5). The KDE then models flattened chunks, and the
world-model data keeps every `chunk_size`-th frame with the executed chunk packed into one action vector.

For the real robot, record demonstrations and failure rollouts in the npz format (`deploy_real.py --save_dir`) and train
the three components with `--data_format=npz`, e.g. for π0-FAST (5 executed actions per query):

```bash
python experiments/fit_kde.py --data_format=npz --data_path=data/real/demos --chunk_size=5 \
    --output=checkpoints/real/kde_pi0fast_c5.npz
python experiments/train_value.py --data_format=npz --data_path=data/real/demos --output=checkpoints/real/value.pt
python experiments/export_world_model_data.py --data_format=npz --data_path=data/real/demos \
    --failure_path=rollouts/real --chunk_size=5 --resolution=64 --output_dir=data/wm/real
```

`experiments/ivideogpt_launch.py` registers the exported folder as an iVideoGPT dataset mix and runs the unmodified
upstream `train_tokenizer.py` / `train_gpt.py` from the iVideoGPT directory. Stage 2 (`train_gpt.py`) builds iVideoGPT's
video-metric evaluator, which loads the I3D TorchScript model from `$IVG/pretrained_models/i3d/i3d_torchscript.pt`;
download it once as described in iVideoGPT's README
(`mkdir -p $IVG/pretrained_models/i3d && wget -O $IVG/pretrained_models/i3d/i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"`).
`scripts/finetune_world_model.sh` checks for this file before stage 2.

Use the last `checkpoints/checkpoint_<step>` directory of the transformer run as `--wm_transformer` (its
`model.safetensors` holds the `HeadModelWithAction` weights), the tokenizer's `unwrapped_model` directory as
`--wm_tokenizer`, and `$IVG/configs/llama/config_medium.json` as `--wm_config`.

## Evaluation

```bash
# LIBERO (Table I): OpenVLA-SFT, 50 trials x 10 tasks = 500 episodes per suite, with and without the reasoner
IVG=/path/to/iVideoGPT bash scripts/eval_libero.sh
python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
    --task_suite_name=libero_spatial --use_reasoner --alpha=0.6 \
    --kde_path=checkpoints/libero_spatial/kde.npz --value_path=checkpoints/libero_spatial/value.pt \
    --ivideogpt_root=$IVG --wm_tokenizer=... --wm_transformer=... --wm_config=$IVG/configs/llama/config_medium.json

# SimplerEnv WidowX (Table I): 100 episodes per task
IVG=... bash scripts/eval_simpler_octo.sh          # Octo-Small, V-GPS environment
IVG=... bash scripts/eval_simpler_spatialvla.sh    # SpatialVLA, SimplerEnv-OpenVLA environment

# Ablations (Fig. 5, Fig. 6)
IVG=... bash scripts/ablation_alpha.sh             # alpha in {1.0, 0.8, 0.6, 0.4, 0.2}
IVG=... bash scripts/ablation_techniques.sh        # Gaussian-noise sampling; token reward head (needs
                                                   # WM_REWARD, see Implementation details, item 10)

# Real world (Table II): implement vla_reasoner.deploy.Robot for your arm, serve pi0-FAST with openpi
ROBOT=my_pkg.my_robot:MyArm IVG=... bash scripts/deploy_real_pi0fast.sh
```

`--world_model=oracle_debug` replaces the world model by the simulator itself (state save/restore). This is a **debugging aid
only**. It is not part of VLA-Reasoner, and results obtained with it are not VLA-Reasoner results.

## Hyper-parameters

| Parameter | Value | Notes |
|---|---|---|
| Injection strength α | 0.6 (ablation grid {1.0, 0.8, 0.6, 0.4, 0.2}) | paper |
| UCB constant c | 1/√2 | paper (example value after Eq. 5) |
| Executed actions per query | OpenVLA 1, Octo 1, SpatialVLA 4, π0-FAST 5 | paper |
| KDE kernel | Gaussian | paper |
| KDE bandwidth | Scott's rule on standardized actions | Silverman's rule or a manual value via `fit_kde.py --bandwidth` |
| KDE samples per expansion N | 1000 | sets the size of the Top-k neighbourhood |
| Candidates per expansion k | 5 | |
| MaxDepth | 2 | loop `d = 0..MaxDepth`, i.e. 3 expansions, 15 world-model calls per search |
| Search interval | every VLA query | `--search_interval` |
| Value encoder | ImageNet ResNet-34, frozen | paper |
| Value head | 2-layer MLP (512 → 256 → 1, ReLU, no output activation), MSE | |
| Value training | Adam, lr 1e-3, 50 epochs, batch 256, 10% episodes held out | |
| Frame down-sampling | keep a frame if its mean abs. grayscale change to the last kept frame ≥ 0.01 | `--downsample=stride` also available |
| Value labels | i / (L − 1) over kept frames | paper |
| World model | iVideoGPT, 64×64, 2 context frames, segment 12, top-k 100 sampling | iVideoGPT's action-conditioned recipes |
| World-model size | tokenizer 114M + "medium" transformer 436M | |
| Failure rollouts for the world model | ≤ 10 per task | |
| Episodes | LIBERO 500 per suite; SimplerEnv 100 per task; real world 20 per task | paper |

## Implementation details

1. **Visit counts.** `N(a) ∝ p_KDE(a)` (Sec. III-C). The `Q` backup of Eq. 4 does not depend on the scale of the counts
   but Eq. 5 does, so per expansion the sibling densities are normalized to pseudo-counts `n_j = k · p_j / Σ_l p_l`
   (computed from log-densities), and an expanded node has `N(o_i) = Σ N(a) = k` (Eq. 4). A leaf's count is its own
   `N(a_i)`. The root's own value is not evaluated by Algorithm 1, so Eq. 4 has no self term at the root.
   `ln N(parent) < 0` is clamped to 0 in Eq. 5.
2. **Search.** Algorithm 1 descends a single path: at each depth only the UCB-selected child is expanded, the other
   children keep their one-step value, and the selected one gets a backed-up `Q` (Eq. 4). `TopK(π_KDE, a_d)` at depth `d`
   is centred on the selected child's action. `Update(o(d), v̂)` attaches the child with value `v̂` and recomputes Eq. 4
   along the path to the root. `for d = 0 to MaxDepth` is inclusive, the k simulations of one expansion are batched, and
   ties go to the first child. `a^Reasoner` is the argmax over the root's children.
3. **KDE.** Per-dimension standardization, an isotropic Gaussian kernel in standardized space, and Scott's bandwidth.
   Top-k distances are measured in the same standardized space over the dimensions that vary in the data (dimensions
   that are constant in the data keep their constant value and are left out of the distance). Chunks are sliding
   windows (stride 1) of the executed length; episodes shorter than a chunk are skipped. One KDE per dataset (e.g. per
   LIBERO suite).
4. **Gaussian-noise ablation.** Noise std = 0.1 × per-dimension dataset std. The pseudo-counts come from the Gaussian noise
   density.
5. **Search schedule.** On queries without a search (`search_interval > 1`), the VLA action is executed unchanged.
6. **Action space.** The reasoner edits the raw VLA output before benchmark post-processing (gripper binarization,
   Euler → axis-angle), including the gripper dimension. For Octo it acts on the action after Octo's temporal ensembling
   (`OctoInference.get_action`). For SpatialVLA, temporal ensembling is disabled because 4-action chunks are executed.
7. **World-model rollout.** One tree step = one generated frame, following iVideoGPT's `VideoPredictor.rollout`: the action is
   added to the separator-token embedding, 16 dynamics tokens are sampled (top-k 100, temperature 1), and the frame is
   decoded from the context + new tokens. Each node keeps its own token sequence. For chunked policies, one tree step covers
   one executed chunk (frames at stride H, chunk packed into one action), consistently with the KDE. The context frames are
   the last real frames seen at VLA queries; at the start of an episode they are padded by repetition. The world model is
   the 114M iVideoGPT tokenizer with the 436M "medium" transformer.
8. **Value function.** Trained only on successful demonstrations. Frozen features are cached and only the MLP is optimized.
   `--degrade_resolution` (off by default) can resize training frames to the world-model resolution first.
9. **Training data.** For SimplerEnv WidowX, the KDE, value function and world model are trained on BridgeData V2, the data
   Octo and SpatialVLA were trained on. Failure rollouts for the world model: in LIBERO they are collected from random
   initial placements rather than the 50 fixed test states per task; `collect_failures_simpler.sh` samples from the same
   finite configuration set as the SimplerEnv evaluation.
10. **Token reward head (Fig. 6).** `--value=token_reward` expects a transformer checkpoint whose `reward_linear` head
    was trained on the Sec. III-D progress labels; `scripts/ablation_techniques.sh` takes it as `WM_REWARD`.

## Toy example

`examples/toy_reach.py` uses a 2-D point in a corridor. A frozen "VLA" has a constant lateral bias and too-weak feedback,
so the deviation accumulates until the point leaves the corridor. The world model is the true dynamics, the value is a
hand-written progress function, and the KDE is fit on expert actions. The constants were chosen by hand so that the bias is
within reach of the Top-k neighbourhood. The toy uses N = 50, k = 5 and MaxDepth = 2.

```bash
python examples/toy_reach.py --episodes 100 --alpha 0.6
```

Success over seeds 0–99 (100 episodes, same initial states and VLA noise for both columns):

| Setting | Base VLA | + Reasoner |
|---|---|---|
| α = 1.0 / 0.8 / 0.6 / 0.4 / 0.2 | 0.00 | 0.00 / 0.11 / 0.79 / 0.99 / 1.00 |

In this 2-D toy the correction available per step is bounded by the Top-k neighbourhood. It is a sanity check of the
search machinery, not a benchmark.

## Tests

```bash
pytest
```

The CPU test suite covers the KDE (normalization, log-density, Top-k selection, constant dimensions, chunk flattening,
save/load), the density pseudo-counts, hand-computed Eq. 4 and Eq. 5 values, the exact Algorithm 1 call sequence (with
mocked world model and value function), injection, the search interval, chunk handling and context padding, value
labels and down-sampling, the value model (forward/backward and training), the data transforms, the iVideoGPT adapter
bookkeeping, the deployment loop, the toy example, and import hygiene (the core imports without torch/tensorflow/jax).

## Citation

```bibtex
@article{guo2025vla,
  title={{VLA-Reasoner}: Empowering Vision-Language-Action Models with Reasoning via Online {Monte Carlo} Tree Search},
  author={Guo, Wenkai and Lu, Guanxing and Deng, Haoyuan and Wu, Zhenyu and Tang, Yansong and Wang, Ziwei},
  journal={arXiv preprint arXiv:2509.22643},
  year={2025}
}
```

## Credits

This code builds on, or follows the interfaces of:
[V-GPS](https://github.com/nakamotoo/V-GPS) (repository and evaluation-script structure),
[iVideoGPT](https://github.com/thuml/iVideoGPT) (world model),
[SimplerEnv](https://github.com/simpler-env/SimplerEnv) and its forks by
[nakamotoo](https://github.com/nakamotoo/SimplerEnv) and [DelinQu](https://github.com/DelinQu/SimplerEnv-OpenVLA),
[Octo](https://github.com/octo-models/octo), [OpenVLA](https://github.com/openvla/openvla),
[SpatialVLA](https://github.com/SpatialVLA/SpatialVLA), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
and [openpi](https://github.com/Physical-Intelligence/openpi).

Short code excerpts are adapted from MIT-licensed projects (license texts in `THIRD_PARTY_NOTICES.md`):
`vla_reasoner/policies/spatialvla_simpler.py` (DelinQu/SimplerEnv-OpenVLA, © 2024 simpler-env);
`vla_reasoner/policies/openvla.py`, `vla_reasoner/data/libero_hdf5.py` and `experiments/eval_libero.py` (OpenVLA,
© 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti); `vla_reasoner/world_model/ivideogpt.py` (iVideoGPT, © 2024
THUML @ Tsinghua University); and numpy ports in `vla_reasoner/data/bridge_rlds.py` (Octo, © 2023 Robotic AI & Learning
Lab Berkeley). Those excerpts remain under the MIT License of their projects. The rest of this repository is released
under the Apache-2.0 License (see `LICENSE`).
