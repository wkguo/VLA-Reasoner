# VLA-Reasoner

Official implementation of **VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search** (ICRA 2026).

[Paper](https://arxiv.org/abs/2509.22643) | [Project page](https://wkguo.github.io/VLA-Reasoner/)

VLA-Reasoner is a plug-in for frozen VLAs. At every step the VLA proposes an action, a Monte Carlo tree search looks ahead in an action-conditioned world model, and the best searched action is blended into the VLA action:

```
a = α · a_VLA + (1 − α) · a_Reasoner        (α = 0.6)
```

Expansion samples candidates from a KDE action prior fitted on the demonstrations, and states are scored by a progress value network (frozen ResNet-34 + MLP). The VLA is never retrained.

## Installation

```bash
git clone https://github.com/wkguo/VLA-Reasoner.git && cd VLA-Reasoner
pip install -e ".[torch,scripts,libero,dev]"
pytest
```

Each benchmark runs in its own environment, listed in `requirements/`:

| Setting | File |
|---|---|
| LIBERO + OpenVLA | `libero_openvla.txt` |
| SimplerEnv + Octo | `simpler_octo.txt` |
| SimplerEnv + SpatialVLA | `simpler_spatialvla.txt` |
| iVideoGPT world model (training) | `ivideogpt.txt` |
| Real robot + pi0-FAST | `real_openpi.txt` |

Install VLA-Reasoner into the environment you use. The world model needs a clone of [iVideoGPT](https://github.com/thuml/iVideoGPT) (`git clone` and checkout as in `requirements/ivideogpt.txt`); pass its path as `--ivideogpt_root`.

## Training

Each benchmark needs three components trained on the VLA's fine-tuning data: the KDE prior, the value function, and the world model.

```bash
bash scripts/collect_failures_libero.sh                 # failure rollouts of the vanilla VLA (for the world model)
LIBERO_DATA=/path/to/libero bash scripts/prepare_libero.sh   # KDE, value function, world-model data
IVG=... PRETRAINED=... DATA=data/wm/libero_spatial bash scripts/finetune_world_model.sh   # tokenizer
TOKENIZER=<tokenizer dir> IVG=... PRETRAINED=... DATA=... bash scripts/finetune_world_model.sh   # transformer
```

For SimplerEnv use `collect_failures_simpler.sh` and `BRIDGE=/path/to/tfds bash scripts/prepare_bridge.sh`. `PRETRAINED` holds the Hugging Face checkpoints `thuml/ivideogpt-oxe-64-act-free` and `thuml/ivideogpt-oxe-64-act-free-medium`. The transformer stage needs the I3D model from the iVideoGPT README at `$IVG/pretrained_models/i3d/i3d_torchscript.pt`.

For chunked policies, `--chunk_size` in `fit_kde.py` and `export_world_model_data.py` is the number of actions executed per query (SpatialVLA 4, π0-FAST 5). Your own data works with `--data_format=npz` (one `.npz` per episode, see `vla_reasoner/data/episode.py`).

## Evaluation

```bash
IVG=... bash scripts/eval_libero.sh                  # LIBERO, 500 episodes per suite
POLICY=octo IVG=... bash scripts/eval_simpler.sh     # SimplerEnv WidowX (POLICY=spatialvla for SpatialVLA)
IVG=... bash scripts/ablation.sh                     # α grid and Gaussian-noise sampling
```

Use the last `checkpoint_<step>` of the transformer run as `--wm_transformer` and the tokenizer's `unwrapped_model` directory as `--wm_tokenizer`.

For a real robot, implement `vla_reasoner.deploy.Robot` for your arm, serve pi0-FAST with [openpi](https://github.com/Physical-Intelligence/openpi), and run `ROBOT=my_pkg.my_robot:MyArm IVG=... bash scripts/deploy_real_pi0fast.sh`.

## Citation

```bibtex
@article{guo2025vla,
  title={{VLA-Reasoner}: Empowering Vision-Language-Action Models with Reasoning via Online {Monte Carlo} Tree Search},
  author={Guo, Wenkai and Lu, Guanxing and Deng, Haoyuan and Wu, Zhenyu and Tang, Yansong and Wang, Ziwei},
  journal={arXiv preprint arXiv:2509.22643},
  year={2025}
}
```

## Acknowledgements

Built on [V-GPS](https://github.com/nakamotoo/V-GPS), [iVideoGPT](https://github.com/thuml/iVideoGPT), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [openpi](https://github.com/Physical-Intelligence/openpi).

Released under the Apache-2.0 License. Code adapted from MIT-licensed projects is listed in `THIRD_PARTY_NOTICES.md`.
