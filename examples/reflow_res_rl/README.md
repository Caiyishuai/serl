# ReFlow + Residual RL (SERL DrQ)

Combines a local **ReFlow** base policy (PyTorch) with **SERL DrQ** residual RL (JAX).

```
a_final = a_base(obs) + alpha_scale * a_delta(obs)
```

The base policy (ReFlowMLP + 2×ResNet18) is frozen; DrQ learns `a_delta`.

---

## Prerequisites

1. **Train ReFlow** first → checkpoint at `runs/<env>__reflow__<seed>__<timestamp>/checkpoints/best_success_once.pt`
   - See `reflow_train/README.md` and `reflow_train/1_train_reflow.py`

2. **Collect expert demos** (optional, improves sample efficiency):
   ```bash
   python rl/3_collect_data_maniskill_ppo_checkpoints.py
   python rl/4_convert_h5_to_pkl.py
   ```
   Demos stored in `serl_data/<env>_<reward>_<n>.pkl`

---

## Running Flow

### Step 0 (Optional): Generate residual demos

Convert expert demos so actions are relative to the ReFlow base:

```bash
python examples/reflow_res_rl/make_residual_demos.py \
    --demo_path serl_data/mani_skill_push_cube_dense_no_clip_100_fixed_complete.pkl \
    --output_path examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
    --reflow_ckpt runs/PushCube-v1__reflow__42__<timestamp>/checkpoints/best_success_once.pt \
    --env PushCube-v1
```

Output: `examples/reflow_res_rl/demo_data/pushcube_residual.pkl`

Skip this step to train without demos (slower convergence).

---

### Step 1: Start Learner

```bash
export WANDB_API_KEY=<your_key>          # optional

DEMO_PATH=examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
bash examples/reflow_res_rl/run_learner.sh
```

Or without demos:
```bash
bash examples/reflow_res_rl/run_learner.sh
```

The learner starts a parameter server on ports **5490** (server) / **5491** (broadcast).

---

### Step 2: Start Actor

```bash
export REFLOW_CKPT=runs/PushCube-v1__reflow__42__<timestamp>/checkpoints/best_success_once.pt

bash examples/reflow_res_rl/run_actor.sh
```

The actor connects to the learner, collects experience using ReFlow + RL delta, and pushes transitions.

---

## Data and Checkpoint Paths

| Item | Path |
|---|---|
| Expert demos | `serl_data/<env>_<reward>_<n>_fixed_complete.pkl` |
| Residual demos | `examples/reflow_res_rl/demo_data/<env>_residual.pkl` |
| ReFlow checkpoint | `runs/<env>__reflow__<seed>__<timestamp>/checkpoints/best_success_once.pt` |
| DrQ checkpoints | `--checkpoint_path` arg (default `/tmp/serl_ckpt/<exp_name>`) |
| Eval videos | `--eval_video_dir` arg (default `/tmp/reflow_res_eval_videos/`) |
| W&B logs | online (disable with `--debug`) |

---

## Environment Wrapper Chain

```
gym.make(env)
  └─ PotentialBasedRewardWrapper     (optional PBRS: r = φ(s') - φ(s))
       └─ ManiSkillMultiCameraWrapper (obs → {state:(35,), hand_camera:(128,128,3), base_camera:(128,128,3)})
            └─ _ReFlowResetWrapper   (calls reflow_policy.reset() on env.reset())
                 └─ AddPolicyActionWrapper  (a_final = a_base + alpha*a_delta)
                      └─ _GymEnvAdapter    (gym.Env compatibility shim)
                           └─ ChunkingWrapper(obs_horizon=1)  (adds leading dim for DrQ)
```

---

## Key Parameters

| Flag | Default | Description |
|---|---|---|
| `--reflow_ckpt` | required for actor | Path to ReFlow `.pt` checkpoint |
| `--alpha_scale` | `0.1` | Residual action scale |
| `--reflow_obs_horizon` | `2` | Obs window (must match training) |
| `--reflow_pred_horizon` | `4` | Prediction horizon (must match training) |
| `--reflow_act_horizon` | `4` | Action chunk size |
| `--reward_mode` | `normalized_dense` | `normalized_dense` / `dense` / `sparse` |
| `--potential_reward_shaping` | `True` | Enable PBRS |
| `--max_steps` | `1000000` | Total actor env steps |
| `--batch_size` | `256` | Learner batch size |
| `--server_port` | `5490` | AgentLace server port |
| `--broadcast_port` | `5491` | AgentLace broadcast port |

---

## Architecture

```
Observation (from env):
  state: (1, 35)   hand_camera: (1, 128, 128, 3)   base_camera: (1, 128, 128, 3)
         │
         ▼  (inside ReFlowPolicy)
  8-dim eef state + 2×ResNet18 features
         │
         ▼
  ReFlowMLP  ──►  a_base (7-dim, act_horizon=4 chunk)
         │
         ▼
  DrQAgent (JAX, resnet-pretrained encoder)  ──►  a_delta (7-dim)
         │
         ▼
  a_final = a_base + 0.1 * a_delta  ──►  env.step()
```

---

## File Overview

| File | Description |
|---|---|
| `reflow_policy.py` | Local PyTorch ReFlow policy (VisualEncoder + ReFlowMLP) |
| `async_drq_sim.py` | Main actor/learner script (DrQ + ReFlow residual) |
| `make_residual_demos.py` | Convert expert demos to residual demos |
| `run_learner.sh` | Learner launch script |
| `run_actor.sh` | Actor launch script |
