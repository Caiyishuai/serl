# SERL + ManiSkill + Rsync 实验工作区

本项目基于 [rail-berkeley/SERL](https://github.com/rail-berkeley/serl)，在其 JAX/Flax SAC、DrQ、RLPD、MuJoCo 和真机基础设施上增加了：

- Rsync reward-manifold-aware adaptive target synchronization；
- ManiSkill3 在线 DrQ/RLPD；
- ManiSkill demo 的纯离线 SERL 训练；
- MuJoCo motion-planning demo 采集；
- macOS/CPU 单进程验证脚本；
- ReFlow 基础策略 + SERL 残差强化学习。

上游 SERL 已宣布由 HIL-SERL 接替。本仓库保留上游实现，同时作为 Rsync 和 ManiSkill 的研究实验台；不要把它视为未修改的官方 SERL 快照。

## 三项目关系

```text
maniskill-ws
  官方 motion-planning / PPO demo
        │ list transition 或 stacked pickle
        ▼
serl
  SAC / DrQ / RLPD + adaptive target synchronization
        │
        └── 当前使用环境原生 reward

Rsync/label
  二值 demo → dense labels → visual reward model
        │
        └── 尚未接入本仓库训练 loop
```

当前已接通的是：

1. `maniskill-ws/auto_research/scripts/convert_demo_h5.py --out_serl`
2. `serl/auto_research/scripts/train_serl_offline_maniskill.py --data ...`
3. `serl_launcher` 中的 adaptive-τ SAC/DrQ 更新。

当前未接通的是：

- `Rsync/label` 的自动稠密标注；
- Rsync 视觉奖励模型 checkpoint；
- 用奖励模型输出替换本仓库环境 reward 的在线训练。

因此，“SERL 能跑 ManiSkill demo”和“论文完整奖励模型闭环已经复现”是两个不同结论。

## 训练路径选择

| 目标 | 入口 | 特点 |
|---|---|---|
| 官方 MuJoCo state SAC | `examples/async_sac_state_sim/` | actor/learner 两进程 |
| 官方 MuJoCo vision DrQ + demo | `examples/async_drq_sim/` | agentlace + RLPD |
| macOS/CPU 快速验证 | `auto_research/scripts/train_sac_single.py` | 单进程、避免分布式依赖 |
| ManiSkill demo 纯离线训练 | `auto_research/scripts/train_serl_offline_maniskill.py` | 跨仓库 stacked pickle |
| ManiSkill 在线 DrQ/RLPD | `examples/async_drq_sim/async_drq_sim_maniskill_ty.py` | Linux + CUDA，两进程 |
| MuJoCo motion-planning demo | `examples/motion_planning/` | pick/stack 脚本 |
| ReFlow 残差 RL | `examples/reflow_res_rl/` | 依赖 sibling `maniskill-ws` |
| 真机 SERL/HIL | `examples/train_rlpd.py`、`serl_robot_infra/` | 需要 Franka/ROS/SpaceMouse |

## 安装

### 基础 SERL + MuJoCo

```bash
conda create -n serl python=3.10
conda activate serl

# GPU
pip install --upgrade "jax[cuda12_pip]==0.4.35" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 或 CPU
pip install --upgrade "jax[cpu]"

cd /Users/yishuaicai/mywork/serl/serl_launcher
pip install -e .
pip install -r requirements.txt

cd ../franka_sim
pip install -e .
pip install -r requirements.txt
```

注意：

- `franka_sim/requirements.txt` 固定 `mujoco==2.3.7`，而 `auto_research` 的已验证环境使用 MuJoCo 3.1.6；两条路径不是同一锁定环境。
- 仓库根目录可能使 `franka_sim` 被当作 namespace package 而不注册 gym 环境。遇到环境不存在时，从 `/tmp` 等仓库外目录运行绝对路径脚本。
- macOS 不应设置 `MUJOCO_GL=egl`。

### auto_research 已有环境

当前机器记录的独立环境：

```text
auto_research/venv_serl
Python 3.11
JAX 0.4.35
MuJoCo 3.1.6
```

它用于已有研究记录，不应代替正式 requirements/lockfile。

## MuJoCo

### 分布式 state SAC

```bash
cd /Users/yishuaicai/mywork/serl/examples/async_sac_state_sim
bash tmux_launch.sh
```

也可分别启动：

```bash
bash run_learner.sh
bash run_actor.sh
```

### 分布式 vision DrQ + 20 demos

```bash
cd /Users/yishuaicai/mywork/serl/examples/async_drq_sim

wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
wget https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl

bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl
bash run_actor.sh
```

### macOS/CPU 单进程 RLPD + adaptive τ

```bash
cd /tmp

/Users/yishuaicai/mywork/serl/auto_research/venv_serl/bin/python \
  /Users/yishuaicai/mywork/serl/auto_research/scripts/train_sac_single.py \
  --env PandaPickCube-v0 \
  --max_steps 2000 \
  --demo_path /Users/yishuaicai/mywork/serl/auto_research/data/demos_pickcube_state_20.pkl \
  --adaptive_tau
```

### Motion-planning demo

Pick：

```bash
cd /Users/yishuaicai/mywork/serl
python examples/motion_planning/collect_pick_trajs_motion_planning_cys.py
```

输出：

```text
examples/motion_planning/franka_lift_cube_image_<N>_trajs.pkl
```

Stack：

```bash
python examples/motion_planning/collect_stack_trajs_motion_planning_cys.py
```

输出：

```text
examples/motion_planning/franka_stack_image_<N>_trajs_dense.pkl
```

`examples/motion_planning/README_stack_data_collection.md` 提到的 `collect_stack_trajs_simple.py` 和 `collect_stack_trajs_motion_planning_v2.py` 当前不存在；以上述实际脚本名为准。`tools/generate_demos.py` 的现有实验记录为 0% success，不推荐作为复现入口。

## ManiSkill

### 官方 motion-planning demo → SERL 离线训练

先在 `maniskill-ws` 的 ManiSkill 环境下载、回放并转换：

```bash
cd /Users/yishuaicai/mywork/maniskill-ws
V=auto_research/venv_maniskill

$V/bin/python -m mani_skill.utils.download_demo PushCube-v1 \
  -o auto_research/ms_assets/demos

$V/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path auto_research/ms_assets/demos/PushCube-v1/motionplanning/trajectory.h5 \
  -o state \
  -b physx_cpu \
  --use-env-states \
  --record-rewards \
  --reward-mode dense \
  --save-traj \
  --count 100

$V/bin/python auto_research/scripts/convert_demo_h5.py \
  --h5 auto_research/ms_assets/demos/PushCube-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
  --out_serl auto_research/data/ms_pushcube_expert_serl.pkl
```

再在 SERL 环境中训练：

```bash
cd /tmp

/Users/yishuaicai/mywork/serl/auto_research/venv_serl/bin/python \
  /Users/yishuaicai/mywork/serl/auto_research/scripts/train_serl_offline_maniskill.py \
  --data /Users/yishuaicai/mywork/maniskill-ws/auto_research/data/ms_pushcube_expert_serl.pkl \
  --max_updates 3000 \
  --adaptive_tau
```

该脚本只进行离线更新，并报告 action MSE/critic loss；它没有 ManiSkill 环境 rollout，因此这些指标不能替代 policy success rate。

### ManiSkill 在线 DrQ/RLPD

该路径假设 Linux、CUDA、ManiSkill、两进程通信和图像 observation：

```bash
cd /Users/yishuaicai/mywork/serl/examples/async_drq_sim
bash run_maniskill_tmux.sh
```

或：

```bash
bash run_learner_maniskill.sh
bash run_actor_maniskill.sh
```

运行前必须检查脚本中的：

- `/home/.../maniskill-ws/serl_data/...` demo 绝对路径；
- CUDA 设备；
- W&B 配置；
- `server_port`/`broadcast_port`；
- observation keys、shape 和 action dimension。

`async_drq_sim_maniskill_ty.py` 会严格校验 demo observation 与环境 shape，不匹配时直接报错。

### PPO checkpoint 采集 ManiSkill demo

```bash
cd examples/async_drq_sim/maniskill_data
python 3_collect_data_maniskill_ppo_checkpoints_complete.py
```

该脚本使用 `physx_cuda`，不能直接在无 NVIDIA GPU 的 macOS 上运行。

## Demo schema

### 通用 SERL transition

```python
{
    "observations": np.ndarray | dict,
    "actions": np.ndarray,
    "next_observations": np.ndarray | dict,
    "rewards": float,
    "masks": float,
    "dones": bool,
    # optional: infos, task_stage, grasp_penalty
}
```

MuJoCo state、MuJoCo vision 和 ManiSkill vision 的 observation 结构并不相同：

- state-only：flat vector 或可被 gym flatten 的 dict；
- MuJoCo vision：常见 `state`、`front`、`wrist`；
- ManiSkill vision：常见 `state`、`hand_camera`、`base_camera`；
- chunked DrQ：每个 observation 通常还需前导 horizon 维。

### ManiSkill 纯离线 stacked dict

```python
{
    "observations": np.ndarray,       # [N, obs_dim]
    "next_observations": np.ndarray,  # [N, obs_dim]
    "actions": np.ndarray,            # [N, action_dim]
    "rewards": np.ndarray,            # [N]
    "masks": np.ndarray,              # [N]
    "dones": np.ndarray,              # [N]
}
```

### MuJoCo vision demo

motion-planning 脚本通常输出 `list[transition]`，observation 示例：

```python
{
    "state": np.ndarray,  # 带 horizon 维
    "front": np.ndarray,
    "wrist": np.ndarray,
}
```

跨环境使用 demo 时必须同时核对：

1. observation keys；
2. 每个 tensor 的 shape；
3. action dimension 和控制模式；
4. `masks` 对 terminated/truncated 的语义；
5. image dtype/range；
6. 是否已经添加 chunk/horizon 维。

## Rsync adaptive target synchronization

核心实现位于：

```text
serl_launcher/serl_launcher/agents/continuous/sac.py
serl_launcher/serl_launcher/utils/launcher.py
```

SAC/DrQ agent 配置包含 adaptive synchronization 开关及运行时 `curr_tau`。训练脚本通过 `--adaptive_tau` 或工厂参数 `adaptive_tau_enabled` 启用；不同入口命名不完全一致，应以对应脚本的 `--help` 为准。

其目标是根据 reward manifold/progress 信号调整 target-network Polyak 更新速度，而不是始终使用固定 τ。

当前实现边界：

- SAC/DrQ 中的 adaptive τ 已存在；
- MuJoCo 单进程和 ManiSkill 离线脚本已有对照实验入口；
- ManiSkill 在线 DrQ 脚本可加载 demo；
- `Rsync/label` 和视觉 reward model 尚未接入本仓库；
- 纯离线脚本没有 rollout success 评估；
- 完整论文表格需要固定任务、数据、种子和统一评估脚本。

## ReFlow 残差 RL

`examples/reflow_res_rl/` 将 PyTorch ReFlow 基础策略与 JAX DrQ 残差动作结合：

```text
a_final = a_base(obs) + alpha_scale × a_delta(obs)
```

基础策略权重冻结，DrQ 只学习残差。主要入口：

```bash
# 可选：专家动作转残差动作
python examples/reflow_res_rl/make_residual_demos.py \
  --demo_path serl_data/<demo>.pkl \
  --output_path examples/reflow_res_rl/demo_data/<env>_residual.pkl \
  --reflow_ckpt runs/<run>/checkpoints/best_success_once.pt \
  --env PushCube-v1

# learner
DEMO_PATH=examples/reflow_res_rl/demo_data/<env>_residual.pkl \
  bash examples/reflow_res_rl/run_learner.sh

# actor
REFLOW_CKPT=runs/<run>/checkpoints/best_success_once.pt \
  bash examples/reflow_res_rl/run_actor.sh
```

该目录的导入和 checkpoint 路径假定 `maniskill-ws` 是 sibling 仓库；换机器后需修改路径。详见 `examples/reflow_res_rl/README_zh.md`。

## 目录与重要文件

```text
serl/
├── README.md
│   本文档。
├── docs/
│   ├── sim_quick_start.md
│   │   上游 MuJoCo state/vision SERL 快速开始。
│   └── real_franka.md
│       上游 Franka 真机任务说明。
├── serl_launcher/
│   ├── setup.py / requirements.txt
│   │   核心 JAX 包及依赖。
│   └── serl_launcher/
│       ├── agents/continuous/
│       │   SAC、DrQ、BC、VICE；sac.py 含 adaptive τ。
│       ├── data/
│       │   replay buffer 和 memory-efficient image buffer。
│       ├── utils/launcher.py
│       │   make_sac_agent/make_drq_agent 等工厂。
│       ├── wrappers/
│       │   observation、chunking 和环境包装器。
│       └── vision/
│           视觉 encoder 和预训练参数工具。
├── franka_sim/
│   MuJoCo Panda gym 环境和 XML assets，注册 pick/stack/insert 等任务。
├── serl_robot_infra/
│   Franka 真机 Flask/ROS 服务和 gym 环境。
├── examples/
│   ├── async_sac_state_sim/
│   │   上游 MuJoCo state SAC actor/learner。
│   ├── async_drq_sim/
│   │   vision DrQ，以及本地 ManiSkill actor/learner 扩展。
│   ├── motion_planning/
│   │   MuJoCo pick/stack 脚本化 demo 采集。
│   ├── reflow_res_rl/
│   │   ReFlow 基础策略 + DrQ residual RL。
│   ├── res_rl_sim/
│   │   MuJoCo residual RL 实验。
│   ├── meanflow_res_rl_2026_0115/
│   │   MeanFlow/residual 实验副本。
│   ├── train_rlpd.py
│   │   真机 HIL learner/actor 入口。
│   ├── record_demos.py
│   │   真机演示记录。
│   └── experiments/
│       ram、usb、handover、egg_flip 等真机配置。
├── tools/
│   ├── generate_demos.py
│   │   旧 pick demo 生成器；现有记录成功率为 0。
│   ├── pkl_to_lerobot.py
│   │   SERL pickle → LeRobot 的可选桥接工具。
│   └── visualize_trajs.py
│       轨迹可视化。
├── auto_research/
│   ├── 00_RESEARCH_LOG.md
│   │   macOS 复现、命令、版本和实验结论。
│   ├── scripts/train_sac_single.py
│   │   MuJoCo 单进程 SAC/RLPD。
│   ├── scripts/train_serl_offline_maniskill.py
│   │   ManiSkill stacked demo 的纯离线 JAX 更新。
│   ├── scripts/gen_demos_state.py
│   │   MuJoCo state demo 生成器，成功率不稳定。
│   ├── data/
│   │   小型实验 demo。
│   └── logs/
│       adaptive/fixed τ 和任务矩阵日志。
└── third_party/hil-serl/
    上游 HIL-SERL 快照/子模块，与本仓库主 serl_launcher 并行。
```

## 评估语义

不同入口的“评估”不可直接比较：

- 分布式 SAC/DrQ：环境 rollout return/success；
- ManiSkill 在线 DrQ：episode success 平均；
- `train_serl_offline_maniskill.py`：BC action MSE 和 critic loss，无 rollout；
- `train_sac_single.py`：episode return 日志；
- motion-planning 采集：成功 transition/episode 数。

论文复现应使用环境 success rate，并报告多随机种子的均值与方差，不能用离线 action MSE 代替。

## 已知问题

- ManiSkill learner shell 脚本含绝对 demo 路径、CUDA 和 W&B 配置。
- 部分源码中存在硬编码 W&B key；公开或共享代码前应移除并轮换凭据。
- 上游 MuJoCo requirements 与本机已验证版本不一致。
- `tools/generate_demos.py` 和 `gen_demos_state.py` 的脚本化抓取成功率不稳定。
- stack 数据采集文档引用了不存在的脚本。
- ManiSkill PPO demo 采集固定 `physx_cuda`。
- 纯离线 ManiSkill 训练没有环境 rollout。
- `Rsync/label` 的 dense labels 和视觉 reward model 未接入。
- `examples/experiments/` 需要真机硬件、ROS 和任务标定，不能作为仿真复现入口。
- 论文 Table 5 使用 `ξ=0.05`、`δτ=0.2`、`ρ=1.1`、`τ∈[0.001,0.05]`。`make_drq_agent` 默认值与之相符，但 `make_sac_agent` 和 `SACAgent.create` 仍默认 `ξ=0.3`、`δτ=0.4`、`ρ=1.2`、`τ∈[0.005,0.2]`；复现实验必须显式传参，不能依赖默认值。
- 论文使用 DrQ + 10 critic REDQ，并用 PBRS potential difference 作为在线 reward；本仓库的不同训练入口并不都同时满足这两个条件。
- 论文仿真还包含 PlaceSphere、LIBERO 和 π0.5 residual 对照；当前仓库没有一套统一脚本可复现这些结果。
- 论文报告多随机种子环境 success，而本仓库现有 `auto_research` 日志主要证明代码路径可运行，不等价于论文统计复现。

## 上游资料

- SERL 项目页：https://serl-robot.github.io/
- HIL-SERL：https://hil-serl.github.io/
- MuJoCo quick start：`docs/sim_quick_start.md`
- 真机说明：`docs/real_franka.md`

## Citation

使用上游 SERL 时请引用：

```bibtex
@misc{luo2024serl,
  title={SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning},
  author={Jianlan Luo and Zheyuan Hu and Charles Xu and You Liang Tan and Jacob Berg and Archit Sharma and Stefan Schaal and Chelsea Finn and Abhishek Gupta and Sergey Levine},
  year={2024},
  eprint={2401.16013},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

使用 Rsync 扩展时还应引用对应 Rsync 论文；论文匿名投稿阶段请按投稿要求处理作者信息。
