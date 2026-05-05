# ReFlow + 残差强化学习（SERL DrQ）

将本地 **ReFlow** 基础策略（PyTorch）与 **SERL DrQ** 残差 RL（JAX）结合：

```
a_final = a_base(obs) + alpha_scale * a_delta(obs)
```

基础策略（ReFlowMLP + 2×ResNet18）权重冻结，DrQ 只学习残差修正量 `a_delta`。

---

## 前置条件

1. **先训练 ReFlow** → 检查点保存在 `runs/<env>__reflow__<seed>__<timestamp>/checkpoints/best_success_once.pt`
   - 参考 `reflow_train/README.md` 和 `reflow_train/1_train_reflow.py`

2. **采集专家演示数据**（可选，能提升采样效率）：
   ```bash
   python rl/3_collect_data_maniskill_ppo_checkpoints.py
   python rl/4_convert_h5_to_pkl.py
   ```
   演示数据保存在 `serl_data/<env>_<reward>_<n>.pkl`

---

## 运行流程

### 步骤 0（可选）：生成残差演示数据

将专家演示的动作转换为相对于 ReFlow 基础策略的残差动作：

```bash
python examples/reflow_res_rl/make_residual_demos.py \
    --demo_path serl_data/mani_skill_push_cube_dense_no_clip_100_fixed_complete.pkl \
    --output_path examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
    --reflow_ckpt runs/PushCube-v1__reflow__42__<timestamp>/checkpoints/best_success_once.pt \
    --env PushCube-v1
```

输出：`examples/reflow_res_rl/demo_data/pushcube_residual.pkl`

跳过此步骤则不使用演示数据训练（收敛会慢一些）。

---

### 步骤 1：启动 Learner（学习端）

```bash
export WANDB_API_KEY=<your_key>   # 可选，不填则不上传 W&B

# 带演示数据
DEMO_PATH=examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
bash examples/reflow_res_rl/run_learner.sh

# 不带演示数据
bash examples/reflow_res_rl/run_learner.sh
```

Learner 启动参数服务器，端口：**5490**（server）/ **5491**（broadcast）。

---

### 步骤 2：启动 Actor（采集端）

在另一个终端执行：

```bash
export REFLOW_CKPT=runs/PushCube-v1__reflow__42__<timestamp>/checkpoints/best_success_once.pt

bash examples/reflow_res_rl/run_actor.sh
```

Actor 连接 Learner，用 ReFlow + RL 残差采集经验并推送 transition。

---

## 数据与检查点路径

| 内容 | 路径 |
|---|---|
| 专家演示数据 | `serl_data/<env>_<reward>_<n>_fixed_complete.pkl` |
| 残差演示数据 | `examples/reflow_res_rl/demo_data/<env>_residual.pkl` |
| ReFlow 检查点 | `runs/<env>__reflow__<seed>__<timestamp>/checkpoints/best_success_once.pt` |
| DrQ 检查点 | `--checkpoint_path` 参数（默认 `/tmp/serl_ckpt/<exp_name>`） |
| 评估视频 | `--eval_video_dir` 参数（默认 `/tmp/reflow_res_eval_videos/`） |
| W&B 日志 | 在线上传（加 `--debug` 禁用） |

---

## 环境包装链

```
gym.make(env)
  └─ PotentialBasedRewardWrapper     （可选 PBRS 奖励塑形：r = φ(s') - φ(s)）
       └─ ManiSkillMultiCameraWrapper （obs → {state:(35,), hand_camera:(128,128,3), base_camera:(128,128,3)}）
            └─ _ReFlowResetWrapper   （env.reset() 时同步清空 ReFlow 的 obs 滚动缓冲区）
                 └─ AddPolicyActionWrapper  （a_final = a_base + alpha * a_delta）
                      └─ _GymEnvAdapter    （gymnasium.Env 兼容适配层）
                           └─ ChunkingWrapper(obs_horizon=1)  （为 DrQ 添加前导维度）
```

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--reflow_ckpt` | Actor 必填 | ReFlow `.pt` 检查点路径 |
| `--alpha_scale` | `0.1` | 残差动作缩放系数 |
| `--reflow_obs_horizon` | `2` | 观测窗口长度（需与训练一致） |
| `--reflow_pred_horizon` | `4` | 预测时域长度（需与训练一致） |
| `--reflow_act_horizon` | `4` | 动作块大小 |
| `--reward_mode` | `normalized_dense` | `normalized_dense` / `dense` / `sparse` |
| `--potential_reward_shaping` | `True` | 是否启用 PBRS |
| `--max_steps` | `1000000` | Actor 总交互步数 |
| `--batch_size` | `256` | Learner 每次更新的 batch 大小 |
| `--server_port` | `5490` | AgentLace 服务端口 |
| `--broadcast_port` | `5491` | AgentLace 广播端口 |

---

## 网络结构

```
环境观测：
  state: (1, 35)   hand_camera: (1, 128, 128, 3)   base_camera: (1, 128, 128, 3)
         │
         ▼  （ReFlowPolicy 内部）
  8 维末端执行器状态 + 2×ResNet18 视觉特征
         │
         ▼
  ReFlowMLP  ──►  a_base（7 维，chunk 长度 act_horizon=4）
         │
         ▼
  DrQAgent（JAX，resnet-pretrained 编码器）  ──►  a_delta（7 维）
         │
         ▼
  a_final = a_base + 0.1 × a_delta  ──►  env.step()
```

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `reflow_policy.py` | 本地 PyTorch ReFlow 策略（VisualEncoder + ReFlowMLP） |
| `async_drq_sim.py` | Actor / Learner 主脚本（DrQ + ReFlow 残差） |
| `make_residual_demos.py` | 将专家演示转换为残差演示 |
| `run_learner.sh` | Learner 启动脚本 |
| `run_actor.sh` | Actor 启动脚本 |
