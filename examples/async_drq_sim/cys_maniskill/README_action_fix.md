# Demo 数据导致收敛变慢 - 问题诊断与解决方案

## 🔍 问题诊断报告

### 发现的问题

通过调试分析，发现了导致加入 demo 数据后训练收敛变慢的根本原因：

#### **核心问题：Actions 分布严重不匹配**

**原始 Demo 数据的 Actions 统计：**
```
整体统计:
  Min: -6.9208  ⚠️ 远超 [-1, 1]
  Max: 4.3659   ⚠️ 远超 [-1, 1]
  Mean: -0.1503
  Std: 1.8029

各维度统计:
  维度 0: min=-1.18, max=2.01   (15.10% 超出范围)
  维度 1: min=-1.46, max=1.73   (4.70% 超出范围)
  维度 2: min=-2.10, max=0.97   (16.22% 超出范围)
  维度 3: min=-2.74, max=3.08   (33.69% 超出范围)
  维度 4: min=-5.51, max=2.34   (48.23% 超出范围) ⚠️
  维度 5: min=-1.41, max=0.55   (5.82% 超出范围)
  维度 6: min=-6.92, max=4.37   (95.90% 超出范围) ⚠️⚠️⚠️

总计：99.88% 的 transitions 包含超出 [-1, 1] 范围的 actions！
```

**在线训练的 Actions 分布：**
```
DrQ/SAC 使用 TanhNormal Distribution
- 输出自动限制在 [-1, 1] 范围内
- 所有 actions 都在有效范围内
```

### 问题根源分析

#### 1. **PPO 采集 Demo 时未约束 Actions**

在你的采集代码 `3_collect_data_maniskill_ppo_checkpoints.py` 第 209 行：

```python
# PPO 直接输出 actor_mean，没有 tanh squashing
action = agent.actor_mean(state_tensor)
env.step(action.cpu().numpy())
```

PPO 的 actor 输出是**无界的**（使用 Normal distribution），没有被限制到 [-1, 1]。

#### 2. **DrQ/SAC 使用 Tanh Squash Distribution**

在线训练时，DrQ agent 的 `sample_actions` 方法使用 `TanhNormal` 分布：

```python
# SAC/DrQ 的 forward_policy 返回 TanhNormal
dist = self.forward_policy(observations, rng=seed, train=False)
actions = dist.sample(seed=seed)  # 自动限制在 [-1, 1]
```

#### 3. **后果：梯度冲突和训练不稳定**

- **Demo actions**: 范围 [-6.92, 4.37]，严重超出 [-1, 1]
- **Online actions**: 范围 [-1, 1]（tanh squash）
- **结果**: 
  - 策略网络在两个完全不同的目标之间震荡
  - 梯度方向冲突
  - 训练不稳定，收敛变慢甚至发散

---

## ✅ 解决方案

### 已执行的修复

使用 `5_fix_demo_actions.py` 脚本，将 demo actions clip 到 [-1, 1]：

```bash
cd /home/caiyishuai/workspace/serl/examples/async_drq_sim/cys_maniskill
python 5_fix_demo_actions.py \
    --input /home/caiyishuai/workspace/maniskill-ws/serl_data/mani_skill_place_sphere_100.pkl \
    --output serl_data/mani_skill_place_sphere_100_fixed.pkl
```

**修复结果：**
```
修改了 2487/2490 条 transitions (99.88%)

修复后的 Actions 统计:
  Min: -1.0000 ✓
  Max: 1.0000  ✓
  Mean: -0.0273
  Std: 0.6207

所有维度都在 [-1, 1] 范围内 ✓
包含超出范围 actions 的 transitions: 0/2490 (0.00%) ✓
```

### 使用修复后的数据

更新训练脚本 `run_learner_maniskill.sh`:

```bash
#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
python async_drq_sim_maniskill.py "$@" \
    --learner \
    --exp_name=place_sphere_100_demo_fixed \
    --env PlaceSphere-v1 \
    --obs_mode rgb+state \
    --control_mode pd_ee_delta_pose \
    --seed 0 \
    --batch_size 256 \
    --encoder_type resnet-pretrained \
    --robot_uids panda_wristcam \
    --demo_path cys_maniskill/serl_data/mani_skill_place_sphere_100_fixed.pkl
```

**或者使用相对于训练脚本的路径：**
```bash
--demo_path examples/async_drq_sim/cys_maniskill/serl_data/mani_skill_place_sphere_100_fixed.pkl
```

---

## 📊 预期效果

修复后，你应该能看到：

### 训练稳定性改善
- ✅ Critic loss 更加平滑
- ✅ Actor loss 不再剧烈震荡
- ✅ Q 值估计更准确
- ✅ 收敛速度加快

### Actions 分布一致
- ✅ Demo 和 online actions 都在 [-1, 1] 范围内
- ✅ 策略梯度方向一致
- ✅ 更好地利用 demo 数据

---

## 🛠️ 未来改进建议

### 方案 1: 在采集 Demo 时直接 Clip Actions

修改 `3_collect_data_maniskill_ppo_checkpoints.py`:

```python
# 第 209 行附近
with torch.no_grad():
    action = agent.actor_mean(state_tensor)
    # 添加 clip 到 [-1, 1]
    action = torch.clamp(action, -1.0, 1.0)

obs_full, rew, terminated, truncated, info = env.step(action.cpu().numpy())
```

### 方案 2: 使用 Tanh Squashing 训练 PPO

修改 PPO 的 actor 输出，使其也使用 tanh squashing：

```python
# 在 Agent 的 forward 中
action_mean = self.actor_mean(obs)
action_mean = torch.tanh(action_mean)  # 限制到 [-1, 1]
```

### 方案 3: 归一化 Actions

如果环境的 action_space 不是 [-1, 1]，可以添加归一化：

```python
def normalize_action(action, action_space):
    """将 action 从环境空间归一化到 [-1, 1]"""
    low = action_space.low
    high = action_space.high
    return 2.0 * (action - low) / (high - low) - 1.0

def denormalize_action(action, action_space):
    """将 action 从 [-1, 1] 反归一化到环境空间"""
    low = action_space.low
    high = action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)
```

---

## 📝 总结

### 问题
- PPO 采集的 demo actions 未经约束，范围 [-6.92, 4.37]
- DrQ 在线训练使用 tanh squash，范围 [-1, 1]
- 分布严重不匹配导致训练不稳定

### 解决
- ✅ 使用 `5_fix_demo_actions.py` clip demo actions 到 [-1, 1]
- ✅ 生成修复后的文件: `mani_skill_place_sphere_100_fixed.pkl`
- ✅ 所有 actions 现在都在 [-1, 1] 范围内

### 下一步
1. 使用修复后的 demo 数据重新训练
2. 监控训练指标（critic_loss, actor_loss, Q values）
3. 对比训练曲线，验证改进效果

---

## 🔍 调试命令

如果需要进一步调试，可以使用以下命令：

```python
# 在训练脚本中添加 actions 统计
if demo_iterator is not None:
    demo_batch = next(demo_iterator)
    print(f"Demo actions: min={demo_batch['actions'].min():.4f}, "
          f"max={demo_batch['actions'].max():.4f}, "
          f"mean={demo_batch['actions'].mean():.4f}")
    batch = concat_batches(batch, demo_batch, axis=0)
```

记录修复前后的训练指标对比，验证改进效果。
