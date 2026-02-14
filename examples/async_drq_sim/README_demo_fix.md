# Demo 数据格式修复说明

## 问题诊断

根据你提供的数据对比,offline demo 数据与 online 采集数据存在以下差异:

### 1. **Actions 归一化问题** ⚠️ (最重要)

- **Online 数据**: actions 在 `[-1, 1]` 范围内(已归一化)
  ```python
  actions: min=-0.999, max=1.000
  ```

- **Demo 数据**: actions 在原始控制空间范围内(未归一化)
  ```python
  actions: min=-6.21, max=4.17
  ```

这会导致训练时的数值不匹配,影响学习效果。

### 2. 图像内容差异

- **Online 数据**: `base_camera` 大部分是黑色 `[0, 0, 0]`
- **Demo 数据**: `base_camera` 大部分是灰色 `[141, 142, 143]`

这可能是背景或渲染设置不同导致的,通常不会造成严重问题。

### 3. Shape 问题(已在 `fix_demo_format.py` 中处理)

- 图像: 需要 `(num_stack, H, W, 3)` shape
- State: 需要 `(1, state_dim)` shape

### 4. `batch[0]` 错误

`FrozenDict` 不支持整数索引,应该用键访问:
```python
# 错误
batch[0]

# 正确
batch['actions'][0]
```

## 解决方案

我为你创建了两个修复脚本:

### 方案 1: 完整修复脚本 (推荐)

`fix_demo_complete.py` - 一次性修复所有问题(actions 归一化 + shape)

**用法示例:**

```bash
# 自动从环境获取 action_space 并修复
python fix_demo_complete.py your_demo.pkl --env PickCube-v1

# 手动指定 action_space
python fix_demo_complete.py your_demo.pkl \
    --action-low -2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175 -2.8973 \
    --action-high 2.8973 1.7628 2.8973 -0.0698 2.8973 3.7525 2.8973

# 指定输出文件和 stack 数量
python fix_demo_complete.py your_demo.pkl \
    --env PickCube-v1 \
    --num-stack 2 \
    --output fixed_demo.pkl

# 自动检测是否需要归一化
python fix_demo_complete.py your_demo.pkl \
    --env PickCube-v1 \
    --auto-detect
```

### 方案 2: 分步修复

如果你想分步处理:

**步骤 1**: 归一化 actions
```bash
python normalize_demo_actions.py your_demo.pkl --env PickCube-v1
```

**步骤 2**: 修复 shape
```bash
python fix_demo_format.py your_demo_normalized.pkl --num_stack 1
```

## 如何找到你的环境的 action_space

### 方法 1: 从代码查看

查看你的环境配置文件或代码:
```python
import gymnasium
env = gymnasium.make('PickCube-v1')
print("action_low:", env.action_space.low)
print("action_high:", env.action_space.high)
```

### 方法 2: 查看 demo 数据的 actions 范围

```bash
python check_demo_format.py your_demo.pkl
```

然后根据输出的 actions 范围,推断原始 action_space。

### 方法 3: 常见环境的 action_space

**Franka 机械臂 (7-DoF)**:
```bash
--action-low -2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175 -2.8973
--action-high 2.8973 1.7628 2.8973 -0.0698 2.8973 3.7525 2.8973
```

**UR5 机械臂 (6-DoF)**:
```bash
--action-low -6.2832 -6.2832 -6.2832 -6.2832 -6.2832 -6.2832
--action-high 6.2832 6.2832 6.2832 6.2832 6.2832 6.2832
```

## 验证修复结果

修复后,你应该看到:

```
=== 修复后 Transition ===

--- observations ---
  base_camera: shape=(1, 128, 128, 3), dtype=uint8
  hand_camera: shape=(1, 128, 128, 3), dtype=uint8
  state: shape=(1, 7), dtype=float32

--- actions ---
  shape=(7,), dtype=float32
  min=-0.998, max=0.999, mean=0.123, std=0.456

所有修复后的 actions 统计:
  min: -1.000
  max: 1.000
  mean: 0.015
  std: 0.512
```

## 常见问题

### Q1: 为什么需要归一化 actions?

A: 因为强化学习训练时,policy 网络输出的是归一化后的 actions ([-1, 1] 范围),然后由环境自动反归一化。如果 demo 数据的 actions 没有归一化,会导致数值不匹配。

### Q2: 我的 demo 数据已经归一化了吗?

A: 运行 `check_demo_format.py` 查看 actions 的范围:
- 如果 min/max 在 [-1, 1] 附近 → 已归一化
- 如果 min/max 超出 [-1.5, 1.5] → 需要归一化

或使用 `--auto-detect` 参数让脚本自动判断。

### Q3: 修复后训练还是有问题怎么办?

A: 检查以下几点:
1. actions 的统计分布是否合理(不应该全是 0 或全是极值)
2. 图像的内容是否正确(不应该全黑或全白)
3. rewards 和 masks 是否正确
4. 使用 wandb 等工具对比 demo batch 和 online batch 的数值范围

### Q4: 如何在代码中使用修复后的数据?

A: 直接替换原来的 demo 文件路径:
```bash
python async_drq_sim_maniskill.py \
    --learner \
    --demo_path /path/to/your_demo_fixed.pkl \
    ...其他参数
```

## 调试技巧

如果训练时仍然有问题,可以在代码中添加调试信息:

```python
# 在 learner 函数中,concat_batches 之前
if demo_iterator is not None:
    demo_batch = next(demo_iterator)
    
    # 调试: 打印 batch 的统计信息
    print("=== Online batch ===")
    print(f"actions: min={batch['actions'].min()}, max={batch['actions'].max()}")
    print(f"rewards: min={batch['rewards'].min()}, max={batch['rewards'].max()}")
    
    print("=== Demo batch ===")
    print(f"actions: min={demo_batch['actions'].min()}, max={demo_batch['actions'].max()}")
    print(f"rewards: min={demo_batch['rewards'].min()}, max={demo_batch['rewards'].max()}")
    
    batch = concat_batches(batch, demo_batch, axis=0)
```

这样可以实时查看两种数据的差异。

## 联系与反馈

如果遇到问题,请提供:
1. `check_demo_format.py` 的输出
2. 修复前后的 actions 统计信息
3. 你的环境名称和 action_space

祝训练顺利! 🚀
