# Demo 数据格式修复指南

## 问题诊断

你遇到的问题是 demo 数据和运行时环境的键名不匹配。

### Demo 数据格式（从 Franka 环境采集）
```python
{
    "observations": {
        "state": (1, 7),           # ✅ 正确
        "front": (1, H, W, 3),     # ❌ 键名应该是 "base_camera"
        "wrist": (1, H, W, 3),     # ❌ 键名应该是 "hand_camera"
    },
    "next_observations": {...},
    "actions": (action_dim,),
    "rewards": float,
    "masks": float,
    "dones": bool
}
```

### ManiSkill 环境格式
```python
{
    "observations": {
        "state": (1, state_dim),
        "base_camera": (1, H, W, 3),    # ✅ ManiSkill 使用的键名
        "hand_camera": (1, H, W, 3),    # ✅ ManiSkill 使用的键名
    },
    ...
}
```

## 解决方案

使用 `fix_demo_format.py` 脚本来重命名键名。

### 使用方法

```bash
# 基本用法：重命名图像键
python fix_demo_format.py <输入文件.pkl> \
    --key_mapping front:base_camera,wrist:hand_camera \
    --output <输出文件.pkl>

# 示例
python fix_demo_format.py \
    /path/to/franka_lift_cube_image_20_trajs.pkl \
    --key_mapping front:base_camera,wrist:hand_camera \
    --output /path/to/franka_lift_cube_image_20_trajs_fixed.pkl
```

### 参数说明

- `<输入文件.pkl>`: 原始 demo 数据文件（必需）
- `--key_mapping old1:new1,old2:new2`: 键名映射（可选）
  - 格式：`旧键名:新键名,旧键名:新键名`
  - 示例：`front:base_camera,wrist:hand_camera`
- `--output <输出文件.pkl>`: 输出文件路径（可选，默认为输入文件名加 `_fixed` 后缀）

## 步骤

1. **找到你的 demo 数据文件**
   ```bash
   # 查找 demo 文件
   find ~/workspace -name "*.pkl" -type f
   ```

2. **运行修复脚本**
   ```bash
   cd /home/caiyishuai/workspace/serl/examples/async_drq_sim
   
   python fix_demo_format.py \
       <你的demo文件路径.pkl> \
       --key_mapping front:base_camera,wrist:hand_camera
   ```

3. **验证修复后的数据**
   脚本会打印出修复前后的格式对比，确保：
   - 键名已经正确映射
   - 所有数据的 shape 保持不变
   - state 的格式是 `(1, state_dim)`
   - 图像的格式是 `(1, H, W, 3)`

4. **在训练脚本中使用修复后的数据**
   ```bash
   # 编辑 run_learner_maniskill.sh
   # 取消注释并更新 demo_path
   --demo_path /path/to/franka_lift_cube_image_20_trajs_fixed.pkl
   ```

## 注意事项

1. **数据格式本身没问题**：`(1, H, W, 3)` 是 `ChunkingWrapper(obs_horizon=1)` 的正确输出格式

2. **只需要重命名键名**：主要问题是 demo 数据使用了不同的键名

3. **检查图像分辨率**：确保 demo 数据的图像分辨率与 ManiSkill 环境一致
   - ManiSkill 默认：128x128 或 256x256（取决于配置）
   - Franka 环境：检查你采集时的设置

4. **state 维度**：
   - Demo 数据 state: `(1, 7)` - [tcp_pos(3), tcp_vel(3), gripper_pos(1)]
   - ManiSkill 环境：可能有不同的 state 维度
   - 如果维度不匹配，需要调整 demo 数据或环境配置

## 常见问题

### Q1: 键名不匹配怎么办？
A: 使用 `--key_mapping` 参数来重命名键。

### Q2: state 维度不匹配怎么办？
A: 需要在 `fix_demo_format.py` 中添加自定义的 state 转换逻辑，或者调整环境的 observation space。

### Q3: 图像分辨率不匹配怎么办？
A: 需要在脚本中添加图像 resize 功能，使用 `cv2.resize()` 调整到目标分辨率。

### Q4: 如何验证修复是否成功？
A: 脚本会打印详细的格式信息。你也可以手动加载并检查：
```python
import pickle
with open('fixed_demo.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data[0].keys())
print(data[0]['observations'].keys())
for key, value in data[0]['observations'].items():
    print(f"{key}: {value.shape}")
```

## 下一步

修复完 demo 数据后：

1. 更新 `run_learner_maniskill.sh` 中的 `--demo_path`
2. 启动 learner：`bash run_learner_maniskill.sh`
3. 启动 actor：`bash run_actor_maniskill.sh`
4. 监控训练日志，确保 demo buffer 正确加载
