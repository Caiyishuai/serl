# 数据格式修改总结

## 🔄 关键变更

### State维度：19维 → 22维

添加了物体位置信息（block_pos）到state数组的末尾。

### 新的State格式 (22维)

```python
state[22] = [
    gripper_pose[1],      # [0]     - 夹爪位置
    tcp_force[3],         # [1:4]   - 末端力
    tcp_pose[6],          # [4:10]  - 末端位姿
    tcp_torque[3],        # [10:13] - 末端力矩
    tcp_vel[6],           # [13:19] - 末端速度
    block_pos[3],         # [19:22] - 物体位置 ⭐新增
]
```

## 📦 数据结构

### observations格式
```python
{
    'state': np.array(shape=(N, 22), dtype=float32),
    'images': {...}  # 可选
}
```

### infos格式
```python
{
    'original_state_obs': {
        'tcp_pose': array[6],
        'tcp_vel': array[6],
        'gripper_pose': array[1],
        'tcp_force': array[3],
        'tcp_torque': array[3],
        'block_pos': array[3],  # ⭐新增
    },
    # ... 其他字段
}
```

## ✨ 新增功能

1. **物体位置追踪**: state数组最后3维 `[19:22]` 包含实时物体位置
2. **双重存储**: 物体位置既在state数组中，也在 `info['original_state_obs']['block_pos']` 中
3. **便于分析**: 可直接从state数组切片获取所有时间步的物体轨迹

## 💡 使用示例

```python
import pickle as pkl
import numpy as np

# 加载数据
with open('success_trajs_20_force_dense.pkl', 'rb') as f:
    trajs = pkl.load(f)

traj = trajs[0]
states = traj['observations']['state']  # (N, 22)

# 提取物体位置轨迹
block_trajectory = states[:, 19:22]  # (N, 3)

# 计算物体提升高度
initial_height = block_trajectory[0, 2]
final_height = block_trajectory[-1, 2]
lift_height = final_height - initial_height

print(f"物体提升高度: {lift_height:.3f} m")

# 从info中获取详细信息
info = traj['infos'][0]
block_pos = info['original_state_obs']['block_pos']
print(f"初始物体位置: {block_pos}")
```

## 🔍 验证

运行验证脚本检查格式：

```bash
python examples/async_drq_sim/verify_data_format.py
```

## 📋 相关文件

- `1_eval_load_model_and_generate_data_force.py` - 数据收集脚本（已更新）
- `verify_data_format.py` - 格式验证脚本（已更新）
- `DATA_FORMAT_README.md` - 完整格式文档（已更新）
- `analyze_pkl_data.py` - 数据分析脚本

## ⚠️ 重要提示

所有旧的19维数据将与新的22维数据**不兼容**。需要重新生成数据以使用新格式。

```bash
# 重新生成数据
python examples/async_drq_sim/1_eval_load_model_and_generate_data_force.py
```

生成的新文件将包含22维state数组和完整的物体位置信息。

