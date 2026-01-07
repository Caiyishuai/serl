# 数据格式说明

## 📋 参考格式

参考文件：`plug_insert_20_demos_2026-01-05_06-37-50.pkl`

### State数组格式 (22维展平数组)

```python
state = np.array([...], shape=(22,), dtype=float32)
```

**索引映射：**
- `[0]`: **gripper_pose** (1维) - 夹爪位置
- `[1:4]`: **tcp_force** (3维) - 末端执行器受力 (Fx, Fy, Fz)
- `[4:10]`: **tcp_pose** (6维) - 末端执行器位姿 (x, y, z, qx, qy, qz)
- `[10:13]`: **tcp_torque** (3维) - 末端执行器力矩 (Tx, Ty, Tz)
- `[13:19]`: **tcp_vel** (6维) - 末端执行器速度 (vx, vy, vz, ωx, ωy, ωz)
- `[19:22]`: **block_pos** (3维) - 物体位置 (x, y, z)

### 轨迹数据结构

```python
trajectory = {
    'observations': {
        'state': np.array(shape=(N, 22), dtype=float32),  # N个时间步的状态
        'front': np.array(shape=(N, H, W, 3), dtype=uint8),  # 前视摄像头图像（可选）
        'wrist': np.array(shape=(N, H, W, 3), dtype=uint8),  # 手腕摄像头图像（可选）
        # ... 其他图像
    },
    'next_observations': {
        'state': np.array(shape=(N, 22), dtype=float32),
        # ... 同observations
    },
    'actions': np.array(shape=(N, action_dim), dtype=float32),
    'rewards': np.array(shape=(N,), dtype=float32),
    'dones': np.array(shape=(N,), dtype=bool),
    'infos': [
        {
            'original_state_obs': {  # 原始的字典格式状态
                'tcp_pose': np.array([...]),       # 6维
                'tcp_vel': np.array([...]),        # 6维
                'gripper_pose': np.array([...]),   # 1维
                'tcp_force': np.array([...]),      # 3维
                'tcp_torque': np.array([...]),     # 3维
                'block_pos': np.array([...]),      # 3维 - 物体位置
            },
            # ... 其他info字段
        },
        # ... N个时间步的info
    ]
}
```

## 🔄 代码修改

### 1. `augment_obs_with_force_torque()` 函数

**修改前：**
- 返回字典格式的state：`{'tcp_pose': ..., 'tcp_vel': ..., ...}`

**修改后：**
- 返回22维展平数组的state
- 同时返回原始字典格式（用于保存到info）

```python
augmented_obs, original_state_dict = augment_obs_with_force_torque(obs, env)

# augmented_obs['state'] 是 shape=(22,) 的numpy数组
# original_state_dict 是字典格式，保存到info中
```

### 2. 数据保存格式转换

添加了 `convert_trajs_to_reference_format()` 函数，将轨迹列表转换为参考格式：

- observations和next_observations从列表转换为字典
- state从列表合并为shape=(N, 19)的数组
- images从列表合并为shape=(N, H, W, 3)的数组
- 其他字段（actions, rewards, dones）转换为numpy数组

## ✅ 验证

运行验证脚本检查格式：

```bash
python examples/async_drq_sim/verify_data_format.py
```

该脚本会对比参考文件和新生成文件的格式，确保一致性。

## 📊 格式对比

| 字段 | 旧格式 | 新格式 |
|------|--------|--------|
| observations | list of dict | dict with arrays |
| state | dict | numpy array (N, 22) |
| images | 每个obs一个dict | 合并为(N, H, W, 3)数组 |
| actions | list | numpy array |
| rewards | list | numpy array |
| dones | list | numpy array |
| infos | list of dict | list of dict (保持) |
| original_state_obs | N/A | 新增到infos中 (含block_pos) |

## 🎯 优势

1. **一致性**：与参考数据格式类似，便于后续处理
2. **效率**：数组格式比列表格式更高效，访问速度更快
3. **兼容性**：保留原始字典格式在info中，便于调试和理解
4. **可扩展**：易于添加新的传感器数据（如额外的图像视角）
5. **完整性**：包含物体位置信息（block_pos），支持任务相关的分析

## 📍 物体位置信息 (block_pos)

物体位置信息被包含在state数组的最后3维 `[19:22]`，同时也保存在 `info['original_state_obs']['block_pos']` 中。

**用途：**
- 监控任务进度（如物体是否被抓取、提升高度等）
- 计算奖励（基于物体位置的距离奖励）
- 分析成功/失败原因
- 验证任务完成条件

**坐标系：**
- `x`: 前后方向（桌面坐标系）
- `y`: 左右方向
- `z`: 垂直方向（高度）

**示例值：**
- 初始位置：`[0.4, 0.0, 0.02]` (桌面上)
- 成功抓取后：`[0.4, -0.1, 0.42]` (提升约40cm)

## 📝 使用示例

```python
import pickle as pkl
import numpy as np

# 加载数据
with open('success_trajs_20_force_dense.pkl', 'rb') as f:
    trajs = pkl.load(f)

# 访问第一条轨迹的状态
traj = trajs[0]
states = traj['observations']['state']  # shape: (N, 22)

# 解析状态
gripper_poses = states[:, 0]           # 所有时间步的夹爪位置
tcp_forces = states[:, 1:4]            # 所有时间步的末端力
tcp_poses = states[:, 4:10]            # 所有时间步的末端位姿
tcp_torques = states[:, 10:13]         # 所有时间步的末端力矩
tcp_vels = states[:, 13:19]            # 所有时间步的末端速度
block_positions = states[:, 19:22]     # 所有时间步的物体位置

# 访问原始字典格式（在info中）
info = traj['infos'][0]
original_state = info['original_state_obs']
print(f"TCP Force: {original_state['tcp_force']}")
print(f"Block Position: {original_state['block_pos']}")
```

## 🔧 生成数据

使用更新后的脚本生成符合格式的数据：

```bash
cd /home/chenhaojun/workspace_cys/serl
python examples/async_drq_sim/1_eval_load_model_and_generate_data_force.py
```

生成的文件：
- `success_trajs_20_force_dense.pkl` - 成功轨迹
- `fail_trajs_20_force_dense.pkl` - 失败轨迹
- `eval_success_video.mp4` - 成功案例视频
- `eval_fail_video.mp4` - 失败案例视频

