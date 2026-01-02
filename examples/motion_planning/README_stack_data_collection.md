# Stack 任务数据采集说明

本目录包含两个用于采集 stack 任务数据的脚本。

## 脚本说明

### 1. collect_stack_trajs_simple.py (推荐)

**特点：**
- 代码简洁，逻辑清晰
- 直接导入 `PandaStackGymEnv`，不重新定义
- 更好的运动规划算法，避免震荡
- 完整的异常处理
- 支持可视化渲染

**8个任务阶段：**
- 阶段 0: 移动到 block 正上方
- 阶段 1: 下降接近 block
- 阶段 2: 关闭夹爪抓取
- 阶段 3: 抬起到安全高度 (15cm)
- 阶段 4: 水平移动到 pillar 上方
- 阶段 5: 下降到 pillar 顶部
- 阶段 6: 打开夹爪放下
- 阶段 7: 抬起离开

**使用方法：**
```bash
cd /home/chenhaojun/workspace_cys/serl/examples
python collect_stack_trajs_simple.py
```

**参数调整：**
在脚本中可以修改：
- `num_trajectories`: 收集的轨迹数量（默认 20）
- `render_visual`: 是否显示可视化（默认 True）

### 2. collect_stack_trajs_motion_planning_v2.py

**特点：**
- 完整的环境定义
- 参考 pick cube 的运动规划策略
- 8个清晰的任务阶段
- 支持可视化渲染

**使用方法：**
```bash
cd /home/chenhaojun/workspace_cys/serl/examples
python collect_stack_trajs_motion_planning_v2.py
```

**参数调整：**
在脚本中可以修改：
- `max_transition_data_num`: 收集的轨迹数量（默认 20）

## 数据保存位置

两个脚本都会将数据保存到：
```
data/stack_trajs/panda_stack_{num}/
├── demo_data.pkl          # 完整数据
├── act_0.pkl              # 阶段 0 数据
├── act_1.pkl              # 阶段 1 数据
├── act_2.pkl              # 阶段 2 数据
├── act_3.pkl              # 阶段 3 数据
├── act_4.pkl              # 阶段 4 数据
├── act_5.pkl              # 阶段 5 数据
├── act_6.pkl              # 阶段 6 数据
└── act_7.pkl              # 阶段 7 数据
```

## 数据格式

每个转换包含以下字段：
- `observations`: 当前观察
- `actions`: 执行的动作 [x, y, z, gripper]
- `next_observations`: 下一步观察
- `rewards`: 奖励
- `masks`: 1 - done
- `dones`: 是否结束
- `task_stage`: 任务阶段 (0-7)

## 渲染说明

- 使用 `render_mode="human"` 会显示可视化窗口
- 渲染在 `_compute_observation()` 中自动完成
- **不要手动调用 `env.render()`**，这会导致 EGL 冲突错误

## 故障排除

### EGLError
如果遇到 `EGLError: EGL_BAD_ACCESS` 错误：
- 确保没有手动调用 `env.render()`
- 使用 `render_mode="rgb_array"` 代替 `"human"`（不显示可视化）

### 轨迹采集失败
- 检查 block 和 pillar 的初始位置是否合理
- 调整运动规划参数（速度、死区等）
- 查看终端输出的错误信息

## 建议

1. **首次使用**：建议先用 `num_trajectories=5` 测试
2. **批量采集**：确认没问题后再增加轨迹数量
3. **可视化**：开发调试时开启 `render_mode="human"`
4. **批量生产**：大量采集时使用 `render_mode="rgb_array"` 提高速度
