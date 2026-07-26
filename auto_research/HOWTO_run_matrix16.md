# HOWTO：复现 16 组 SERL 离线训练矩阵 + 训练曲线

> 目标读者：拿到本项目、想在 **macOS (Apple Silicon, 无 CUDA)** 上一步步复现
> 「ManiSkill 4 任务 × {dense, sparse} × {fixed τ, adaptive τ} = 16 组纯离线 RLPD 训练」
> 并画出训练曲线的人。照着从上往下执行即可。
>
> 关联文档：`00_RESEARCH_LOG.md`（完整实验流水与背景推理）。本文件只讲「怎么跑通」。

---

## 0. 这套实验在做什么

- **数据**：用 ManiSkill3 官方 motionplanning demo 回放出的**专家成功轨迹**（state 观测 + reward），
  转成 SERL 可读的 pkl。4 个任务：PushCube / PickCube / StackCube / PullCubeTool。
- **训练**：纯离线（demo-only）RLPD/SAC，只用 demo buffer，不与环境在线交互
  （因为 ManiSkill env 依赖 sapien，只在 maniskill venv 可用；SERL venv 无法在线交互）。
- **两个维度各 2 档**：
  - reward：`dense`（原始连续奖励） vs `sparse`（相对阈值二值化）
  - τ（target 软更新系数）：`fixed`（固定 0.005） vs `adaptive`（Rsync 机制，随 critic TD 误差在 [0.005, 0.2] 动态调整）
- 4 × 2 × 2 = **16 组**。每组 1200 updates，串行总耗时约 9 分钟。

---

## 1. 前置条件（环境 & 数据）

### 1.1 SERL venv（已建好，无需重装）
- 路径：`/Users/yishuaicai/mywork/serl/auto_research/venv_serl`
- Python 3.11.9
- 关键依赖版本（**必须对齐，否则 jax 会崩**）：
  ```
  jax 0.4.35 | numpy 2.1.3 | ml_dtypes 0.5.0
  jax devices = [CpuDevice(id=0)]   # CPU 后端, mac 无 CUDA
  ```
- ⚠️ **切勿在此 venv 装 tensorflow**：会把 numpy 降到 <2、ml_dtypes 降到 0.3，直接摧毁 jax 0.4.35。
  训练脚本顶部已注入 **tensorflow / agentlace 轻量 stub** 绕过 serl_launcher 的深层 import，不需要真的装它们。

### 1.2 四个专家数据集（已生成，无需重跑）
路径：`/Users/yishuaicai/mywork/maniskill-ws/auto_research/data/`
```
ms_pushcube_expert_serl.pkl       obs_dim=35   ~6.9k  transitions   reward max=4
ms_pickcube_expert_serl.pkl       obs_dim=42   ~7.7k  transitions   reward max=5
ms_stackcube_expert_serl.pkl      obs_dim=48   ~10.7k transitions   reward max=8
ms_pullcubetool_expert_serl.pkl   obs_dim=39   ~25k   transitions   reward max=10.23
```
`*_serl.pkl` 是 SERL 原生 stacked-array 格式（observations/actions/rewards/... 各一个大数组）。

> 如果这些 pkl 不存在，需要先在 **maniskill venv** 里重新生成（见第 5 节「从零生成数据」）。
> 正常复现训练不需要这步。

---

## 2. 一键跑完 16 组（含曲线记录）

在**任意目录**执行（脚本用绝对路径，cd /tmp 只是避免污染工作目录）：

```bash
SVENV=/Users/yishuaicai/mywork/serl/auto_research/venv_serl
SCRIPT=/Users/yishuaicai/mywork/serl/auto_research/scripts/train_serl_offline_maniskill.py
BASE=/Users/yishuaicai/mywork/maniskill-ws/auto_research/data
CURVED=/Users/yishuaicai/mywork/serl/auto_research/logs/curves
mkdir -p $CURVED
cd /tmp

for name in pushcube pickcube stackcube pullcubetool; do
  DATA=$BASE/ms_${name}_expert_serl.pkl
  for rmode in dense sparse; do
    for tmode in fixed adaptive; do
      SP=""; [ "$rmode" = "sparse" ]   && SP="--sparse_reward"
      AT=""; [ "$tmode" = "adaptive" ] && AT="--adaptive_tau"
      CSV=$CURVED/curve_${name}_${rmode}_${tmode}.csv
      $SVENV/bin/python $SCRIPT --data $DATA \
        --max_updates 1200 --time_limit_min 4 \
        --record_period 10 --metrics_csv $CSV $SP $AT \
        > /tmp/curve_${name}_${rmode}_${tmode}.log 2>&1
      echo "done ${name} ${rmode} ${tmode} -> $(wc -l < $CSV) rows"
    done
  done
done
echo "=== ALL 16 DONE ==="
```

**预期**：16 行 `done ... -> 121 rows`，无报错。每组末尾日志有一行
`[done] wall=..s updates=1200 final_critic_loss=... bc_action_mse=...`。

---

## 3. 单组怎么跑（调试 / 只看某一组）

```bash
SVENV=/Users/yishuaicai/mywork/serl/auto_research/venv_serl
SCRIPT=/Users/yishuaicai/mywork/serl/auto_research/scripts/train_serl_offline_maniskill.py
BASE=/Users/yishuaicai/mywork/maniskill-ws/auto_research/data

# 例：StackCube + sparse reward + adaptive τ
$SVENV/bin/python $SCRIPT \
  --data $BASE/ms_stackcube_expert_serl.pkl \
  --max_updates 1200 --time_limit_min 4 \
  --sparse_reward --adaptive_tau \
  --record_period 10 --metrics_csv /tmp/one.csv
```

### 关键参数速查
| 参数 | 含义 | 默认 |
|------|------|------|
| `--data` | SERL 格式 pkl 路径 | 必填 |
| `--max_updates` | 训练步数 | 1200 |
| `--time_limit_min` | 墙钟保护（超时提前停） | 4 |
| `--sparse_reward` | 开启则把 reward 二值化 | 关（=dense） |
| `--sparse_frac` | sparse 阈值 = frac × 该数据集 reward 最大值 | 0.7 |
| `--adaptive_tau` | 开启 Rsync 自适应 τ | 关（=fixed τ=0.005） |
| `--metrics_csv` | 输出曲线 CSV 路径 | 不写 |
| `--record_period` | 每几步记一次曲线点 | 10 |
| `--utd_ratio` | 高 UTD 更新比 | 4 |
| `--batch_size` | batch | 256 |

### sparse reward 为什么用「相对阈值」
各任务 dense reward 量级差异大（max 4~10.23），固定绝对阈值不通用。
脚本用 `thr = sparse_frac × 该数据集 reward.max()`，`reward >= thr → 1 else 0`。
因为用的是**专家成功轨迹**，二值化后各任务仍有 7%~12% 正样本，能真正学习（不会像随机数据那样恒 0 学不动）。

### adaptive τ（Rsync 机制）
critic TD 误差大 → τ 升到上限 0.2（target 快追）；误差小 → 回落下限 0.005（稳定）。
脚本内配置：`critic_loss_threshold=0.3, tau_min=0.005, tau_max=0.2`。

---

## 4. 画训练曲线

16 组 CSV 齐了之后：

```bash
SVENV=/Users/yishuaicai/mywork/serl/auto_research/venv_serl
$SVENV/bin/python /Users/yishuaicai/mywork/serl/auto_research/scripts/plot_curves.py
```

产出三张图到 `serl/auto_research/logs/`：
- `curves_critic_loss.png` —— 4 任务子图，每图 4 条曲线（dense/sparse × fixed/adaptive τ），critic_loss 对数轴
- `curves_actor_loss.png` —— actor_loss 曲线
- `curves_tau.png` —— adaptive τ 动态（仅 adaptive 组）

> ⚠️ matplotlib 缺中文字体（DejaVu Sans），标题/label 一律用英文，否则出现方框警告。

---

## 5.（可选）从零生成 4 个专家数据集

**这步在 maniskill venv 里做，不是 SERL venv。** 正常复现训练不需要。

```bash
# maniskill venv（有 sapien / ManiSkill3）
MVENV=<你的 maniskill venv>/bin/python

# 每个任务：下载官方 motionplanning demo -> 确定性回放导出 state+reward -> 转 pkl
for TASK in PushCube-v1 PickCube-v1 StackCube-v1 PullCubeTool-v1; do
  $MVENV -m mani_skill.utils.download_demo $TASK
  # 回放导出 state 观测 + reward（MoltenVK 让 SAPIEN 用 Metal 渲染）
  VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
  MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=1 \
  $MVENV -m mani_skill.trajectory.replay_trajectory \
    --traj-path <demo>/trajectory.h5 -o state \
    --use-env-states --record-rewards --count 100
done
# 再用 maniskill-ws/auto_research/scripts/convert_demo_h5.py 把回放 h5 转成两种 pkl
```

### ⚠️ 已知坑（复现时注意）
1. **PokeCube-v1 没有 motionplanning demo**（只有 PPO 的 rl demo，无法回放成成功轨迹）。
   → 第 4 个任务改用 **PullCubeTool-v1**（有 motionplanning trajectory.h5，回放 100/100 success）。
2. **mplib 在 mac 装不上**（依赖 libclang==11.0.1，只有 Linux x86 wheel）。
   → 所以不能本地重新做运动规划，只能**回放官方已有的 demo 轨迹**。
3. **SERL 与 ManiSkill 是两个独立 venv**：ManiSkill 负责出数据（需 sapien），SERL 只吃 pkl 训练。
   两边依赖冲突，不要合并。

---

## 6. 结果自查

跑完确认：
- `logs/curves/curve_*.csv` 共 **16 个**，每个 **121 行**（表头 + 120 点）。
- 每组 `[done]` 行的 `final_critic_loss` 为**有限值**（无 NaN / inf）即算跑通。
  - dense 组在 StackCube/PullCubeTool 上 loss 偏大（几~十几）是 reward 量级大导致，非发散。
  - sparse 组 loss 普遍小（0.01~1）。
- 16 组基准结果（final_critic_loss）见 `00_RESEARCH_LOG.md` 的矩阵表；实测全部 OK、零 FAIL。
