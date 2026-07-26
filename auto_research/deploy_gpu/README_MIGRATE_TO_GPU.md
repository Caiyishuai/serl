# 迁移到 GPU 机器（H20 八卡 / CUDA 12 / Linux）指南

> 一句话结论：**venv 不能直接搬，代码/数据/文档可以搬；到 GPU 机上用本目录脚本重建环境即可。**

---

## 为什么 venv 不能直接搬

当前 mac 上的 `venv_serl` 是 **macOS ARM64 + CPU-only** 的：
- `jaxlib 0.4.35` 是 mac ARM 的 **CPU wheel**（`jax.devices()` 只有 `CpuDevice`，无任何 `nvidia-*` 包）；
- `mujoco / numpy / ml_dtypes` 等都是 mac ARM 的 **C 扩展二进制**；
- venv 里还写死了 mac 的绝对路径。

搬到 Linux + CUDA 的 H20 上，这些二进制**架构和 OS 都不对，import 就会失败**。所以正确做法是**在 GPU 机重建 venv**，而 jax 是设备无关的——**同一份训练代码在 GPU 上自动用 CUDA，训练逻辑一行都不用改。**

## 什么能直接搬 / 什么要重建

| 类别 | 能否搬 | 说明 |
|------|:---:|------|
| 训练脚本 `auto_research/scripts/*.py` | ✅ | 纯 Python |
| 专家数据 `*_expert_serl.pkl` | ✅ | numpy pickle（GPU 机重采则更好，见下） |
| 文档 md（HOWTO / RESEARCH_LOG） | ✅ | — |
| **venv_serl 目录** | ❌ | mac CPU 二进制，重建 |
| jaxlib / mujoco / numpy 等二进制包 | ❌ | 重装 Linux+CUDA 版 |

## 迁移三步

### ① 同步代码与仓库
把 `serl/`（含 `serl_launcher/`、`auto_research/`）和 `maniskill-ws/` 同步/checkout 到 GPU 机。
**不要** 把 `venv_serl*` 目录同步过去（加进 .gitignore / rsync --exclude）。

### ② 建 SERL 训练环境（H20 侧）
```bash
cd serl/auto_research/deploy_gpu
SERL_ROOT=$HOME/serl bash setup_gpu_env.sh
```
脚本会：建 venv → 装 `jax[cuda12]==0.4.35` → 装锁定依赖（`requirements-gpu.txt`）→
editable 装 `serl_launcher` → **红线自检（禁止 tensorflow、禁止 numpy 降级）** → 验证 `jax.devices()` 是 `CudaDevice`。

### ③ 建 ManiSkill 采集环境（你选了 GPU 机重新采集）
```bash
cd serl/auto_research/deploy_gpu
MANISKILL_ROOT=$HOME/maniskill-ws bash setup_gpu_maniskill_env.sh
```
Linux 上 **mplib 能装**（mac 装不上），所以可跑真正的 motionplanning 重新生成轨迹；
且 Linux 的 SAPIEN 直接用 CUDA/Vulkan，**不需要 mac 那套 MoltenVK 环境变量**。

## 跑训练（命令与 mac 完全一致）
建好后把 `HOWTO_run_matrix16.md` 里的 `SVENV` 指向新 venv 即可，16 组矩阵脚本、画曲线命令原样可用。

## H20 专属提醒（Hopper sm_90）
- H20 是 Hopper 架构，**必须 CUDA 12**（jax[cuda11] 不支持）。
- `jaxlib 0.4.35` 相对 Hopper 偏老。若报 `sm_90` 相关警告或性能异常，按 `setup_gpu_env.sh` 末尾「备选」升级：
  `pip install -U "jax[cuda12]" flax optax orbax-checkpoint` 后重跑自检。
- **八卡但当前脚本是单进程单卡**：纯离线 RLPD 训练很轻（1200 updates 几十秒），单卡足够。
  想吃满八卡需要改成多进程/`pmap` 或并行跑多组实验（每卡一组），这属于扩展项，需要时再做。

## 唯一的红线（务必记住）
**不要装 tensorflow。** serl_launcher 深层 import 它只用类型别名，脚本已注入 stub 绕过；
真装 tf 会把 numpy 降到 <2、ml_dtypes 降到 0.3，直接摧毁 jax。setup 脚本里已加了自动检查拦截。

## 本目录文件
- `README_MIGRATE_TO_GPU.md` —— 本文件
- `setup_gpu_env.sh` —— SERL 训练环境一键重建
- `setup_gpu_maniskill_env.sh` —— ManiSkill 采集环境一键重建
- `requirements-gpu.txt` —— SERL 侧锁定依赖（不含 jaxlib，jax GPU 版在脚本里单独装）
- `requirements-mac-cpu-freeze.txt` —— mac 当前完整 freeze（参考锚点，记录验证过的确切版本）
