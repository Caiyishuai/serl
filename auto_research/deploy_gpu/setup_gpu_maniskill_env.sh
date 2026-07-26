#!/usr/bin/env bash
# =============================================================================
# GPU 机（Linux x86_64）上重建 ManiSkill 数据采集环境
# =============================================================================
# 与 SERL 训练环境是【两个独立 venv】，因为依赖冲突（sapien/jax）。
# 这个环境负责：下载官方 motionplanning demo -> 回放导出 state+reward -> 转 pkl。
#
# ★ Linux 相比 mac 的关键优势：mplib 能装！
#   mac 上 mplib 依赖 libclang==11.0.1 只有 Linux x86 wheel，装不上；
#   所以 mac 只能“回放官方已有 demo”。Linux 上可以跑真正的 motionplanning 重新生成轨迹。
#
# 用法：
#   MANISKILL_ROOT=$HOME/maniskill-ws bash setup_gpu_maniskill_env.sh
# 前置：nvidia-smi 正常，CUDA 12。
# =============================================================================
set -euo pipefail

MANISKILL_ROOT="${MANISKILL_ROOT:-$HOME/maniskill-ws}"
VENV_DIR="${VENV_DIR:-$MANISKILL_ROOT/auto_research/venv_ms_gpu}"
PYBIN="${PYBIN:-python3.11}"

echo "==> MANISKILL_ROOT = $MANISKILL_ROOT"
echo "==> VENV_DIR       = $VENV_DIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || { echo "!! 无 GPU"; exit 1; }

$PYBIN -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# ManiSkill3（会带 sapien 的 Linux GPU 版；Linux 上 SAPIEN 直接用 CUDA/Vulkan，无需 MoltenVK）
pip install mani_skill

# 运动规划库（Linux x86 可装，mac 不行）——用于重新采集
pip install mplib

# 验证
python -c "import mani_skill, sapien; print('mani_skill OK, sapien', sapien.__version__)"
python -c "import mplib; print('mplib OK')" || echo "!! mplib 装失败，检查 libclang"

cat <<EOF

============================================================
✅ ManiSkill 采集 venv 建好了： $VENV_DIR

重新采集 4 个任务数据（Linux 上不需要 MoltenVK 那套环境变量）：

for TASK in PushCube-v1 PickCube-v1 StackCube-v1 PullCubeTool-v1; do
  python -m mani_skill.utils.download_demo \$TASK
  python -m mani_skill.trajectory.replay_trajectory \\
    --traj-path ~/.maniskill/demos/\$TASK/motionplanning/trajectory.h5 \\
    -o state --use-env-states --record-rewards --count 100
done

# 再用 convert_demo_h5.py 转成两种 pkl（trainer dict + SERL stacked）：
python $MANISKILL_ROOT/auto_research/scripts/convert_demo_h5.py --help

⚠️ 注意：第 4 个任务 mac 上因 PokeCube 无 motionplanning demo 改用了 PullCubeTool。
   Linux + mplib 可用后，如果你想用回论文原设定的 PokeCube，可以自己跑 motionplanning
   重新生成 PokeCube 成功轨迹（mac 上做不到，Linux 可以）。
============================================================
EOF
