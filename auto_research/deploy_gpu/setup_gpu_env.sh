#!/usr/bin/env bash
# =============================================================================
# 在 GPU 机器（H20 / CUDA 12 / Linux x86_64）上一键重建 SERL 训练环境
# =============================================================================
# 背景：mac 上的 venv 是 CPU-only + ARM64 二进制，无法直接搬到 GPU 机。
#       代码 / 数据 / 文档可直接搬，但 Python 环境必须在 GPU 机上重建。
#
# 用法：
#   1) 把整个 serl/ 和 maniskill-ws/ 目录 checkout / 同步到 GPU 机
#   2) cd serl/auto_research/deploy_gpu
#   3) bash setup_gpu_env.sh            # 建 SERL 训练 venv
#   4) 按最后打印的验证命令确认 jax 认到 GPU
#
# 前置：GPU 机已装好 NVIDIA 驱动 + CUDA 12 runtime，`nvidia-smi` 正常。
# =============================================================================
set -euo pipefail

# ---- 按需修改这两个路径 ----
SERL_ROOT="${SERL_ROOT:-$HOME/serl}"          # serl 仓库根目录
VENV_DIR="${VENV_DIR:-$SERL_ROOT/auto_research/venv_serl_gpu}"
PYBIN="${PYBIN:-python3.11}"                   # 建议 3.11，与 mac 端一致

echo "==> SERL_ROOT = $SERL_ROOT"
echo "==> VENV_DIR  = $VENV_DIR"

# ---- 0. 前置检查 ----
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || {
  echo "!! 未检测到 nvidia-smi，请确认在 GPU 机上且驱动已装"; exit 1; }
[ -d "$SERL_ROOT/serl_launcher" ] || {
  echo "!! 找不到 $SERL_ROOT/serl_launcher，请先把 serl 仓库同步过来，或改 SERL_ROOT"; exit 1; }

# ---- 1. 建 venv ----
$PYBIN -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# ---- 2. 装 GPU 版 jax（CUDA 12）----
# 先尝试与 mac 端一致的 0.4.35；H20=Hopper(sm_90) 若报 “Unable to use sm_90” 或性能异常，
# 改用下方“备选”里更新的版本（见文件末尾说明）。
pip install "jax[cuda12]==0.4.35"
# 如需更新版（H20 更稳）：
#   pip install -U "jax[cuda12]"        # 会同时升级 jaxlib，可能需同步升级 flax/orbax

# ---- 3. 装其余锁定依赖 ----
pip install -r "$(dirname "$0")/requirements-gpu.txt"

# ---- 4. editable 装本地 serl_launcher（+ franka_sim，若需要 franka 仿真）----
pip install -e "$SERL_ROOT/serl_launcher"
[ -d "$SERL_ROOT/franka_sim" ] && pip install -e "$SERL_ROOT/franka_sim" || true

# ---- 5. 红线自检：确认没被误装 tensorflow / 没降级 numpy ----
python - <<'PY'
import importlib.util, sys
bad = importlib.util.find_spec("tensorflow")
if bad is not None:
    print("!! 检测到 tensorflow 被安装，会摧毁 jax，请 pip uninstall tensorflow", file=sys.stderr)
    sys.exit(1)
import numpy, ml_dtypes
assert numpy.__version__.startswith("2."), f"numpy 被降级了: {numpy.__version__}"
print("OK numpy", numpy.__version__, "ml_dtypes", ml_dtypes.__version__)
PY

# ---- 6. 验证 jax 认到 GPU ----
echo "==> 验证 jax 设备："
python -c "import jax; print('jax', jax.__version__, 'devices =', jax.devices())"

cat <<EOF

============================================================
✅ SERL GPU venv 建好了： $VENV_DIR
   上面应打印 devices = [CudaDevice(id=0), ...]（不是 CpuDevice）

跑训练（命令与 mac 端完全一致，只把 venv 换成这个）：
  SVENV=$VENV_DIR
  SCRIPT=$SERL_ROOT/auto_research/scripts/train_serl_offline_maniskill.py
  \$SVENV/bin/python \$SCRIPT --data <某个 *_expert_serl.pkl> \\
    --max_updates 1200 --adaptive_tau --metrics_csv /tmp/one.csv

一键跑 16 组 + 画曲线：见 serl/auto_research/HOWTO_run_matrix16.md 第 2、4 节，
把其中的 SVENV 指向 $VENV_DIR 即可。

【H20 备选】若 jax 0.4.35 在 Hopper 上报 sm_90 相关警告/性能差：
  pip install -U "jax[cuda12]"
  pip install -U flax optax orbax-checkpoint    # 同步升级避免 API 不兼容
  然后重跑第 5、6 步自检。
============================================================
EOF
