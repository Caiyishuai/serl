"""
诊断脚本：检查 demo buffer 和 online buffer 的数据结构与分布。

用法（不需要重启训练）:
  # 只检查 demo pkl 文件
  python inspect_buffers.py --demo_pkl path/to/demo.pkl

  # 检查 demo pkl + 模拟 _adapt_demo_for_chunking 之后的 shape
  python inspect_buffers.py --demo_pkl path/to/demo.pkl --show_adapted
"""

import argparse
import pickle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 复制自 async_drq_sim_maniskill_ty.py，用于本地模拟
# ─────────────────────────────────────────────────────────────────────────────
def _adapt_demo_for_chunking(transition):
    adapted = {}
    for k, v in transition.items():
        if k in ("observations", "next_observations"):
            adapted[k] = {
                ok: np.expand_dims(np.asarray(ov), axis=0)
                for ok, ov in v.items()
            }
        else:
            adapted[k] = v
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────
SEP = "─" * 72


def print_obs_structure(label, obs, indent=4):
    pad = " " * indent
    print(f"{pad}observations ({label}):")
    for key, val in obs.items():
        arr = np.asarray(val)
        if arr.dtype == np.uint8:
            extra = f"  uint8 [{arr.min()}, {arr.max()}]"
        else:
            extra = f"  range [{arr.min():.4f}, {arr.max():.4f}]  mean {arr.mean():.4f}"
        print(f"{pad}  {key:20s} shape={arr.shape}  dtype={arr.dtype}{extra}")


def print_scalar_stats(label, values, fmt=".4f"):
    arr = np.asarray(values, dtype=np.float32)
    print(f"  {label:20s}  n={len(arr)}  "
          f"min={arr.min():{fmt}}  max={arr.max():{fmt}}  "
          f"mean={arr.mean():{fmt}}  std={arr.std():{fmt}}")


def print_value_counts(label, values, top=10):
    arr = np.asarray(values)
    unique, counts = np.unique(arr, return_counts=True)
    print(f"  {label} unique values ({len(unique)} total):")
    order = np.argsort(-counts)
    for i in order[:top]:
        print(f"    {unique[i]!r:10}  count={counts[i]}")


# ─────────────────────────────────────────────────────────────────────────────
# 主检查逻辑
# ─────────────────────────────────────────────────────────────────────────────
def inspect_demo_pkl(path: str, show_adapted: bool, n_sample: int):
    print(f"\n{SEP}")
    print(f"Demo PKL 文件: {path}")
    print(SEP)

    with open(path, "rb") as f:
        trajs = pickle.load(f)

    n = len(trajs)
    print(f"总 transitions 数: {n}")

    # ── 1. 第一条 transition 的原始 key / shape ───────────────────────────────
    print(f"\n{'─'*40}")
    print("[ 原始格式（存储在 pkl 中）] — 第 0 条 transition")
    print(f"{'─'*40}")
    t0 = trajs[0]
    print(f"  顶层 keys: {list(t0.keys())}")
    print_obs_structure("observations", t0["observations"])
    print_obs_structure("next_observations", t0["next_observations"])

    act = np.asarray(t0["actions"])
    print(f"    actions              shape={act.shape}  dtype={act.dtype}")
    print(f"    rewards              {t0['rewards']!r}")
    print(f"    masks                {t0['masks']!r}")
    print(f"    dones                {t0['dones']!r}")

    # ── 2. adapt 后的 shape（模拟 insert 进 demo_buffer 时的变换）────────────
    if show_adapted:
        print(f"\n{'─'*40}")
        print("[ 经 _adapt_demo_for_chunking 后 ] — 第 0 条 transition")
        print(f"{'─'*40}")
        ta = _adapt_demo_for_chunking(t0)
        print_obs_structure("observations", ta["observations"])
        print_obs_structure("next_observations", ta["next_observations"])

    # ── 3. 全量统计 ───────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("[ 全量数据分布 ]")
    print(f"{'─'*40}")

    rewards  = [t["rewards"] for t in trajs]
    masks    = [t["masks"]   for t in trajs]
    dones    = [t["dones"]   for t in trajs]
    actions  = np.stack([np.asarray(t["actions"]).ravel() for t in trajs])

    print_scalar_stats("rewards", rewards)
    print_value_counts("masks", masks)
    print_value_counts("dones", dones)

    print(f"\n  actions  shape per transition: {np.asarray(trajs[0]['actions']).shape}")
    print(f"  actions  global range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  actions  per-dim mean / std:")
    for i, (m, s) in enumerate(zip(actions.mean(0), actions.std(0))):
        print(f"    dim {i:2d}:  mean={m:+.4f}  std={s:.4f}")

    # ── 4. 抽样几条检查 image 数值 ───────────────────────────────────────────
    if n_sample > 0:
        print(f"\n{'─'*40}")
        print(f"[ 随机抽样 {min(n_sample, n)} 条 — image 像素统计 ]")
        print(f"{'─'*40}")
        idxs = np.random.choice(n, size=min(n_sample, n), replace=False)
        image_keys = [k for k in trajs[0]["observations"] if k != "state"]
        for cam in image_keys:
            pixvals = []
            for i in idxs:
                arr = np.asarray(trajs[i]["observations"][cam])
                pixvals.append(arr.ravel())
            pv = np.concatenate(pixvals).astype(np.float32)
            print(f"  {cam}  pixel: min={pv.min():.1f}  max={pv.max():.1f}  "
                  f"mean={pv.mean():.1f}  std={pv.std():.1f}")

    print(f"\n{SEP}")
    print("提示：online buffer 数据通过 agentlace 由 actor 写入，")
    print("      在 learner 侧用 next(replay_iterator) 取到。")
    print("      如需检查 online batch，可在 learner 循环里临时加 print_batch_debug(batch)。")
    print(SEP)


def print_batch_debug_snippet():
    """打印可直接粘贴进 learner 循环的调试代码片段。"""
    print(f"\n{SEP}")
    print("[ 在 learner 循环里检查 online/demo batch 的调试代码片段 ]")
    print(f"{SEP}")
    snippet = '''
# ── 粘贴到 learner() 函数中，batch = next(replay_iterator) 之后 ──
if step == 0:
    import jax, numpy as np
    def _show_batch(name, b):
        print(f"\\n=== {name} batch keys: {list(b.keys())} ===")
        obs = b.get("observations", b)
        for k, v in obs.items():
            arr = np.asarray(v)
            print(f"  obs[{k!r}]: shape={arr.shape} dtype={arr.dtype} "
                  f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        for field in ("actions", "rewards", "masks", "dones"):
            if field in b:
                arr = np.asarray(b[field])
                print(f"  {field}: shape={arr.shape} range=[{arr.min():.3f}, {arr.max():.3f}]")
    _show_batch("online", batch)
    if demo_iterator is not None:
        _show_batch("demo", demo_batch)
# ── 结束 ──
'''
    print(snippet)
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 demo / online buffer 数据分布")
    parser.add_argument("--demo_pkl",     type=str, default=None,
                        help="demo pkl 文件路径")
    parser.add_argument("--show_adapted", action="store_true", default=True,
                        help="显示经 _adapt_demo_for_chunking 后的 shape（默认 True）")
    parser.add_argument("--n_sample",     type=int, default=20,
                        help="随机抽样条数（用于 image 像素统计，0=跳过）")
    parser.add_argument("--snippet",      action="store_true",
                        help="打印粘贴进 learner 的调试代码片段")
    args = parser.parse_args()

    if args.demo_pkl:
        inspect_demo_pkl(args.demo_pkl, args.show_adapted, args.n_sample)

    if args.snippet or not args.demo_pkl:
        print_batch_debug_snippet()
