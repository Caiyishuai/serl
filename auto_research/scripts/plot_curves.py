#!/usr/bin/env python
"""读取 16 组训练曲线 CSV, 画成按任务分组的训练曲线图。
- 图1: critic_loss 随 update step 变化 (4 任务子图, 每图 4 条: dense/sparse x fixed/adaptive tau)
- 图2: adaptive tau 的 tau 动态调整曲线 (4 任务子图)
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVED = "/Users/yishuaicai/mywork/serl/auto_research/logs/curves"
OUT = "/Users/yishuaicai/mywork/serl/auto_research/logs"

TASKS = ["pushcube", "pickcube", "stackcube", "pullcubetool"]
TASK_TITLE = {
    "pushcube": "PushCube", "pickcube": "PickCube",
    "stackcube": "StackCube", "pullcubetool": "PullCubeTool",
}
# (rmode, tmode) -> (label, color, linestyle)
SERIES = {
    ("dense", "fixed"):     ("dense | fixed tau",    "#4C9BE8", "-"),
    ("dense", "adaptive"):  ("dense | adaptive tau", "#E8823C", "-"),
    ("sparse", "fixed"):    ("sparse | fixed tau",   "#5CC98A", "--"),
    ("sparse", "adaptive"): ("sparse | adaptive tau","#D95F6B", "--"),
}


def load_csv(path):
    steps, crit, actor, tau = [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            crit.append(float(row["critic_loss"]))
            actor.append(float(row["actor_loss"]))
            tau.append(float(row["tau"]))
    return steps, crit, actor, tau


def plot_metric(metric_idx, title, ylabel, fname, logy=False, only_adaptive=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axes = axes.ravel()
    for i, task in enumerate(TASKS):
        ax = axes[i]
        for (rmode, tmode), (label, color, ls) in SERIES.items():
            if only_adaptive and tmode != "adaptive":
                continue
            p = os.path.join(CURVED, f"curve_{task}_{rmode}_{tmode}.csv")
            if not os.path.exists(p):
                continue
            data = load_csv(p)
            steps = data[0]
            y = data[metric_idx]
            ax.plot(steps, y, label=label, color=color, linestyle=ls, linewidth=1.6)
        ax.set_title(TASK_TITLE[task], fontsize=12, fontweight="bold")
        ax.set_xlabel("update step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)
    plt.close()


if __name__ == "__main__":
    # 图1: critic_loss (对数轴, 因量级差异大)
    plot_metric(1, "SERL Offline Training - Critic Loss Curves (4 tasks x dense/sparse x fixed/adaptive tau)",
                "critic_loss (log)", "curves_critic_loss.png", logy=True)
    # 图2: actor_loss
    plot_metric(2, "SERL Offline Training - Actor Loss Curves",
                "actor_loss", "curves_actor_loss.png", logy=False)
    # 图3: tau 动态 (仅 adaptive 组)
    plot_metric(3, "Adaptive tau Dynamics (adaptive-tau groups only, Rsync mechanism)",
                "tau", "curves_tau.png", logy=False, only_adaptive=True)
    print("ALL PLOTS DONE")
