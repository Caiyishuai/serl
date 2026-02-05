#!/bin/bash
#
# Start learner and actor in a tmux session (2 windows).
# Usage: ./run_maniskill_tmux.sh [optional args for both scripts]
# Example: ./run_maniskill_tmux.sh --env=PickCube-v1 --seed=42
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="maniskill_drq"

# Kill existing session if present (suppress "no server running" when tmux not running)
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" 2>/dev/null

# Window 0: Learner (start first so server is ready)
# Use '; exec bash' so window stays open on error and session is not destroyed
tmux new-session -d -s "$SESSION" -n learner "cd '$SCRIPT_DIR' && bash run_learner_maniskill.sh $*; exec bash"

# Window 1: Actor (same: keep window open after exit)
tmux new-window -t "$SESSION" -n actor "cd '$SCRIPT_DIR' && bash run_actor_maniskill.sh $*; exec bash"

# Focus learner window and attach
tmux select-window -t "$SESSION:learner"
exec tmux attach -t "$SESSION"

# tmux attach -t maniskill_drq
# Ctrl+b + 0 / 1 切到 learner 或 actor
# 操作	按键
# 切到 learner	Ctrl+b 然后 0
# 切到 actor	Ctrl+b 然后 1
# 下一个窗口	Ctrl+b 然后 n
# 上一个窗口	Ctrl+b 然后 p
# 列出所有窗口	Ctrl+b 然后 w
# 脱开 session	Ctrl+b 然后 d
# 重新连上	tmux attach -t maniskill_drq
