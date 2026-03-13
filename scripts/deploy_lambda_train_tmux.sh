#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_lambda_train_tmux.sh --host ubuntu@HOST [options]

Copies a prepared data directory to a remote machine, updates or clones the
Counterpoignant repo there, and starts the staged 4k->8k->DroPE->8k-Bach
training run inside a detached tmux session. On the remote host, the script
verifies that PyTorch can see CUDA before starting training.

Options:
  --host HOST                 Remote SSH target (required), e.g. ubuntu@1.2.3.4
  --local-data-dir PATH       Local prepared data dir
                              (default: /Users/tannerfokkens/Documents/2pt-bach_update/datamidiall_sd_refresh)
  --remote-repo-dir PATH      Remote repo dir (default: Counterpoignant)
  --repo-url URL              Repo to clone/update
                              (default: https://github.com/tannerfokkens-maker/Counterpoignant)
  --branch NAME               Git branch to deploy (default: main)
  --session NAME              tmux session name
                              (default: counterpoignant-train-YYYYmmdd-HHMMSS)
  --remote-data-name NAME     Remote data subdir name under the repo root
                              (default: basename of local data dir)
  --remote-log-dir PATH       Remote log dir (default: <remote-repo-dir>/logs)
  --dry-run                   Print actions without executing them
  -h, --help                  Show this help

Example:
  scripts/deploy_lambda_train_tmux.sh \
    --host ubuntu@192.222.50.215 \
    --local-data-dir /Users/tannerfokkens/Documents/2pt-bach_update/datamidiall_sd_refresh
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATA_DIR="$ROOT_DIR/datamidiall_sd_refresh"
HOST=""
LOCAL_DATA_DIR="$DEFAULT_DATA_DIR"
REMOTE_REPO_DIR="Counterpoignant"
REPO_URL="https://github.com/tannerfokkens-maker/Counterpoignant"
BRANCH="main"
SESSION="counterpoignant-train-$(date +%Y%m%d-%H%M%S)"
REMOTE_DATA_NAME=""
REMOTE_LOG_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --local-data-dir)
      LOCAL_DATA_DIR="${2:-}"
      shift 2
      ;;
    --remote-repo-dir)
      REMOTE_REPO_DIR="${2:-}"
      shift 2
      ;;
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --session)
      SESSION="${2:-}"
      shift 2
      ;;
    --remote-data-name)
      REMOTE_DATA_NAME="${2:-}"
      shift 2
      ;;
    --remote-log-dir)
      REMOTE_LOG_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "--host is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$LOCAL_DATA_DIR" ]]; then
  echo "Local data dir not found: $LOCAL_DATA_DIR" >&2
  exit 1
fi

if [[ -z "$REMOTE_DATA_NAME" ]]; then
  REMOTE_DATA_NAME="$(basename "$LOCAL_DATA_DIR")"
fi

if [[ -z "$REMOTE_LOG_DIR" ]]; then
  REMOTE_LOG_DIR="$REMOTE_REPO_DIR/logs"
fi

REMOTE_LOG_FILE="$REMOTE_LOG_DIR/${SESSION}.log"
REMOTE_RUNNER="$REMOTE_REPO_DIR/.launch_${SESSION}.sh"
REMOTE_DATA_DIR="$REMOTE_REPO_DIR/$REMOTE_DATA_NAME"
LOCAL_LOCK_FILE="$ROOT_DIR/uv.lock"

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo "Host:            $HOST"
echo "Local data dir:  $LOCAL_DATA_DIR"
echo "Remote repo dir: $REMOTE_REPO_DIR"
echo "Remote data dir: $REMOTE_DATA_DIR"
echo "tmux session:    $SESSION"
echo

run ssh "$HOST" bash -s -- "$REMOTE_REPO_DIR" "$REPO_URL" "$BRANCH" <<'EOF'
set -euo pipefail
repo_dir="$1"
repo_url="$2"
branch="$3"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required on the remote host" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required on the remote host" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
  else
    echo "uv is missing and neither curl nor wget is available to install it" >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation failed on the remote host" >&2
  exit 1
fi

mkdir -p "$(dirname "$repo_dir")"

if [[ -d "$repo_dir/.git" ]]; then
  git -C "$repo_dir" fetch origin
  if git -C "$repo_dir" rev-parse --verify "$branch" >/dev/null 2>&1; then
    git -C "$repo_dir" switch "$branch"
  else
    git -C "$repo_dir" switch -c "$branch" "origin/$branch"
  fi
  git -C "$repo_dir" pull --ff-only origin "$branch"
else
  git clone --branch "$branch" "$repo_url" "$repo_dir"
fi
EOF

run ssh "$HOST" mkdir -p "$REMOTE_DATA_DIR"
run rsync -az --delete --progress "$LOCAL_DATA_DIR"/ "$HOST:$REMOTE_DATA_DIR/"
if [[ -f "$LOCAL_LOCK_FILE" ]]; then
  run rsync -az "$LOCAL_LOCK_FILE" "$HOST:$REMOTE_REPO_DIR/uv.lock"
fi

run ssh "$HOST" bash -s -- \
  "$REMOTE_REPO_DIR" \
  "$REMOTE_DATA_NAME" \
  "$SESSION" \
  "$REMOTE_RUNNER" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_LOG_FILE" <<'EOF'
set -euo pipefail
repo_dir="$1"
data_name="$2"
session="$3"
runner="$4"
log_dir="$5"
log_file="$6"
repo_abs="$HOME/$repo_dir"
data_dir="$HOME/$repo_dir/$data_name"
log_dir="$HOME/$log_dir"
log_file="$HOME/$log_file"

mkdir -p "$log_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

cat > "$runner" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting training"
cd "$repo_abs"
export UV_CACHE_DIR="\$HOME/.cache/uv"
export PATH="\$HOME/.local/bin:\$PATH"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] uv=\$(uv --version)"
if [[ -f uv.lock ]]; then
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Found uv.lock; using frozen sync"
  uv sync --frozen
else
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] No uv.lock found; using non-frozen sync"
  uv sync
fi
TORCH_VERSION="\$(.venv/bin/python - <<'PY'
import importlib.metadata as md
print(md.version("torch"))
PY
)"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Reinstalling torch \$TORCH_VERSION with uv automatic backend selection"
UV_TORCH_BACKEND=auto uv pip install --python .venv/bin/python --reinstall "torch==\$TORCH_VERSION"
.venv/bin/python - <<'PY'
import platform
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"machine={platform.machine()}")
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch; aborting before training.")

print(f"cuda_device_count={torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
PY
resume_args=()
models_dir="$repo_abs/models"
target_embed_dim=384
target_num_heads=8
target_num_layers=9
target_num_recurrent_steps=1
mkdir -p "\$models_dir"
if [[ -f "\$models_dir/pretrain_latest.pt" ]]; then
  if .venv/bin/python - "\$models_dir/pretrain_latest.pt" \
      "\$target_embed_dim" "\$target_num_heads" "\$target_num_layers" "\$target_num_recurrent_steps" <<'PY'
import sys
import torch

path = sys.argv[1]
target_embed_dim = int(sys.argv[2])
target_num_heads = int(sys.argv[3])
target_num_layers = int(sys.argv[4])
target_num_recurrent_steps = int(sys.argv[5])

ckpt = torch.load(path, map_location="cpu", weights_only=False)
cfg = ckpt.get("config")
if isinstance(cfg, dict):
    embed_dim = int(cfg.get("embed_dim", -1))
    num_heads = int(cfg.get("num_heads", -1))
    num_layers = int(cfg.get("num_layers", -1))
    num_recurrent_steps = int(cfg.get("num_recurrent_steps", 1))
else:
    embed_dim = int(getattr(cfg, "embed_dim", -1))
    num_heads = int(getattr(cfg, "num_heads", -1))
    num_layers = int(getattr(cfg, "num_layers", -1))
    num_recurrent_steps = int(getattr(cfg, "num_recurrent_steps", 1))

if (
    embed_dim == target_embed_dim
    and num_heads == target_num_heads
    and num_layers == target_num_layers
    and num_recurrent_steps == target_num_recurrent_steps
):
    raise SystemExit(0)

print(
    "checkpoint architecture mismatch:",
    f"embed_dim={embed_dim}",
    f"num_heads={num_heads}",
    f"num_layers={num_layers}",
    f"num_recurrent_steps={num_recurrent_steps}",
)
raise SystemExit(1)
PY
  then
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Resuming from \$models_dir/pretrain_latest.pt"
    resume_args+=(--resume "\$models_dir/pretrain_latest.pt")
  else
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Existing checkpoint in \$models_dir is incompatible with target 384/8/9 model; starting fresh"
  fi
fi
.venv/bin/bach-gen train \
  --curriculum \
  --data-dir "$data_dir" \
  --models-dir "\$models_dir" \
  --finetune bach \
  --seq-len-stages "4096:40,8192:25" \
  --batch-size 16 \
  --accumulation-steps 1 \
  --lr 4e-4 \
  --finetune-lr 2e-4 \
  --log-interval 1 \
  --val-interval 5 \
  --embed-dim 384 \
  --num-heads 8 \
  --num-layers 9 \
  --num-recurrent-steps 1 \
  --no-looplm-exit-gate \
  --no-loop-step-embedding \
  --pos-encoding pope \
  --piece-balance sqrt \
  "\${resume_args[@]}" \
  --bf16
SCRIPT

chmod +x "$runner"
tmux new-session -d -s "$session" "$runner"

echo "session=$session"
echo "log_file=$log_file"
echo "repo_dir=$repo_abs"
echo "data_dir=$data_dir"
EOF

cat <<EOF

Remote training started.

Attach:
  ssh $HOST "tmux attach -t $SESSION"

Tail log:
  ssh $HOST "tail -f $REMOTE_LOG_FILE"

Repo:
  ssh $HOST "cd $REMOTE_REPO_DIR && git status --short"
EOF
