#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./run_all_dock_retry.sh [-s SEED]
# 说明：
#   -s 传给 exps/dock/run.py 的 -s（默认 0）

SEED=0

usage() {
  cat <<'EOF'
Usage: ./run_all_dock_retry.sh [-s SEED]
  -s SEED   value passed to exps/dock/run.py -s (default: 0)
  -h        show help
EOF
}

while getopts ":s:h" opt; do
  case "${opt}" in
    s) SEED="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Unknown option: -${OPTARG}" >&2
      usage >&2
      exit 2
      ;;
    :)
      echo "Option -${OPTARG} requires an argument" >&2
      usage >&2
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

# 每条命令失败后等待多少秒再重试
RETRY_SLEEP=1

# 统一设置代理，继承现有 http_proxy
export https_proxy="${http_proxy:-}"
export HTTPS_PROXY="${http_proxy:-}"

commands=(
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python exps/dock/run.py -o parp1 -s ${SEED} -n 3000"
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python exps/dock/run.py -o fa7 -s ${SEED} -n 3000"
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 python exps/dock/run.py -o 5ht1b -s ${SEED} -n 3000"
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python exps/dock/run.py -o braf -s ${SEED} -n 3000"
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 python exps/dock/run.py -o jak2 -s ${SEED} -n 3000"
)

run_with_retry() {
  local cmd="$1"
  while true; do
    echo "Running: $cmd"
    if eval "$cmd"; then
      echo "Succeeded: $cmd"
      break
    fi
    echo "Failed at $(date): $cmd" >&2
    echo "Retrying in ${RETRY_SLEEP}s..." >&2
    sleep "${RETRY_SLEEP}"
  done
}

for cmd in "${commands[@]}"; do
  run_with_retry "$cmd" &
done

wait
