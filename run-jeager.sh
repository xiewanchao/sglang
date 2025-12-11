#!/usr/bin/env bash
set -e

#-------------------------------
# 端口检查函数
#-------------------------------
check_and_kill_port() {
    local port=$1
    local pid

    pid=$(lsof -t -i:"$port" || true)
    if [[ -n "$pid" ]]; then
        echo "Port $port is in use by PID $pid. Killing..."
        kill -9 "$pid" || true
        echo "Port $port cleared."
    else
        echo "Port $port is free."
    fi
}

#-------------------------------
# 启动前的环境变量
#-------------------------------
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/sgl-workspace/hicachefile

MODEL_PATH=/home/models/QwQ-32B
PORT=30000
TP=2

export CUDA_VISIBLE_DEVICES=0,1

#-------------------------------
# 启动前检查端口
#-------------------------------
echo "Checking required ports..."
check_and_kill_port 4317   # otlp grpc
check_and_kill_port 4327   # jaeger grpc
check_and_kill_port 4328   # jaeger http
check_and_kill_port $PORT  # sglang server

#-------------------------------
# 捕获退出信号，统一杀死所有子进程
#-------------------------------
pids=()

cleanup() {
    echo "Received stop signal. Killing all subprocesses..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    exit 1
}

trap cleanup SIGINT SIGTERM EXIT

#-------------------------------
# 启动第 1 个：Jaeger
#-------------------------------
jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=0.0.0.0:4327 \
  --collector.otlp.http.host-port=0.0.0.0:4328 &
pids+=($!)
echo "Started Jaeger, PID=${pids[-1]}"

#-------------------------------
# 启动第 2 个：otelcol-contrib
#-------------------------------
otelcol-contrib --config otel-collector.yaml &
pids+=($!)
echo "Started otelcol-contrib, PID=${pids[-1]}"

#-------------------------------
# 启动第 3 个：sglang server
#-------------------------------
python python/sglang/launch_server.py \
    --model-path $MODEL_PATH \
    --page-size 128 \
    --tp $TP \
    --port $PORT \
    --enable-hierarchical-cache \
    --hicache-write-policy write_through \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy wait_complete \
    --enable-trace \
    --otlp-traces-endpoint 127.0.0.1:4317 &
pids+=($!)
echo "Started sglang server, PID=${pids[-1]}"

#-------------------------------
# 等待任意子进程退出
#-------------------------------
echo "All processes started. Waiting for any to exit..."
wait -n

echo "One process exited. Killing all..."
cleanup
