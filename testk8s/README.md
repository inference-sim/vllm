# vLLM Kubernetes Test Environment

This directory contains Kubernetes manifests for testing vLLM with observability features enabled:
- **Journey tracing**: Per-request OTEL spans tracking the full request lifecycle
- **Step tracing**: Scheduler step metrics exported as OTEL spans
- **KV cache events**: Block-level cache operations streamed via ZMQ to JSONL

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    vllm-server Pod                      │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │   vllm container    │  │ kv-events-subscriber     │  │
│  │   (GPU inference)   │  │ (ZMQ → JSONL)            │  │
│  │                     │  │                          │  │
│  │  :8000 HTTP API     │  │                          │  │
│  │  :5557 ZMQ PUB ─────┼──┼─► tcp://localhost:5557   │  │
│  │  :5558 ZMQ REQ ─────┼──┼─► tcp://localhost:5558   │  │
│  └─────────┬───────────┘  └────────────┬─────────────┘  │
│            │ OTLP                       │ JSONL         │
└────────────┼───────────────────────────┼────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────┐    ┌─────────────────────────┐
│   otel-collector Pod   │    │       data-pvc          │
│                        │    │                         │
│  :4317 gRPC            │    │  /data/traces.json      │
│  :4318 HTTP            │    │  /data/kv_events.jsonl  │
│           │            │    │                         │
│           ▼            │    └─────────────────────────┘
│  /data/traces.json ────┼──────────────►│
└────────────────────────┘
```

## Prerequisites

- Kubernetes cluster with GPU nodes (NVIDIA H100 80GB HBM3)
- `kubectl` configured to access the cluster
- vLLM image with KV events support (v0.6.0+ or built from this branch)
- Hugging Face token stored as a secret:
  ```bash
  kubectl create secret generic hf-token --from-literal=token=<your-hf-token>
  ```

> **Note:** The KV events subscriber requires the production-ready subscriber script
> (`examples/online_serving/kv_events_subscriber.py`) that supports environment variable
> configuration and JSONL file output. Images built from v0.5.0 or earlier have an older
> version that only prints to stdout.

## Deployment

### 1. Apply the manifest

```bash
kubectl apply -f testk8s/collectanddebug.yaml
```

This creates:
- `vllm-server` Deployment (vLLM + kv-events-subscriber sidecar)
- `otel-collector` Deployment (OTLP receiver → file exporter)
- `pvc-debug` Pod (for extracting output files)
- `vllm-cache-pvc` PersistentVolumeClaim (model cache)
- `data-pvc` PersistentVolumeClaim (traces and events output)
- Services for vLLM and OTEL collector

### 2. Wait for pods to be ready

```bash
kubectl wait --for=condition=available deployment/vllm-server --timeout=300s
kubectl wait --for=condition=available deployment/otel-collector --timeout=120s
kubectl wait --for=condition=ready pod/pvc-debug --timeout=60s
```

### 3. Verify vLLM is running

```bash
kubectl logs deployment/vllm-server -c vllm -f --tail=100
```

Look for output indicating the server is ready:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to exit the log stream.

To check the kv-events-subscriber sidecar:
```bash
kubectl logs deployment/vllm-server -c kv-events-subscriber --tail=20
```

### 4. Port forward to local machine

```bash
kubectl port-forward svc/vllm-server 8000:8000
```

Run this in a separate terminal or background it with `&`.

### 5. Send inference requests

**Quick test** (single short request to verify the server is working):
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "prompt": "The meaning of life is",
    "max_tokens": 20,
    "temperature": 0.7
  }' | jq .
```

**Trigger CPU KV offloading** (300 concurrent requests with long sequences):

With `--gpu-memory-utilization 0.1`, vLLM gets ~8GB of GPU memory (10% of H100 80GB).
After subtracting model weights (~250MB for opt-125m) and runtime overhead, the GPU
KV cache is roughly 6-7GB, holding approximately 170-200K tokens. The separate
`--kv-offloading-size 8.0` reserves 8GB of CPU memory for offloaded KV blocks.

To trigger offloading, exceed the GPU KV cache capacity with concurrent long sequences:

```bash
# Generate a ~900 token prompt
LONG_PROMPT=$(python3 -c "print('The ' * 450)")

# Send 300 concurrent requests to exceed GPU KV cache capacity
for i in {1..300}; do
  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "opt-125m",
      "prompt": "'"$LONG_PROMPT"'",
      "max_tokens": 100,
      "temperature": 0.7
    }' > /dev/null &
done
echo "Sent 300 requests, waiting for completion..."
wait
echo "All requests complete"
```

This sends ~300K tokens into KV cache (300 requests × ~1000 tokens), exceeding the
GPU capacity and forcing offloading to the CPU buffer.

**Verify offloading occurred** by checking the KV events:
```bash
kubectl cp pvc-debug:/mnt/data/kv_events.jsonl ./kv_events.jsonl
grep -c "CacheStoreCommitted\|CacheLoadCommitted\|TransferInitiated" kv_events.jsonl
```

If offloading triggered, you'll see non-zero counts for these event types.

### 6. Extract output files

Wait a few seconds for traces to flush, then extract:

```bash
kubectl cp pvc-debug:/mnt/data/traces.json ./traces.json
kubectl cp pvc-debug:/mnt/data/kv_events.jsonl ./kv_events.jsonl
```

## Output Files

### traces.json (OTEL spans)

Contains journey tracing and step tracing spans in OTLP JSON format.

```bash
# Count resource spans
jq '.resourceSpans | length' traces.json

# List span names
jq -r '.resourceSpans[].scopeSpans[].spans[].name' traces.json | sort | uniq -c

# View a specific span
jq '.resourceSpans[0].scopeSpans[0].spans[0]' traces.json
```

Expected span types:
- `llm_core`: Journey tracing spans (one per request)
- `scheduler_steps_N`: Step tracing spans (one per closure interval)

### kv_events.jsonl (KV cache events)

Contains KV cache block operations in JSONL format (one JSON object per line).

```bash
# Count events
wc -l kv_events.jsonl

# View first event batch
head -1 kv_events.jsonl | jq .

# Extract event types
cat kv_events.jsonl | jq -r '.[1][][0]' | sort | uniq -c
```

Event types:
- `BlockStored`: KV cache block added to GPU cache
- `BlockRemoved`: KV cache block evicted from GPU cache
- `AllBlocksCleared`: GPU cache cleared
- `CacheLoadCommitted`: Scheduler commits to loading blocks from CPU cache (offloading)
- `CacheStoreCommitted`: Scheduler commits to storing blocks to CPU cache (offloading)
- `CacheEviction`: Blocks evicted from CPU cache to make room (offloading)
- `TransferInitiated` / `TransferCompleted`: GPU↔CPU DMA transfers (offloading)
- `RemoteTransferInitiated` / `RemoteTransferCompleted`: Cross-machine transfers (disaggregated)

For detailed event schema, see [KV_EVENTS_SUBSCRIBER.md](../examples/online_serving/KV_EVENTS_SUBSCRIBER.md).

## Configuration Reference

### vLLM Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--model` | `facebook/opt-125m` | Model to serve |
| `--served-model-name` | `opt-125m` | API model name |
| `--enable-journey-tracing` | (flag) | Enable per-request OTEL spans |
| `--step-tracing-enabled` | (flag) | Enable scheduler step tracing |
| `--step-tracing-sample-rate` | `0.1` | 10% of batches sampled |
| `--step-tracing-rich-subsample-rate` | `0.1` | 10% of sampled batches get detailed metrics |
| `--step-tracing-closure-interval` | `10` | Close spans every 10 steps |
| `--otlp-traces-endpoint` | `http://otel-collector:4318/v1/traces` | OTLP HTTP endpoint |
| `--kv-events-config` | `{"enable_kv_cache_events": true, ...}` | KV events ZMQ publisher config |
| `--kv-offloading-size` | `8.0` | CPU memory (GiB) for KV cache offloading |

### KV Events Config

```json
{
  "enable_kv_cache_events": true,
  "publisher": "zmq",
  "endpoint": "tcp://*:5557",
  "replay_endpoint": "tcp://*:5558",
  "topic": "kv-events"
}
```

### Environment Variables

The subscriber sidecar uses:
- `VLLM_KV_EVENTS_SUB_ADDR`: ZMQ SUB socket (`tcp://localhost:5557`)
- `VLLM_KV_EVENTS_REPLAY_ADDR`: ZMQ REQ socket for replay (`tcp://localhost:5558`)
- `VLLM_KV_EVENTS_OUTPUT_FILE`: Output path (`/data/kv_events.jsonl`)

## Troubleshooting

### vLLM pod not starting

Check events and logs:
```bash
kubectl describe pod -l app=vllm-server-opt
kubectl logs deployment/vllm-server -c vllm --previous
```

Common issues:
- Missing `hf-token` secret
- Insufficient GPU memory
- Node selector not matching available nodes

### No KV events in output file

Check subscriber logs:
```bash
kubectl logs deployment/vllm-server -c kv-events-subscriber
```

Verify ZMQ connectivity:
```bash
kubectl exec deployment/vllm-server -c kv-events-subscriber -- \
  python3 -c "import zmq; print(zmq.zmq_version())"
```

### Empty traces.json

OTEL collector batches spans before writing. Wait 15-30 seconds after sending requests, then extract again.

Check collector logs:
```bash
kubectl logs deployment/otel-collector
```

## Cleanup

```bash
kubectl delete -f testk8s/collectanddebug.yaml
```

To also delete the PVCs (removes cached models and data):
```bash
kubectl delete pvc vllm-cache-pvc data-pvc
```
