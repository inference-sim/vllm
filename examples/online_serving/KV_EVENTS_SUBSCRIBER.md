# KV Cache Events Subscriber

A production-ready subscriber for vLLM KV cache events. This subscriber connects to the ZMQ event publisher and writes events in JSONL format for downstream processing such as simulator training and analytics.

## Quick Start

```bash
# Output events to stdout
python examples/online_serving/kv_events_subscriber.py

# Output events to a file
VLLM_KV_EVENTS_OUTPUT_FILE=/var/log/kv_events.jsonl python examples/online_serving/kv_events_subscriber.py

# Connect to a remote vLLM server
VLLM_KV_EVENTS_SUB_ADDR=tcp://vllm-server:5557 \
VLLM_KV_EVENTS_REPLAY_ADDR=tcp://vllm-server:5558 \
python examples/online_serving/kv_events_subscriber.py
```

## Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_KV_EVENTS_SUB_ADDR` | `tcp://localhost:5557` | ZMQ SUB socket address to receive events |
| `VLLM_KV_EVENTS_REPLAY_ADDR` | `tcp://localhost:5558` | ZMQ REQ socket address for replay requests |
| `VLLM_KV_EVENTS_OUTPUT_FILE` | None (stdout) | Output file path for JSONL events |
| `VLLM_KV_EVENTS_TOPIC` | `""` (all topics) | Topic filter for ZMQ subscription |

## Enabling KV Events in vLLM

To enable KV cache event publishing in vLLM, set the following configuration:

```python
from vllm import LLM

llm = LLM(
    model="your-model",
    kv_events_config={
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": "tcp://*:5557",
        "replay_endpoint": "tcp://*:5558",
    }
)
```

Or via command line:
```bash
vllm serve your-model \
    --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq"}'
```

## JSONL Output Format

Each line in the output is a JSON-encoded `KVEventBatch`. The format uses msgspec's `array_like` encoding for efficiency:

```json
[1706886400.123, [event1, event2, ...], null, 42]
```

Fields (in order):
1. `ts` (float): Unix timestamp when the batch was created
2. `events` (array): List of events (see Event Types below)
3. `data_parallel_rank` (int|null): DP rank that emitted the batch
4. `scheduler_step` (int|null): Scheduler step for correlation

### Event Types

Each event is encoded as a tagged array. The first element is the event type name.

#### Cache State Events

**BlockStored**: New block added to cache
```json
["BlockStored", [[block_hash1, block_hash2]], parent_hash, [token_ids], block_size, lora_id, "GPU", lora_name]
```

**BlockRemoved**: Block removed from cache
```json
["BlockRemoved", [[block_hash1]], "GPU"]
```

**AllBlocksCleared**: All blocks cleared
```json
["AllBlocksCleared"]
```

#### Local Transfer Events

**TransferInitiated**: DMA transfer submitted
```json
["TransferInitiated", transfer_id, "request-id", "GPU", "CPU", block_count, scheduler_step]
```

**TransferCompleted**: DMA transfer finished
```json
["TransferCompleted", transfer_id, "request-id", "GPU", "CPU", block_count, true, scheduler_step]
```

#### Remote Transfer Events

**RemoteTransferInitiated**: Cross-machine transfer started
```json
["RemoteTransferInitiated", transfer_id, "request-id", "NIXL", source_rank, dest_rank, block_count]
```

**RemoteTransferCompleted**: Cross-machine transfer finished
```json
["RemoteTransferCompleted", transfer_id, "request-id", "NIXL", source_rank, dest_rank, block_count, true]
```

#### Cache Operation Events

**CacheLoadCommitted**: Scheduler committed to loading from cache
```json
["CacheLoadCommitted", "request-id", "CPU", block_count, scheduler_step]
```

**CacheStoreCommitted**: Scheduler committed to storing to cache
```json
["CacheStoreCommitted", "request-id", "DISK", block_count, scheduler_step]
```

**CacheEviction**: Blocks evicted from cache
```json
["CacheEviction", "CPU", blocks_evicted, "lru", scheduler_step, null]
```

### Storage Mediums

| Value | Description |
|-------|-------------|
| `"GPU"` | GPU HBM (fastest) |
| `"CPU"` | System RAM |
| `"DISK"` | NVMe/SSD |
| `"LMCACHE"` | External LMCache |

### Connector Types (Remote Transfers)

| Value | Description |
|-------|-------------|
| `"NIXL"` | NIXL connector (async RDMA reads) |
| `"P2P"` | P2P NCCL connector (sync send/recv) |
| `"MOONCAKE"` | Mooncake connector |

### Eviction Reasons

| Value | Description |
|-------|-------------|
| `"lru"` | Least recently used eviction |
| `"capacity"` | Capacity limit reached |
| `"preemption"` | Request preemption |

## Graceful Shutdown

The subscriber handles `SIGTERM` and `SIGINT` signals for graceful shutdown:
- Stops polling for new messages
- Flushes and closes the output file
- Exits with code 0

## Kubernetes Deployment

Example pod specification:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kv-events-subscriber
spec:
  containers:
  - name: subscriber
    image: your-registry/vllm-subscriber:latest
    command:
      - python
      - examples/online_serving/kv_events_subscriber.py
    env:
      - name: VLLM_KV_EVENTS_SUB_ADDR
        value: "tcp://vllm-service:5557"
      - name: VLLM_KV_EVENTS_REPLAY_ADDR
        value: "tcp://vllm-service:5558"
      - name: VLLM_KV_EVENTS_OUTPUT_FILE
        value: "/data/kv_events.jsonl"
    volumeMounts:
      - name: events-data
        mountPath: /data
    # Graceful shutdown configuration
    terminationGracePeriodSeconds: 30
  volumes:
    - name: events-data
      persistentVolumeClaim:
        claimName: kv-events-pvc
```

For a sidecar deployment alongside vLLM:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-with-subscriber
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    # ... vLLM configuration ...

  - name: subscriber
    image: vllm/vllm-openai:latest
    command:
      - python
      - examples/online_serving/kv_events_subscriber.py
    env:
      - name: VLLM_KV_EVENTS_SUB_ADDR
        value: "tcp://localhost:5557"
      - name: VLLM_KV_EVENTS_OUTPUT_FILE
        value: "/data/kv_events.jsonl"
```

## Correlation Keys

Use these fields to correlate events:

| Key | Use Case |
|-----|----------|
| `scheduler_step` | Correlate with journey events and step metrics |
| `transfer_id` | Pair TransferInitiated with TransferCompleted |
| `request_id` | Track per-request KV cache behavior |

### Transfer ID Encoding

Transfer IDs are globally unique across ranks:
```
transfer_id = (emitter_rank << 32) | local_counter
```

To extract rank and counter:
```python
from vllm.distributed.kv_events import TransferIdGenerator

transfer_id = 21474836481  # Example
rank = TransferIdGenerator.extract_rank(transfer_id)      # 5
counter = TransferIdGenerator.extract_counter(transfer_id) # 1
```

## Calculating Transfer Latency

Transfer latency can be calculated from event batch timestamps:

```python
import json

# Read JSONL file
initiated_times = {}  # transfer_id -> ts
completed_times = {}  # transfer_id -> ts

with open("kv_events.jsonl") as f:
    for line in f:
        batch = json.loads(line)
        ts = batch[0]  # First element is timestamp
        events = batch[1]  # Second element is events list

        for event in events:
            event_type = event[0]
            if event_type == "TransferInitiated":
                transfer_id = event[1]
                initiated_times[transfer_id] = ts
            elif event_type == "TransferCompleted":
                transfer_id = event[1]
                completed_times[transfer_id] = ts

# Calculate latencies
for transfer_id, completed_ts in completed_times.items():
    if transfer_id in initiated_times:
        latency = completed_ts - initiated_times[transfer_id]
        print(f"Transfer {transfer_id}: {latency*1000:.2f}ms")
```

## Error Handling

The subscriber is designed to be resilient:
- Decode errors are logged but don't crash the subscriber
- ZMQ connection failures are handled with automatic retry
- File write errors are logged and the subscriber continues
- Missing messages are detected and replayed from the buffer

## Logging

Logs are written to stderr so stdout remains available for event output (when no file is specified). Log levels:

| Level | Events |
|-------|--------|
| INFO | Startup, shutdown, replay completion |
| WARNING | Missed messages |
| ERROR | Decode errors, ZMQ errors, file errors |
