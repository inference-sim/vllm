# Request Journey Tracing in vLLM

**Real-time request observability using OpenTelemetry distributed tracing**

---

## What is Request Journey Tracing?

Request journey tracing gives you complete visibility into how requests flow through vLLM, from the moment they arrive at your API server to when responses are sent back to clients.

It uses **OpenTelemetry (OTEL)** to create distributed traces with two linked spans:

- **API Span** (`llm_request`) - Tracks the request through your API server
- **Core Span** (`llm_core`) - Tracks processing in the inference engine

*Note: When using the OpenAI-compatible API server (`vllm serve`), both spans are created. Direct AsyncLLM usage can produce core-only traces, but only for requests where you manually pass trace_headers={"x-vllm-journey-sampled": "1"}. Otherwise no spans are created.*

Events are emitted **in real-time** as requests progress through different states, giving you detailed timing and progress information at every step.

**For production workloads:** Journey tracing supports **probabilistic sampling** to reduce overhead at high request rates. You can trace a subset of requests (e.g., 10% or 1%) while maintaining negligible overhead for untraced requests. See [Sampling for Production](#sampling-for-production) for details.

**Note:** Sampling currently works with the OpenAI-compatible API server (`vllm serve`). Direct usage of `AsyncLLM` requires manual trace header management.

---

## Why Use This?

### 🔍 Debug Performance Issues
- See exactly where time is spent: queuing, prefill, decode
- Identify bottlenecks in your serving pipeline
- Understand why some requests are slow

### 📊 Monitor Production
- Track time-to-first-token (TTFT) for every request
- Detect preemption patterns and resource contention
- Measure end-to-end latency from API arrival to departure

### 🎯 Optimize Resource Usage
- Understand how your workload behaves
- See which requests get preempted and why
- Correlate performance with load patterns

---

## Quick Start

### Step 1: Start an OTEL Collector

The easiest way to view traces is using Jaeger with OTEL:

```bash
# Start Jaeger all-in-one (includes OTEL collector)
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/jaeger:latest

# Open Jaeger UI in your browser
open http://localhost:16686
```

### Step 2: Start vLLM with Journey Tracing

```bash
# Enable journey tracing and point to OTEL collector
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

That's it! Journey tracing is now enabled.

### Step 3: Send Some Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Step 4: View Traces in Jaeger

1. Open http://localhost:16686
2. In the service dropdown, select your vLLM service (default: "vllm" or value of `OTEL_SERVICE_NAME`)
3. Click "Find Traces"
4. Click on any trace to see the complete request journey

You'll see a timeline with two spans:
- **llm_request** (API layer) - parent span, created by scope `vllm.api`
- **llm_core** (Engine layer) - child span, created by scope `vllm.scheduler`

Each span contains events showing the request lifecycle.

**Note:** The service dropdown shows `service.name` (configurable via `OTEL_SERVICE_NAME` environment variable). The tracer scopes `vllm.api` and `vllm.scheduler` appear as attributes within spans (`scope.name`), not as selectable services in the UI.

---

## What You'll See in Traces

### Two-Layer Span Architecture

When using the OpenAI API server (`vllm serve`), each traced request creates two linked spans that show the complete journey:

```
┌─────────────────────────────────────────────────────────────┐
│ llm_request (API Layer - Parent Span)                       │
│                                                              │
│  ARRIVED → HANDOFF_TO_CORE → FIRST_RESPONSE → DEPARTED     │
│               │                                              │
│               └──┐                                           │
│                  │                                           │
│  ┌───────────────▼──────────────────────────────────────┐  │
│  │ llm_core (Engine Layer - Child Span)                 │  │
│  │                                                       │  │
│  │  QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### API Layer Events (Parent Span)

Events on the `llm_request` span:

| Event | When | What It Means |
|-------|------|---------------|
| **ARRIVED** | Request received by API server | Client request has been accepted |
| **HANDOFF_TO_CORE** | Request sent to inference engine | API handed off to scheduler |
| **FIRST_RESPONSE_FROM_CORE** | First token received from engine | Prefill complete, tokens flowing |
| **DEPARTED** | Response sent to client | Request completed successfully |
| **ABORTED** | Request terminated with error | Client disconnect, timeout, or error |

### Engine Core Events (Child Span)

Events on the `llm_core` span:

| Event | When | What It Means |
|-------|------|---------------|
| **QUEUED** | Added to scheduler waiting queue | Waiting for GPU resources |
| **SCHEDULED** | Allocated resources, executing | Actively processing on GPU |
| **FIRST_TOKEN** | First output token generated | Prefill done, decode started |
| **PREEMPTED** | Resources reclaimed, paused | Temporarily paused for other requests |
| **FINISHED** | Completed in scheduler | Done processing, resources freed |

**Note on Event Names:** In OTEL traces, events appear with prefixes: API events use `api.<EVENT>` (e.g., `api.ARRIVED`, `api.DEPARTED`), and core events use `journey.<EVENT>` (e.g., `journey.QUEUED`, `journey.FINISHED`). The tables above show the event type names without prefixes for readability.

**Important:** `DEPARTED` is an API layer terminal event only. The core scheduler uses `FINISHED` as its terminal event since "finished processing" is the natural terminus for the scheduler. The API layer tracks the full request lifecycle from arrival to departure, while the core layer focuses on scheduling and execution.

### Event Attributes

Each event includes detailed attributes:

**Progress Tracking:**
- `phase` - Current phase: "WAITING" (queued), "PREFILL" (processing prompt), or "DECODE" (generating output)
- `prefill_done_tokens` / `prefill_total_tokens` - Prompt processing progress
- `decode_done_tokens` / `decode_max_tokens` - Output generation progress

**Timing:**
- `ts.monotonic` - High-precision timestamp for latency calculations
- `scheduler.step` - Scheduler iteration number (for correlation)

**Lifecycle:**
- `num_preemptions` - How many times request was preempted
- `schedule.kind` - Whether this is first schedule or resume after preemption
- `finish.status` - Terminal status: stopped, length, aborted, ignored, error

---

## Understanding Request Flow

### Normal Request (No Preemption)

```
API:    ARRIVED → HANDOFF_TO_CORE → FIRST_RESPONSE_FROM_CORE → DEPARTED
                       │                        │
Core:                  └→ QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED
```

**Timeline:**
1. Request arrives at API server (ARRIVED)
2. API validates and sends to engine (HANDOFF_TO_CORE)
3. Engine queues request for scheduling (QUEUED)
4. Scheduler allocates resources (SCHEDULED)
5. Prefill completes, first token generated (FIRST_TOKEN)
6. First token sent back to API (FIRST_RESPONSE_FROM_CORE)
7. Generation continues, request finishes (FINISHED)
8. Final response sent to client (DEPARTED)

### Request with Preemption

```
Core: QUEUED → SCHEDULED → PREEMPTED → SCHEDULED → FIRST_TOKEN → FINISHED
                 (first)      ↓         (resume)
                              └─ Resources reclaimed temporarily
```

When the scheduler needs to free resources, it may **preempt** running requests:
- Request is paused (PREEMPTED)
- Resources freed for higher-priority requests
- Later, request is resumed (SCHEDULED with kind=RESUME)
- Progress is preserved - prefill work is not lost

### Request with Error

```
API:    ARRIVED → HANDOFF_TO_CORE → ABORTED
                       │
Core:                  └→ QUEUED → SCHEDULED → FINISHED (status=error)
```

Errors can occur at various points:
- Client disconnect → ABORTED (core events may be truncated if disconnect happens before scheduling)
- Validation error → ABORTED (core may not reach FINISHED)
- Generation error → FINISHED with status=error, then ABORTED

**Note:** Depending on when the error occurs, core events may be incomplete. Early client disconnects or validation failures may result in QUEUED without SCHEDULED/FINISHED. The core span ends when resources are freed, regardless of completion status.

---

## Common Use Cases

### 1. Measuring Time-to-First-Token (TTFT)

TTFT is critical for user experience. Journey tracing gives you precise measurements:

**What to look for:**
- Time from ARRIVED to FIRST_RESPONSE_FROM_CORE = End-to-end TTFT
- Time from QUEUED to FIRST_TOKEN = Engine-only TTFT (excludes API overhead)
- Time from SCHEDULED to FIRST_TOKEN = Prefill duration

**In Jaeger:**
1. Find your trace
2. Look at time between events on the timeline
3. Expand span events to see exact timestamps

**Typical TTFT breakdown:**
```
ARRIVED (t=0ms)
  ↓ API validation + request parsing (~2-5ms)
HANDOFF_TO_CORE (t=3ms)
  ↓ Queue waiting time (varies with load)
SCHEDULED (t=150ms)
  ↓ Prefill compute time (depends on prompt length)
FIRST_TOKEN (t=350ms)
  ↓ Network + API overhead (~1-3ms)
FIRST_RESPONSE_FROM_CORE (t=352ms)
```

### 2. Debugging Slow Requests

When a request is slow, traces show you exactly why:

**High queue time?**
- Long gap between QUEUED and SCHEDULED
- Solution: Scale up, optimize scheduling, reduce load

**High prefill time?**
- Long gap between SCHEDULED and FIRST_TOKEN
- Check: prompt length, model size, batch size

**Frequent preemptions?**
- Multiple PREEMPTED events
- Solution: Adjust scheduling policy, increase KV cache

**In Jaeger:**
- Look for the longest gaps in the timeline
- Check event attributes for clues (preemption count, phase, progress)
- Compare slow traces to fast ones to find patterns

### 3. Understanding Preemption Behavior

Preemption can impact request latency. Traces help you understand it:

**What to look for:**
- `num_preemptions` attribute on events (how many times preempted)
- Multiple SCHEDULED events with `schedule.kind=RESUME`
- Progress preserved: `prefill_done_tokens` doesn't reset

**Example trace with preemption:**
```
SCHEDULED (step=10, kind=FIRST, prefill_done=0/100)
  ↓ Processed 40 tokens
PREEMPTED (step=12, prefill_done=40/100)  ← Paused
  ↓ Other requests processed...
SCHEDULED (step=25, kind=RESUME, prefill_done=40/100)  ← Resumed from 40!
  ↓ Completed remaining 60 tokens
FIRST_TOKEN (step=28, prefill_done=100/100)
```

**High preemption impact?**
- Check system load and request patterns
- Consider increasing KV cache size
- Evaluate scheduling policy (FCFS vs priority-based)

### 4. Monitoring Production Workloads

Journey tracing helps you understand system behavior at scale:

**Key metrics to track:**
- P50/P95/P99 TTFT (from ARRIVED to FIRST_RESPONSE_FROM_CORE)
- Queue time distribution (QUEUED to SCHEDULED)
- Preemption rate (% of requests with preemptions)
- Error rate (% of requests ending in ABORTED)

---

## Sampling for Production

When using the OpenAI API server (`vllm serve`), vLLM creates traces for **all requests** by default when journey tracing is enabled. For high-volume production workloads, you have three sampling strategies:

### Strategy 1: vLLM Native Sampling (Recommended ⭐)

**What it does:** vLLM probabilistically samples requests **before creating any spans**. Sampled-out requests have near-zero tracing overhead.

**When to use:** High request rates (>1000 RPS) where you want to minimize vLLM's CPU/memory/network overhead.

**Requirements:** Works with the OpenAI-compatible API server (`vllm serve`). Direct `AsyncLLM` usage requires manual trace header management.

**How it works:**
- OpenAI API server makes a probabilistic sampling decision per request
- If sampled out: no API span, no engine span, no events (near-zero overhead)
- If sampled in: complete trace created (both API + engine spans)
- **End-to-end atomic**: either both spans exist or neither (no partial traces)

**Configuration:**

```bash
# Sample 10% of requests (90% have near-zero tracing overhead)
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317 \
    --journey-tracing-sample-rate 0.1
```

The `--journey-tracing-sample-rate` parameter accepts values from 0.0 to 1.0:
- `1.0` (default) - Trace all requests (100%, backward compatible)
- `0.1` - Trace 10% of requests (recommended for high-volume production)
- `0.01` - Trace 1% of requests (for very high-volume production)
- `0.0` - Trace no requests (effectively disables journey tracing)

**Performance impact:**
- Sampled-out requests: Sub-microsecond overhead (single random number check)
- Sampled-in requests: Normal tracing overhead (~1-3% CPU)
- Network traffic reduced proportionally (10% sampling = 90% less OTLP traffic)

**Example:** At 10,000 RPS with 10% sampling:
- 1,000 requests/sec get full traces (normal overhead)
- 9,000 requests/sec have sub-microsecond overhead (just sampling check)
- 90% reduction in OTLP network traffic to collector

---

### Strategy 2: OTEL SDK Sampling (Application-Level)

**What it does:** OTEL SDK samples traces **after spans are created** based on the W3C Trace Context `sampled` bit.

**When to use:** When you want vLLM to create all spans (full overhead) but only export a subset to the collector.

**How it works:**
- vLLM creates all spans and events (full overhead)
- OTEL SDK decides whether to export each trace to the collector
- Reduces network traffic and backend storage, but **not vLLM overhead**

**Configuration:**

```bash
# Sample 10% at OTEL SDK level
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

**Important:** OTEL sampling is **independent** from vLLM sampling:
- vLLM sampling gates span creation (saves vLLM overhead)
- OTEL sampling gates span export (saves network/storage)
- They can be combined: e.g., vLLM samples 10%, OTEL samples another 10% of those = 1% final

---

### Strategy 3: Collector-Side Sampling (Backend-Level)

**What it does:** OTEL collector samples traces **after receiving them** from vLLM.

**When to use:** When you want to reduce backend storage but keep vLLM overhead (e.g., for debugging).

**How it works:**
- vLLM creates and exports all traces (full overhead)
- Collector decides which traces to forward to backend
- Reduces backend storage only, **not vLLM or network overhead**

**Configuration:**

```yaml
# OTEL collector config.yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 10

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [probabilistic_sampler]
      exporters: [jaeger]
```

---

### Choosing the Right Strategy

| Strategy | Reduces vLLM Overhead | Reduces Network | Reduces Storage | When to Use |
|----------|----------------------|-----------------|-----------------|-------------|
| **vLLM Native** | ✅ Yes | ✅ Yes | ✅ Yes | **Recommended for production** - reduces all overhead |
| **OTEL SDK** | ❌ No | ✅ Yes | ✅ Yes | Want full vLLM instrumentation with selective export |
| **Collector** | ❌ No | ❌ No | ✅ Yes | Need flexible sampling rules at collector level |

**Recommendation for production:**
- **< 1,000 RPS:** Use default (no sampling) - overhead is negligible
- **1,000 - 10,000 RPS:** Use vLLM native sampling at 10-20% (`--journey-tracing-sample-rate 0.1`)
- **> 10,000 RPS:** Use vLLM native sampling at 1-5% (`--journey-tracing-sample-rate 0.01`)

**Combining strategies:**
You can combine vLLM sampling with OTEL/collector sampling for multi-stage reduction:

```bash
# Stage 1: vLLM samples 10% of requests (reduces vLLM overhead by 90%)
# Stage 2: OTEL SDK samples 50% of those (further reduces network by 50%)
# Final result: 5% of requests traced (0.1 * 0.5 = 0.05)

export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.5

vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317 \
    --journey-tracing-sample-rate 0.1
```

---

## Configuration Options

### Required Flags

```bash
--enable-journey-tracing          # Enable the feature
--otlp-traces-endpoint URL        # OTEL collector endpoint
```

### Optional Flags

```bash
--journey-tracing-sample-rate RATE    # Sampling rate [0.0-1.0], default 1.0
                                       # 1.0 = trace all (100%, no sampling)
                                       # 0.1 = trace 10% of requests
                                       # 0.01 = trace 1% of requests
```

### Common Configurations

**Local development (Jaeger, all requests):**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

**Production (external collector, 10% sampling):**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://otel-collector.internal:4317 \
    --journey-tracing-sample-rate 0.1
```

**High-volume production (1% sampling):**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://otel-collector.internal:4317 \
    --journey-tracing-sample-rate 0.01
```

**With other observability features:**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317 \
    --journey-tracing-sample-rate 0.1 \
    --enable-metrics \
    --enable-mfu-metrics
```

### OTEL Exporter Options

vLLM supports standard OTEL environment variables:

```bash
# Alternative: Use environment variables
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=vllm-production  # This is what you'll select in Jaeger UI

vllm serve MODEL --enable-journey-tracing
```

**Note:** `OTEL_SERVICE_NAME` sets the `service.name` resource attribute, which is what appears in the Jaeger/Tempo service dropdown. If not set, it defaults to "vllm". The tracer scopes (`vllm.api`, `vllm.scheduler`) are separate instrumentation scope names that appear as span attributes.

---

## Trace Backends

Journey tracing works with any OTEL-compatible backend:

### Jaeger (Recommended for Development)

```bash
# All-in-one: collector + UI
docker run -d -p 4317:4317 -p 16686:16686 \
  jaegertracing/jaeger:latest
```

View traces at http://localhost:16686

### Grafana Tempo (Recommended for Production)

Tempo is designed for high-volume tracing:

```bash
# Example docker-compose.yml
services:
  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    ports:
      - "4317:4317"  # OTLP gRPC
      - "3200:3200"  # Tempo API
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

Configure Grafana to query Tempo for trace visualization.

### Other Backends

- **Zipkin** - Compatible via OTEL collector
- **Datadog APM** - Use Datadog OTEL exporter
- **New Relic** - Use New Relic OTEL exporter
- **Honeycomb** - Native OTEL support

---

## Performance Impact

### When Disabled (Default)

Journey tracing is **disabled by default** with negligible overhead:
- No spans created
- No events emitted
- Single boolean check per potential emission point

### When Enabled (No Sampling)

With default `journey_tracing_sample_rate=1.0` (100% of requests traced):

**Typical overhead (ballpark estimates):**
- **CPU:** ~1-3% additional CPU for span creation and event emission
- **Memory:** ~1-2KB per active request for span state
- **Network:** ~5-10KB per trace exported to OTEL collector

*These are rough estimates based on typical workloads. Actual overhead varies by request rate, preemption frequency, and OTEL exporter configuration. Measure in your specific environment for accurate numbers.*

**Factors affecting overhead:**
- Request rate (more requests = more spans)
- Preemption rate (more preemptions = more events per request)
- OTEL exporter config (batching reduces overhead)

### When Enabled (With Sampling)

With `journey_tracing_sample_rate < 1.0` (sampling enabled):

**Per-request overhead:**
- **Sampled-out requests:** Sub-microsecond (single random number check + header check)
- **Sampled-in requests:** Normal tracing overhead (~1-3% CPU)

**Overall impact:**
- At 10% sampling: 90% of requests have near-zero overhead
- At 1% sampling: 99% of requests have near-zero overhead
- Network/storage reduced proportionally

**Example: 10,000 RPS with 10% sampling:**
- 9,000 req/s: Sub-microsecond overhead each (negligible)
- 1,000 req/s: Normal tracing overhead (~1-3% CPU)
- Network: 90% reduction in OTLP traffic
- Overall CPU impact: ~0.1-0.3% instead of ~1-3%

**Recommendations:**
- ✅ Safe to enable in production with moderate traffic (<1000 RPS) without sampling
- ✅ Use vLLM native sampling for high-volume production (>1000 RPS)
  - 1,000-10,000 RPS: `--journey-tracing-sample-rate 0.1` (10%)
  - >10,000 RPS: `--journey-tracing-sample-rate 0.01` (1%)
- ✅ Monitor CPU/memory before and after enabling
- ✅ Adjust sample rate based on observed overhead
- ⚠️ Overhead scales with request rate, not model size

---

## Troubleshooting

### Traces Not Appearing

**Problem:** No traces showing up in Jaeger/Tempo

**Check:**

1. **Is journey tracing enabled?**
   ```bash
   # Check your vllm serve command includes:
   --enable-journey-tracing
   ```

2. **Is OTLP endpoint correct?**
   ```bash
   # Verify endpoint is reachable
   curl http://localhost:4317

   # Check vllm logs for connection errors
   grep -i "otlp" /path/to/vllm.log
   ```

3. **Is OTEL collector running?**
   ```bash
   # Check Jaeger is running
   docker ps | grep jaeger

   # Check collector logs
   docker logs jaeger
   ```

4. **Send a test request:**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "your-model",
       "messages": [{"role": "user", "content": "test"}]
     }'

   # Wait a few seconds, then check Jaeger UI
   ```

### Missing Events

**Problem:** Some events missing from traces

**Common causes:**

1. **Request aborted early**
   - DEPARTED/ABORTED events may be missing if process crashed
   - Check vllm server logs for errors

2. **FIRST_TOKEN never emitted**
   - Request may have finished during prefill (max_tokens=0?)
   - Check FINISHED event for decode_done_tokens=0

3. **OTEL collector dropping events**
   - Check collector is not overloaded
   - Increase collector resource limits

### High Overhead

**Problem:** Journey tracing causing performance issues

**Solutions:**

1. **Use vLLM native sampling** (recommended - reduces all overhead):
   ```bash
   # Sample 10% of requests at source (90% have near-zero overhead)
   vllm serve MODEL \
       --enable-journey-tracing \
       --otlp-traces-endpoint http://localhost:4317 \
       --journey-tracing-sample-rate 0.1
   ```

   This is the most effective solution because:
   - ✅ Reduces vLLM CPU/memory overhead (sampled-out requests skip span creation)
   - ✅ Reduces network traffic to collector (fewer traces exported)
   - ✅ Reduces backend storage (fewer traces stored)

2. **Alternative: OTEL SDK sampling** (reduces network/storage only):
   ```bash
   # Set before starting vLLM to sample 10% at OTEL SDK level
   export OTEL_TRACES_SAMPLER=traceidratio
   export OTEL_TRACES_SAMPLER_ARG=0.1

   vllm serve MODEL --enable-journey-tracing --otlp-traces-endpoint http://localhost:4317
   ```

   Note: This still creates all spans in vLLM (full overhead), only exports subset.

3. **Use batch export:**
   ```bash
   # OTEL exporter batches traces
   export OTEL_BSP_SCHEDULE_DELAY=5000  # Batch every 5s
   export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
   ```

4. **Check request rate:**
   - High request rates (>1000 RPS) should use vLLM sampling
   - Monitor CPU/memory after enabling
   - Adjust sample rate based on observed overhead

### Trace Context Issues

**Problem:** Parent-child spans not linked

**Check:**

1. **W3C Trace Context propagation**
   - vLLM automatically propagates context from API to engine
   - No manual configuration needed

2. **Trace IDs match?**
   - In Jaeger, both spans should have same trace ID
   - If different, context propagation failed (file a bug)

---

## FAQ

### Q: Do I need to change my client code?

**A:** No. Journey tracing is transparent to clients. Just enable it on the server.

### Q: Does this work with all models?

**A:** Yes. Journey tracing works with any model served by vLLM.

### Q: Does this work with distributed inference?

**A:** Currently, journey tracing tracks per-instance. For multi-node tensor parallelism, each node emits its own traces. Correlation across nodes is future work.

### Q: Can I use this with Prometheus?

**A:** Journey tracing uses OTEL spans, not Prometheus metrics. vLLM has separate Prometheus metrics (use `--enable-metrics`). Some OTEL collectors can convert span data to metrics.

### Q: What's the data retention?

**A:** Depends on your backend:
- Jaeger default: 24 hours
- Tempo: Configurable (days to months)
- Configure based on your storage capacity

### Q: Can I export to multiple backends?

**A:** Yes, use an OTEL collector with multiple exporters:

```yaml
exporters:
  jaeger:
    endpoint: localhost:14250
  otlp/tempo:
    endpoint: tempo:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger, otlp/tempo]
```

### Q: What if my OTEL collector is down?

**A:** vLLM will log warnings but continue serving requests normally. Tracing is best-effort and never blocks request processing.

### Q: Can I customize span names or attributes?

**A:** Not currently. Span names (`llm_request`, `llm_core`) and event types are fixed for consistency.

### Q: What's the difference between vLLM sampling and OTEL sampling?

**A:** They operate at different levels:

**vLLM Native Sampling** (`--journey-tracing-sample-rate`):
- Decides **before creating spans** whether to trace a request
- Sampled-out requests have sub-microsecond overhead
- Reduces vLLM CPU, memory, network, and storage
- **End-to-end atomic**: either both API and engine spans exist, or neither
- Works with OpenAI API server (`vllm serve`)
- Recommended for production workloads

**OTEL SDK Sampling** (`OTEL_TRACES_SAMPLER` env vars):
- Decides **after creating spans** whether to export them
- All requests have full span creation overhead
- Only reduces network traffic and backend storage
- Controlled by OpenTelemetry SDK (W3C Trace Context sampled bit)

**Independence:** These are completely independent:
- You can use both together (e.g., vLLM samples 10%, OTEL samples 50% of those → 5% final)
- vLLM sampling does **not** use or modify the W3C traceparent sampled bit
- OTEL sampling can still drop traces that vLLM created

**Example combinations:**
```bash
# Option 1: vLLM sampling only (recommended)
vllm serve MODEL --journey-tracing-sample-rate 0.1  # 10% sampled

# Option 2: OTEL sampling only (full vLLM overhead)
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampled

# Option 3: Both (multiplicative: 0.1 * 0.5 = 5% final)
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.5
vllm serve MODEL --journey-tracing-sample-rate 0.1
```

### Q: Are partial traces possible (API span without engine span)?

**A:** No. vLLM's native sampling is **end-to-end atomic**:
- When the API samples a request, it creates the API span AND signals the engine to create the engine span
- When the API samples out a request, neither span is created
- The custom header `x-vllm-journey-sampled` propagates the decision from API to engine
- This guarantees complete traces: either both spans exist with all events, or nothing exists

This atomicity is maintained even in distributed deployments where API and engine run in separate processes.

### Q: Does sampling work with AsyncLLM (direct Python usage)?

**A:** Partially, but with important differences:

**OpenAI API Server (`vllm serve`):**
- ✅ Automatic sampling works out of the box
- Set `--journey-tracing-sample-rate` and requests are automatically sampled
- API server handles all header management

**Direct AsyncLLM usage:**
- ❌ No automatic sampling - `AsyncLLM` doesn't implement sampling logic
- ✅ Manual control available - set `trace_headers={"x-vllm-journey-sampled": "1"}` on requests you want traced
- **Important:** Header value must be exactly `"1"` (string) - any other value is treated as not sampled
- Without the header, the engine will skip span creation (conservative behavior)
- Useful when you want explicit control over which specific requests to trace

**Example:**
```python
from vllm.v1.engine.async_llm import AsyncLLM

llm = AsyncLLM(...)

# To trace this request, manually set the header:
llm.add_request(
    request_id="req1",
    prompt="Hello",
    params=sampling_params,
    trace_headers={"x-vllm-journey-sampled": "1"}  # Manual header
)

# Without header, no tracing happens (engine skips span creation)
llm.add_request(
    request_id="req2",
    prompt="World",
    params=sampling_params,
)
```

---

## Technical Details (Advanced)

This section covers internal implementation details for developers and advanced users.

### Sampling Architecture

**Authority Model (OpenAI API Server):**
- **OpenAI API server** is the authority: makes the sampling decision
- **Engine layer** obeys: checks the decision and creates span only if sampled
- Decision propagates via custom header: `x-vllm-journey-sampled: 1`
- **Note:** This applies when using `vllm serve`. Direct `AsyncLLM` usage requires manual header management.

**Decision Flow (OpenAI API Server):**
```
1. Request arrives at OpenAI API server
2. API server checks: random.random() < journey_tracing_sample_rate?
   - If NO: Return early, no API span created, no header set → engine skips too
   - If YES: Create API span, set header "x-vllm-journey-sampled: 1"
3. Request forwarded to engine with trace headers
4. Engine checks: header present and value == "1"?
   - If NO: Skip core span creation
   - If YES: Create core span as child of API span
```

**Atomicity Guarantee:**
- API span created ⟺ Header set to "1" ⟺ Engine span created
- No partial traces: either both spans exist or neither
- Thread-safe: decision made once per request at API boundary

**W3C Trace Context Independence:**
- vLLM sampling uses custom header `x-vllm-journey-sampled` (not W3C traceparent sampled bit)
- W3C traceparent still propagated normally for OTEL context linking
- OTEL SDK sampling (traceparent sampled bit) is completely independent
- Client-provided traceparent is respected and propagated unchanged

**Deployment Modes:**

**OpenAI API Server (`vllm serve`):**
- Automatic sampling: API server makes probabilistic decision per request
- Works in both single-process and multi-process deployments
- Headers propagate automatically (in-process dict or serialized)
- No manual configuration needed

**Direct AsyncLLM Usage:**
- **No automatic sampling**: `AsyncLLM` does not implement sampling logic
- Users must manually set `trace_headers` parameter with `x-vllm-journey-sampled: "1"` if tracing desired
- Without header, engine will skip core span creation (conservative behavior)
- Useful when caller wants explicit control over which requests to trace

### Span Hierarchy

All vLLM journey traces follow this structure:

```
Trace (trace_id from client or generated)
└─ llm_request span (API layer, SpanKind.SERVER)
   ├─ Attributes: gen_ai.request_id, gen_ai.response.model, etc.
   ├─ Events: api.ARRIVED, api.DEPARTED / api.ABORTED
   └─ llm_core span (Engine layer, SpanKind.INTERNAL, child of llm_request)
      ├─ Attributes: gen_ai.request_id
      └─ Events: journey.QUEUED, journey.SCHEDULED, journey.FIRST_TOKEN, etc.
```

**Parent-Child Linking:**
- API span context injected into `trace_headers` dict using W3C Trace Context propagation
- Engine extracts parent context from `trace_headers` when creating core span
- Same trace ID, different span IDs, proper parent-child relationship

**Tracer Scopes:**
- `vllm.api` - API layer tracer (creates `llm_request` spans)
- `vllm.scheduler` - Engine layer tracer (creates `llm_core` spans)
- Both use the same global OTEL TracerProvider (singleton pattern)
- **Important:** Scope names (`vllm.api`, `vllm.scheduler`) are NOT service names. In Jaeger/Tempo UI, you select traces by `service.name` (e.g., "vllm", configured via `OTEL_SERVICE_NAME`), then view scope names as attributes within spans.

### Backward Compatibility

The default behavior is **identical** to versions without sampling support:

**Default config:**
- `journey_tracing_sample_rate = 1.0` (100%, no sampling)
- When set to 1.0: `random.random() < 1.0` always true → always sample

**Migration path:**
- Existing deployments: No changes needed, works as before
- Opt-in sampling: Add `--journey-tracing-sample-rate` flag when ready
- Gradual rollout: Start with 0.5 (50%), decrease based on needs

**Config precedence:**
1. `enable_journey_tracing` must be True (master switch)
2. `otlp_traces_endpoint` must be set (where to send traces)
3. `journey_tracing_sample_rate` determines what fraction (default 1.0)

If `enable_journey_tracing=False`, no traces are created regardless of sample rate.

---

## Kubernetes Example

The `testk8s/collectanddebug.yaml` configuration demonstrates journey tracing (along with step tracing) in Kubernetes with file-based trace collection for debugging.

### Configuration Highlights

- **Journey tracing enabled**: `--enable-journey-tracing` captures request lifecycle
- **Step tracing enabled**: `--step-tracing-enabled` captures scheduler-level events
- **File exporter**: OTEL collector writes traces to persistent volume at `/data/traces.json`
- **Debug pod**: Provides easy access to trace files via `kubectl cp`

### Quick Start

```bash
# 1. Deploy vLLM with OTEL collector and debug pod
kubectl apply -f testk8s/collectanddebug.yaml

# 2. Wait for vLLM server to be ready
kubectl wait --for=condition=ready pod -l app=vllm-server-opt --timeout=300s

# 3. Send test requests to generate traces
kubectl port-forward svc/vllm-server 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 4. Wait ~15 seconds for traces to export to file

# 5. Extract traces from debug pod
kubectl cp pvc-debug:/mnt/data/traces.json ./traces.json

# 6. View journey traces (API layer - vllm.api scope)
jq '.resourceSpans[].scopeSpans[] | select(.scope.name == "vllm.api")' traces.json

# 7. View journey traces (Engine layer - vllm.scheduler scope)
jq '.resourceSpans[].scopeSpans[] | select(.scope.name == "vllm.scheduler")' traces.json

# 8. View all journey events across both layers
jq '.resourceSpans[].scopeSpans[] |
    select(.scope.name == "vllm.api" or .scope.name == "vllm.scheduler") |
    .spans[].events[]' traces.json
```

### What You'll See

**API layer spans (`vllm.api` scope):**
- Span name: `llm_request`
- Events: `api.ARRIVED`, `api.HANDOFF_TO_CORE`, `api.FIRST_RESPONSE_FROM_CORE`, `api.DEPARTED`

**Engine layer spans (`vllm.scheduler` scope):**
- Span name: `llm_core`
- Events: `journey.QUEUED`, `journey.SCHEDULED`, `journey.FIRST_TOKEN`, `journey.FINISHED`

**Parent-child relationship:**
- Both spans share the same `trace_id`
- `llm_core` span has `parent_span_id` pointing to `llm_request` span
- Complete request journey visible in the trace hierarchy

### Why This is Useful

This configuration is ideal for:
- **Debugging**: File-based collection allows offline analysis
- **Testing**: Fast span closure intervals for quick iteration
- **Learning**: See both journey and step tracing working together
- **CI/CD**: Extract traces programmatically for validation

For production deployments, replace the file exporter with a backend like Jaeger or Tempo (see [Trace Backends](#trace-backends)).

---

## What's Next?

**For more help:**
- 💬 Ask in [vLLM Discord](https://discord.gg/vllm) #observability channel
- 🐛 File issues at [github.com/vllm-project/vllm/issues](https://github.com/vllm-project/vllm/issues)
- 📖 See [OTEL documentation](https://opentelemetry.io/docs/) for collector setup

**Related observability features:**
- `--enable-metrics` - Prometheus metrics
- `--enable-mfu-metrics` - Model FLOPs utilization
- `--enable-logging-iteration-details` - Detailed scheduler logs
- Step-level tracing (advanced) - Batch summary and per-request snapshots at scheduler step level
  - Enabled via config: `step_tracing_enabled`, `step_tracing_sample_rate`, `step_tracing_rich_subsample_rate`
  - Emits `step.BATCH_SUMMARY` and `step.REQUEST_SNAPSHOT` events
  - For advanced users needing scheduler-level observability

**Advanced topics:**
- Custom OTEL collector pipelines
- Sampling strategies for production
- Correlating traces with logs and metrics

---

**Happy tracing! 🔍✨**
