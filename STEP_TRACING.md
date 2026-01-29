# Step-Level Tracing in vLLM

**Real-time scheduler observability using OpenTelemetry distributed tracing**

---

## What is Step-Level Tracing?

Step-level tracing gives you visibility into how vLLM's scheduler operates over time, showing aggregate metrics for each scheduling iteration (called a "step"). Unlike request journey tracing which follows individual requests, step tracing shows you the **system-wide behavior**: queue depths, batch composition, token throughput, and KV cache pressure.

Each scheduler step processes a batch of requests, decides which get GPU resources, and tracks progress. Step tracing samples these iterations and emits two types of events:
- **Batch Summary** - Aggregate metrics for the entire batch (queue sizes, token counts, KV cache usage)
- **Request Snapshots** - Detailed per-request progress for running requests (optional, higher cardinality)

Events are emitted to OpenTelemetry (OTEL), giving you near-real-time visibility into scheduler performance (timing depends on your OTEL SDK and collector configuration).

---

## When to Use Step Tracing vs Journey Tracing

**Use Journey Tracing** (`--enable-journey-tracing`) when you want to:
- Debug individual slow requests
- Measure per-request Time-to-First-Token (TTFT)
- Understand a specific request's lifecycle from arrival to completion
- Track preemptions for particular requests

**Use Step Tracing** (`--step-tracing-enabled`) when you want to:
- Monitor scheduler performance and throughput
- Detect queue buildup and saturation
- Understand batch composition (prefill vs decode mix)
- Track KV cache pressure over time
- Debug system-wide performance issues

**Use Both Together** for complete observability:
- Journey tracing shows *what happened to request X*
- Step tracing shows *how the scheduler is performing overall*
- They complement each other and use different OTEL tracer scopes

---

## Quick Start

### Step 1: Start an OTEL Collector

Same as journey tracing - use Jaeger or any OTEL-compatible backend:

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

### Step 2: Start vLLM with Step Tracing

```bash
# Enable step tracing with default sampling (recommended for production)
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --step-tracing-enabled \
    --otlp-traces-endpoint http://localhost:4317
```

That's it! Step tracing is now enabled with default sampling (1% of steps). This is appropriate for many production workloads; adjust based on your step rate and observability backend capacity.

### Step 3: Send Some Requests

```bash
# Send multiple requests to generate scheduler activity
for i in {1..10}; do
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "messages": [{"role": "user", "content": "Hello!"}]
    }' &
done
wait
```

### Step 4: View Step Traces in Jaeger

1. Open http://localhost:16686
2. Select your vLLM service (default: "vllm")
3. Click "Find Traces"
4. Look for traces containing a `scheduler_steps` span with `step.BATCH_SUMMARY` events. The exact UI varies by backend:
   - **Jaeger:** Select service → expand span → view events
   - **Other backends:** Consult your tracing UI documentation for viewing span events

You'll see a timeline showing sampled scheduler steps with aggregate metrics.

---

## What You'll See

### Batch Summary Events

Each sampled scheduler step emits a `step.BATCH_SUMMARY` event with:

**Queue State:**
- Running queue depth (actively processing requests)
- Waiting queue depth (queued requests)

**Batch Composition:**
- Number of prefill requests (processing prompts)
- Number of decode requests (generating tokens)
- Total tokens scheduled this step

**Token Distribution:**
- Prefill tokens (prompt processing)
- Decode tokens (generation)
- Total scheduled tokens

**Request Lifecycle:**
- Requests that finished this step
- Requests that were preempted this step

**KV Cache Health:**
- Cache usage ratio (0.0 to 1.0)
- Total GPU blocks available
- Free GPU blocks remaining

### Request Snapshot Events (Optional)

When rich snapshots are enabled, you also get `step.REQUEST_SNAPSHOT` events for each running request:

**Per-Request Details:**
- Request ID and current phase (PREFILL/DECODE)
- Token counts (prompt, computed, output)
- Preemption count
- Tokens scheduled this step

**Per-Request KV Cache:**
- GPU blocks allocated to this request
- GPU blocks with prefix cache hits
- Effective prompt length (after cache reduction)

---

## Common Use Cases

### 1. Monitoring Scheduler Performance

**What to look for:**
- Step duration trends (detect slowdowns)
- Queue depth growth (detect saturation)
- Batch utilization (prefill/decode balance)

**In Jaeger:**
- View `step.duration_us` over time
- Track `queue.waiting_depth` for buildup
- Monitor `batch.scheduled_tokens` for throughput

### 2. Detecting KV Cache Pressure

**What to look for:**
- High `kv.usage_gpu_ratio` (approaching 1.0)
- Low `kv.blocks_free_gpu`
- Increased preemption rate

**Typical pattern (values vary by workload):**
```
Step 100: kv.usage_gpu_ratio=0.65, batch.num_preempted=0
Step 150: kv.usage_gpu_ratio=0.85, batch.num_preempted=2
Step 200: kv.usage_gpu_ratio=0.95, batch.num_preempted=8  ← Cache pressure!
```

**Action:** Consider increasing `--gpu-memory-utilization` or reducing load.

### 3. Understanding Batch Composition

**What to look for:**
- Prefill/decode request ratio
- Token distribution between prefill and decode
- Decode-only steps vs mixed steps

**Example insights:**
- High prefill ratio = more new requests arriving
- All-decode steps = stable generation phase
- Imbalanced token counts = suboptimal batching

### 4. Debugging Queue Buildup

**What to look for:**
- Growing `queue.waiting_depth` over time
- Step duration increasing
- Decreasing tokens per step

**Typical diagnosis:**
```
Step 50:  waiting=5,  duration=15ms, tokens=2048
Step 100: waiting=12, duration=18ms, tokens=1856  ← Slowing down
Step 150: waiting=25, duration=22ms, tokens=1600  ← Queue growing!
```

**Action:** Scale up workers or reduce request rate.

---

## Configuration Options

### Required Flags

```bash
--step-tracing-enabled            # Enable step tracing
--otlp-traces-endpoint URL        # OTEL collector endpoint
```

### Optional Flags

```bash
--step-tracing-sample-rate RATE              # Batch summary sampling [0.0-1.0]
                                              # Default: 0.01 (1% of steps)

--step-tracing-rich-subsample-rate RATE      # Rich snapshot subsampling [0.0-1.0]
                                              # Default: 0.001 (0.1% of steps)
```

### Common Configurations

**Development (high visibility):**
```bash
vllm serve MODEL \
    --step-tracing-enabled \
    --step-tracing-sample-rate 1.0 \
    --step-tracing-rich-subsample-rate 1.0 \
    --otlp-traces-endpoint http://localhost:4317
```
- Every step traced with full detail
- High overhead, only for local debugging

**Production (recommended ⭐):**
```bash
vllm serve MODEL \
    --step-tracing-enabled \
    --step-tracing-sample-rate 0.01 \
    --otlp-traces-endpoint http://otel-collector:4317
```
- 1% of steps sampled (batch summary only)
- No per-request snapshots (keeps cardinality low)
- **This is the default if you only set --step-tracing-enabled**

**Production with occasional deep dives:**
```bash
vllm serve MODEL \
    --step-tracing-enabled \
    --step-tracing-sample-rate 0.05 \
    --step-tracing-rich-subsample-rate 0.02 \
    --otlp-traces-endpoint http://otel-collector:4317
```
- 5% batch summaries
- 0.1% rich snapshots (5% × 2% = 0.1%)
- Good for occasional per-request debugging

**High-volume production (minimal overhead):**
```bash
vllm serve MODEL \
    --step-tracing-enabled \
    --step-tracing-sample-rate 0.001 \
    --otlp-traces-endpoint http://otel-collector:4317
```
- 0.1% of steps sampled
- Minimal data for trend analysis only

---

## Performance Impact

### When Disabled (Default)

Step tracing is **disabled by default** with negligible overhead:
- No spans created
- No events emitted
- Single boolean check per step

### When Enabled (Production Defaults)

With `step_tracing_sample_rate=0.01` (1%, default):

**Overhead per step:**
- 99% of steps: Sub-microsecond (single random number check)
- 1% of steps: ~10-50 microseconds for metric collection + event emission on typical server hardware (varies by CPU, system load, and OTEL SDK configuration)

**Overall impact:**
- CPU: Typically <0.1% additional CPU usage at moderate step rates (impact scales with steps/sec and running request count)
- Memory: ~1-2KB per sampled step (batch summary only)
- Network: ~1-2KB per event × sample rate × steps/sec

**Example workload:**
- 1000 scheduler steps/sec
- 1% sampling rate
- 10 events/sec @ 1.5KB each = 15KB/sec network traffic
- **Impact: Negligible**

### With Rich Snapshots Enabled

Adding rich snapshots increases overhead:

**Per running request:**
- ~20-30 microseconds per request for KV metrics lookup + event emission on typical deployments (varies by system load and KV cache size)

**Example with 100 running requests:**
- Batch summary: ~50μs
- Rich snapshots: ~2-3ms (100 requests × 25μs)
- **Total per rich-sampled step: ~3ms**

**With defaults (0.01 × 0.001 = 0.00001 effective rate):**
- At 1000 steps/sec: 0.01 rich-sampled steps/sec
- 0.01 × 3ms = **0.03ms/sec overhead = negligible**

### Recommendations

**Recommended Starting Points:**
- **< 500 steps/sec:** Defaults (0.01 batch, 0.001 rich) work well for most setups
- **500-2000 steps/sec:** Consider reducing to (0.005 batch, 0.0 rich) depending on backend capacity
- **> 2000 steps/sec:** Start with (0.001 batch, 0.0 rich) and increase if needed

Test in your environment and adjust based on observed overhead and backend capacity.

**General rule:** Batch summaries are cheap, rich snapshots are expensive. Adjust rich rate based on cardinality needs.

---

## Important Warnings

### ⚠️ Rich Snapshots Can Generate High Cardinality

**Problem:** With 100 running requests and rich sampling enabled, each sampled step generates 100+ events.

**Solution:**
- Keep `step-tracing-rich-subsample-rate` very low (≤ 0.01) in production
- Use rich snapshots only for targeted debugging, not continuous monitoring
- Default 0.001 rate is safe for most workloads

### ⚠️ Events Are Sampled Independently

**Behavior:** Each step is sampled independently with probability = sample rate.

**Implication:** You will see gaps in the timeline. This is normal and expected.

**Example:**
```
Step 1:  Sampled ✓ (event emitted)
Step 2:  Not sampled (no event)
Step 3:  Not sampled (no event)
Step 4:  Sampled ✓ (event emitted)
Step 5:  Not sampled (no event)
...
```

For continuous monitoring, aggregate across many steps or reduce sample rate.

### ⚠️ Not a Replacement for Metrics

Step tracing provides **sampled snapshots**, not continuous metrics:

**Use step tracing for:**
- Debugging performance issues
- Understanding scheduler behavior
- Root cause analysis

**Use Prometheus metrics (`--enable-metrics`) for:**
- Continuous monitoring and alerting
- Dashboard gauges and counters
- Production SLO tracking

**Best practice:** Use both together - metrics for monitoring, tracing for debugging.

---

## Troubleshooting

### No Step Events Appearing

**Check:**
1. Is `--step-tracing-enabled` set?
2. Is `--otlp-traces-endpoint` correct and reachable?
3. Is sampling rate > 0.0? (Default 0.01 is fine)
4. Are you sending requests? (Empty idle periods won't generate events)

**Verify:**
```bash
# Check vLLM logs for step tracing initialization
grep -i "step tracing" /path/to/vllm.log

# Should see log messages about step tracing initialization (exact wording may vary by vLLM version)
```

### Too Many Events (High Cardinality)

**Symptoms:**
- OTEL collector lagging or dropping events
- High network bandwidth to collector
- Backend storage filling up quickly

**Solutions:**
1. **Reduce batch sampling:** `--step-tracing-sample-rate 0.001`
2. **Disable rich snapshots:** `--step-tracing-rich-subsample-rate 0.0`
3. **Use collector-side sampling:** Add tail sampling processor to OTEL collector

### Events Missing Attributes

**Expected behavior:**
- Some attributes are optional (e.g., `request.effective_prompt_len` only present with prefix caching)
- If an attribute is consistently missing, check vLLM version

**Debug:**
```bash
# Check which attributes are present in your events
# In Jaeger, expand event details to see all attributes
```

### Performance Degradation After Enabling

**Likely causes:**
1. **Rich subsample rate too high** → Reduce to 0.001 or disable (0.0)
2. **Sample rate too high for your step rate** → Reduce to 0.001-0.01
3. **OTEL collector backpressure** → Check collector logs and increase resources

**Quick fix:**
```bash
# Temporarily reduce all sampling
--step-tracing-sample-rate 0.001
--step-tracing-rich-subsample-rate 0.0
```

---

## FAQ

### Q: Do I need to change my client code?

**A:** No. Step tracing is transparent to clients. Enable it on the server side only.

### Q: Can I use step tracing with journey tracing?

**A:** Yes! They are complementary:
- Journey tracing: Per-request lifecycle events
- Step tracing: System-wide scheduler metrics

Both can be enabled simultaneously:
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --step-tracing-enabled \
    --otlp-traces-endpoint http://localhost:4317
```

They use different tracer scopes and don't interfere with each other.

### Q: How do I find step events in Jaeger?

**A:**
1. Select your service (e.g., "vllm") in the service dropdown
2. Find traces with the `scheduler_steps` span
3. Look for spans with scope name `vllm.scheduler.step` (visible in span details)
4. Expand the span to see `step.BATCH_SUMMARY` and `step.REQUEST_SNAPSHOT` events

### Q: What's the difference between the two sample rates?

**A:**
- **`--step-tracing-sample-rate`** - Controls batch summary events (one per step)
- **`--step-tracing-rich-subsample-rate`** - Controls rich snapshot events (many per step)

They multiply together:
- Batch only: `0.01` = 1% of steps get batch summaries
- Both: `0.01 × 0.02` = 0.0002 = 0.02% of steps get rich snapshots

Rich snapshots are **only emitted if the batch summary was sampled first** (two-stage sampling).

### Q: Are events guaranteed to appear in order?

**A:**
- Events on the same span are ordered by timestamp
- `step.id` attribute provides ordering across all events
- However, due to sampling, you'll see gaps (step 1, 5, 8, 12...)

### Q: Can I sample specific steps (not random)?

**A:** No. Sampling is purely probabilistic. Every step has an independent random chance of being sampled based on the configured rate.

For deterministic analysis, set `--step-tracing-sample-rate 1.0` temporarily.

### Q: What happens if OTEL collector is down?

**A:** vLLM will log warnings and continue serving requests. The exact behavior depends on your OTEL SDK configuration:
- Default `BatchSpanProcessor`: Continues normally, may drop events if export queue fills
- Misconfigured exporters: May experience brief delays or event loss
- Step tracing failures never crash the scheduler, but check your OTEL SDK logs if you see issues.

### Q: How long is the `scheduler_steps` span?

**A:** It's a long-lived span created at scheduler initialization and active for the lifetime of the scheduler process. All step events are emitted on this single span.

**On graceful shutdown:** The span may be closed if the OTEL SDK has time to flush (depends on shutdown timeout and SDK configuration).

**On crashes:** The span will remain open in your tracing backend, which is normal. Incomplete spans indicate process termination.

This is different from journey tracing, where each request gets its own short-lived spans.

### Q: Can I export to multiple backends?

**A:** Yes, configure your OTEL collector with multiple exporters:

```yaml
# otel-collector-config.yaml
exporters:
  jaeger:
    endpoint: jaeger:14250
  otlp/tempo:
    endpoint: tempo:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger, otlp/tempo]
```

---

## Advanced Topics

### Detailed Event Attributes

<details>
<summary><b>Complete Batch Summary Attributes</b></summary>

`step.BATCH_SUMMARY` events include:

**Timing:**
- `step.id` (int) - Monotonically increasing step counter
- `step.ts_start_ns` (int) - Step start timestamp (nanoseconds, monotonic clock)
- `step.ts_end_ns` (int) - Step end timestamp (nanoseconds, monotonic clock)
- `step.duration_us` (int) - Step duration in microseconds

**Queue Depths:**
- `queue.running_depth` (int) - Requests currently being processed
- `queue.waiting_depth` (int) - Requests waiting in queue

**Batch Composition:**
- `batch.num_prefill_reqs` (int) - Requests in prefill phase
- `batch.num_decode_reqs` (int) - Requests in decode phase

**Token Counts:**
- `batch.scheduled_tokens` (int) - Total tokens scheduled this step
- `batch.prefill_tokens` (int) - Tokens for prefill operations
- `batch.decode_tokens` (int) - Tokens for decode operations

**Lifecycle Counts:**
- `batch.num_finished` (int) - Requests that completed this step
- `batch.num_preempted` (int) - Requests that were preempted this step

**KV Cache Metrics:**
- `kv.usage_gpu_ratio` (float) - GPU cache usage, range [0.0, 1.0]
- `kv.blocks_total_gpu` (int) - Total GPU KV blocks available
- `kv.blocks_free_gpu` (int) - Free GPU KV blocks

**Expected Relationships (not guaranteed):**
These relationships typically hold but may not be true in all configurations:
- `batch.prefill_tokens + batch.decode_tokens == batch.scheduled_tokens` (in standard configs without speculative decoding)
- `kv.blocks_free_gpu <= kv.blocks_total_gpu`
- `0.0 <= kv.usage_gpu_ratio <= 1.0`

Do not write monitoring that assumes these relationships are invariants. Use them for sanity checking only.

</details>

<details>
<summary><b>Complete Request Snapshot Attributes</b></summary>

`step.REQUEST_SNAPSHOT` events include:

**Correlation:**
- `step.id` (int) - Same as batch summary (for correlation)
- `request.id` (string) - Unique request identifier

**Request State:**
- `request.phase` (string) - "PREFILL" or "DECODE"
- `request.num_prompt_tokens` (int) - Total prompt tokens
- `request.num_computed_tokens` (int) - Tokens computed so far (prompt + output)
- `request.num_output_tokens` (int) - Output tokens generated
- `request.num_preemptions` (int) - Times this request was preempted
- `request.scheduled_tokens_this_step` (int) - Tokens scheduled for this request in this step

**Per-Request KV Cache:**
- `kv.blocks_allocated_gpu` (int) - GPU blocks allocated to this request
- `kv.blocks_cached_gpu` (int) - GPU blocks from prefix cache hits
- `request.effective_prompt_len` (int, optional) - Effective prompt length after cache reduction

**Notes:**
- `request.effective_prompt_len` only present if prefix caching is enabled and cache was used
- Phase determined by: "PREFILL" if `num_output_tokens == 0`, else "DECODE"

</details>

### Sampling Mathematics

<details>
<summary><b>Understanding Two-Stage Sampling</b></summary>

Step tracing uses **two independent sampling stages**:

**Stage 1: Batch Summary Sampling**
- Probability: `step_tracing_sample_rate` (default 0.01)
- Decision: Made per step, independent
- Result: If sampled, emit `step.BATCH_SUMMARY` event

**Stage 2: Rich Snapshot Subsampling**
- Probability: `step_tracing_rich_subsample_rate` (default 0.001)
- Decision: Only evaluated if Stage 1 passed
- Result: If sampled, emit `step.REQUEST_SNAPSHOT` for each running request

**Effective Sample Rates:**
```
Batch summary rate = step_tracing_sample_rate
Rich snapshot rate = step_tracing_sample_rate × step_tracing_rich_subsample_rate

Examples:
- (0.01, 0.001) → Batch: 1%, Rich: 0.001%
- (0.05, 0.02)  → Batch: 5%, Rich: 0.1%
- (1.0, 1.0)    → Batch: 100%, Rich: 100%
- (0.01, 0.0)   → Batch: 1%, Rich: 0% (disabled)
```

**Event Volume Estimation:**
```
Given:
- S = steps per second
- R = average running requests
- B = batch sample rate
- r = rich subsample rate

Events per second:
- Batch summaries: S × B
- Rich snapshots: S × B × r × R

Example (1000 steps/sec, 50 requests, defaults):
- Batch: 1000 × 0.01 = 10 events/sec
- Rich: 1000 × 0.01 × 0.001 × 50 = 0.5 events/sec
```

**Why Two Stages?**
- Batch summaries are **low cardinality** (1 per step) → can afford higher sampling
- Rich snapshots are **high cardinality** (N per step, N=requests) → need aggressive subsampling
- Decoupling allows independent cost/benefit tuning

</details>

### Span Architecture

<details>
<summary><b>OTEL Span Structure</b></summary>

Step tracing creates a single long-lived span:

```
Service: vllm (or value of OTEL_SERVICE_NAME)
└─ Tracer Scope: vllm.scheduler.step
   └─ Span: scheduler_steps (SpanKind.INTERNAL)
      ├─ Event: step.BATCH_SUMMARY (step=1)
      ├─ Event: step.REQUEST_SNAPSHOT (step=1, req=A)
      ├─ Event: step.REQUEST_SNAPSHOT (step=1, req=B)
      ├─ Event: step.BATCH_SUMMARY (step=3) [step 2 not sampled]
      ├─ Event: step.REQUEST_SNAPSHOT (step=3, req=A)
      └─ ...continues for lifetime of scheduler
```

**Key Properties:**
- **Span lifetime:** Created at scheduler initialization, closed on graceful shutdown (if OTEL SDK flushes successfully). On crashes, span remains open indefinitely.
- **Span name:** `scheduler_steps`
- **Span kind:** `INTERNAL` (not client/server)
- **Tracer scope:** `vllm.scheduler.step` (different from journey tracing's `vllm.scheduler`)
- **All events on one span:** Simplifies querying and reduces span overhead

**Comparison to Journey Tracing:**

| Aspect | Journey Tracing | Step Tracing |
|--------|----------------|--------------|
| Span per | Request | Scheduler lifetime |
| Span count | ~2 per request | 1 total |
| Span lifetime | Request duration | Process lifetime |
| Tracer scope | `vllm.api`, `vllm.scheduler` | `vllm.scheduler.step` |
| Events per span | ~5-10 | Thousands (sampled) |

</details>

### Integration with Other Observability

<details>
<summary><b>Using Step Tracing with Metrics and Logs</b></summary>

**Step Tracing + Prometheus Metrics:**

Step tracing complements Prometheus metrics:

```bash
vllm serve MODEL \
    --enable-metrics \
    --step-tracing-enabled \
    --otlp-traces-endpoint http://localhost:4317
```

**Use metrics for:**
- Continuous counters/gauges (requests/sec, latency P99)
- Alerting on thresholds
- Long-term trend analysis
- Dashboard displays

**Use step tracing for:**
- Root cause analysis ("why was P99 high at 3pm?")
- Correlating events across time
- Understanding batch composition during incidents

**Step Tracing + Journey Tracing:**

Both can run simultaneously:

```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --journey-tracing-sample-rate 0.1 \
    --step-tracing-enabled \
    --step-tracing-sample-rate 0.01 \
    --otlp-traces-endpoint http://localhost:4317
```

**Correlation strategy:**
- Journey traces have `scheduler.step` attributes on events
- Step traces have `step.id` on all events
- Match these values to see what was happening system-wide when a request had issues

**Example investigation:**
1. User reports slow request X
2. Find request X in journey traces → see `scheduler.step=1523` on events
3. Query step traces for `step.id=1523` → see `kv.usage_gpu_ratio=0.98` (cache pressure!)
4. Root cause: System was under heavy load during request X

</details>

---

## What's Next?

**For more help:**
- 💬 Ask in [vLLM Discord](https://discord.gg/vllm) #observability channel
- 🐛 File issues at [github.com/vllm-project/vllm/issues](https://github.com/vllm-project/vllm/issues)
- 📖 See [OTEL documentation](https://opentelemetry.io/docs/) for collector setup

**Related features:**
- `--enable-journey-tracing` - Per-request lifecycle tracing
- `--enable-metrics` - Prometheus metrics
- `--enable-mfu-metrics` - Model FLOPs utilization
- `--kv-cache-metrics` - KV cache residency metrics

---

**Happy tracing! 🔍✨**
