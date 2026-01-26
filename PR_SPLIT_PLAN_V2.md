# Journey Tracing Dual-Stream Architecture - PR Split Plan V2

## Critical Discipline: No Incomplete Resources

**GLOBAL RULE**: If a PR introduces any resource that needs cleanup (span, dict entry, set membership, per-request state), that **same PR** must:

1. ✅ Terminate it on all exits
2. ✅ Clean it on all termination paths
3. ✅ Have tests proving it

**No "we'll add ABORTED later"** - if you create spans now, you close them now.

**No "we'll clean up in the next PR"** - if you create state now, you clean it now.

---

## Self-Consistent PR Chain (9 PRs)

Each PR is safe when merged alone, even if the feature is incomplete.

---

## PR #1: Engine - Scheduler Tracer Init (No Spans Yet)

**Branch**: `journey-tracing-01-scheduler-init`

**Goal**: Initialize tracer in scheduler without creating any per-request state.

### Changes

```python
# vllm/v1/core/sched/scheduler.py

# Add at top
try:
    from vllm.tracing import SpanAttributes
except Exception:
    SpanAttributes = None  # type: ignore

class Scheduler:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Initialize tracer for OTEL span creation
        self.tracer: Any | None = None
        if self._enable_journey_tracing:
            endpoint = self.observability_config.otlp_traces_endpoint
            if endpoint is not None:
                from vllm.tracing import init_tracer
                self.tracer = init_tracer("vllm.scheduler", endpoint)
```

### Safety Checklist

- ✅ No per-request state introduced
- ✅ No spans created
- ✅ No cleanup needed
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled (tracer stays None)

### Tests

1. `test_tracer_init_when_endpoint_set()` - Verify tracer created
2. `test_tracer_none_when_endpoint_not_set()` - Verify tracer=None
3. `test_no_crash_if_otel_missing()` - Verify graceful degradation

**Size**: ~25 lines, 3 tests

---

## PR #2: Engine - Core Span Create AND Close (Complete Lifecycle)

**Branch**: `journey-tracing-02-core-spans-lifecycle`

**Goal**: Create core spans on `add_request`, guarantee closure on all termination paths reachable in this PR.

### Changes

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Track active core spans (request_id → Span)
        # Always initialize to avoid AttributeError when journey tracing disabled
        self._core_spans: dict[str, Any] = {}

    def _create_core_span(self, request: Request) -> Any | None:
        """Create child span for engine-core journey tracing.

        Returns:
            Span object if tracer available, None otherwise
        """
        if not self.tracer:
            return None

        try:
            from vllm.tracing import SpanAttributes, extract_trace_context
            from opentelemetry.trace import SpanKind
        except ImportError:
            return None

        # Extract parent context from trace_headers
        parent_context = None
        if request.trace_headers:
            parent_context = extract_trace_context(request.trace_headers)

        # Create child span
        core_span = self.tracer.start_span(
            name="llm_core",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            start_time=time.time_ns(),
        )

        # Set span attributes
        core_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.request_id)

        return core_span

    def _end_core_span_and_cleanup(self, request: Request) -> None:
        """End core span and cleanup journey tracing state.

        CRITICAL: This is the centralized cleanup method that must be called
        for ALL request termination paths to prevent memory leaks.

        This PR only handles spans. Future PRs will extend this to clean
        other journey state (hiwater, dedup sets).
        """
        if not self._enable_journey_tracing:
            return

        request_id = request.request_id

        # End and remove core span
        core_span = self._core_spans.pop(request_id, None)
        if core_span and core_span.is_recording():
            core_span.end(end_time=time.time_ns())

    def add_request(self, request: Request) -> None:
        # ... existing code ...

        # NEW: Create child span
        if self._enable_journey_tracing:
            core_span = self._create_core_span(request)
            if core_span:
                self._core_spans[request.request_id] = core_span

    def finish_requests(
        self,
        request_ids: list[str],
        finished_status: RequestStatus,
    ) -> None:
        """Abort requests and cleanup spans."""
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None:
                continue

            # ... existing status update ...

            # NEW: End core span and cleanup (CRITICAL: in same PR as span creation)
            self._end_core_span_and_cleanup(request)

    def _update_from_output(self, ...):
        # ... existing code ...

        if stopped:
            # NEW: Ensure cleanup always happens, even if routed_experts fails
            try:
                routed_experts = self._get_routed_experts(request)
            except Exception:
                pass  # routed_experts is optional; failures must not prevent cleanup
            finally:
                # CRITICAL: Always cleanup, even on exceptions
                self._end_core_span_and_cleanup(request)

            # ... rest of existing code (_free_request called AFTER cleanup) ...
```

### Safety Checklist

- ✅ Spans created → Spans closed on all paths
  - ✅ finish_requests() → cleanup called
  - ✅ Natural completion (stopped=True) → cleanup called in finally block
  - ✅ Exception during routed_experts → cleanup called in finally block
- ✅ `_core_spans` dict cleaned on all termination paths
- ✅ Tests prove no leaks
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled (early return)

### Tests

1. `test_core_span_created_on_add_request()` - Verify span in dict
2. `test_core_span_closed_on_finish_requests()` - Verify span ended and removed
3. `test_core_span_closed_on_natural_completion()` - Verify cleanup in finally
4. `test_core_span_closed_on_exception()` - Verify cleanup even on exception
5. `test_no_span_leak_when_tracer_none()` - Verify dict stays empty
6. `test_parent_context_extraction()` - Verify trace_headers → parent context

**Size**: ~100 lines, 6 tests

---

## PR #3: Engine - Journey State (Hiwater + Dedup) WITH Cleanup

**Branch**: `journey-tracing-03-journey-state-cleanup`

**Goal**: Add journey progress tracking state, integrate cleanup into existing `_end_core_span_and_cleanup()`.

### Changes

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... existing init including _core_spans ...

        if self._enable_journey_tracing:
            # NEW: Track which requests have emitted FIRST_TOKEN (dedup)
            self._first_token_emitted: set[str] = set()
            # NEW: Prefill progress high-water marks (survives preemption)
            self._journey_prefill_hiwater: dict[str, int] = {}

    def _compute_progress_snapshot(self, request: Request) -> dict[str, Any]:
        """Compute progress snapshot for journey events.

        Handles preemption correctly by using high-water marks.
        """
        # ... implementation (existing logic from current code) ...
        return {
            "phase": phase,
            "prefill_done_tokens": prefill_done,
            "prefill_total_tokens": prefill_total,
            "decode_done_tokens": decode_done,
            "decode_max_tokens": decode_max,
        }

    def _end_core_span_and_cleanup(self, request: Request) -> None:
        """End core span and cleanup all journey tracing state.

        CRITICAL: Extended from PR #2 to also clean journey state.
        """
        if not self._enable_journey_tracing:
            return

        request_id = request.request_id

        # End and remove core span (from PR #2)
        core_span = self._core_spans.pop(request_id, None)
        if core_span and core_span.is_recording():
            core_span.end(end_time=time.time_ns())

        # NEW: Clean journey tracing state
        self._first_token_emitted.discard(request_id)
        self._journey_prefill_hiwater.pop(request_id, None)
```

### Safety Checklist

- ✅ State created → State cleaned in same function as PR #2
- ✅ All termination paths already call `_end_core_span_and_cleanup()` (from PR #2)
- ✅ Tests prove no set/dict growth
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled (sets/dicts not created)

### Tests

1. `test_journey_state_created()` - Verify state initialized
2. `test_journey_state_cleaned_on_finish()` - Verify cleanup on finish_requests
3. `test_journey_state_cleaned_on_completion()` - Verify cleanup on natural completion
4. `test_no_state_leak()` - Verify dicts/sets don't grow over many requests
5. `test_progress_snapshot_correct()` - Verify progress calculation

**Size**: ~50 lines, 5 tests

---

## PR #4: Engine - Emit Journey Events (Defensive, No New Resources)

**Branch**: `journey-tracing-04-journey-events-emit`

**Goal**: Emit events to core spans. No new resources, just additive event emission.

### Changes

```python
# vllm/v1/core/sched/scheduler.py

def _emit_journey_event(
    self,
    request: Request,
    event_type: RequestJourneyEventType,
    scheduler_step: int | None,
    span: Any | None = None,  # NEW parameter
    schedule_kind: ScheduleKind | None = None,
    finish_status: str | None = None,
) -> None:
    """Emit journey event to span (new) and buffer (legacy, parallel).

    DEFENSIVE: Must never break request processing if tracing fails.
    """
    if not self._enable_journey_tracing:
        return  # Near-zero overhead: single boolean check

    # NEW: Emit to span (parallel to legacy buffering)
    if span and span.is_recording() and SpanAttributes is not None:
        # Compute progress snapshot
        progress = self._compute_progress_snapshot(request)

        # Capture timestamps
        ts_monotonic = time.monotonic()
        ts_epoch_ns = time.time_ns()

        # Build event attributes
        attributes = {
            SpanAttributes.JOURNEY_EVENT_TYPE: event_type.name,
            SpanAttributes.JOURNEY_TS_MONOTONIC: ts_monotonic,
            SpanAttributes.JOURNEY_SCHEDULER_STEP: scheduler_step,
            SpanAttributes.JOURNEY_PHASE: progress["phase"],
            SpanAttributes.JOURNEY_PREFILL_DONE_TOKENS: progress["prefill_done_tokens"],
            SpanAttributes.JOURNEY_PREFILL_TOTAL_TOKENS: progress["prefill_total_tokens"],
            SpanAttributes.JOURNEY_DECODE_DONE_TOKENS: progress["decode_done_tokens"],
            SpanAttributes.JOURNEY_DECODE_MAX_TOKENS: progress["decode_max_tokens"],
            SpanAttributes.JOURNEY_NUM_PREEMPTIONS: request.num_preemptions,
        }

        # Add optional fields
        if schedule_kind is not None:
            attributes[SpanAttributes.JOURNEY_SCHEDULE_KIND] = schedule_kind.name
        if finish_status is not None:
            attributes[SpanAttributes.JOURNEY_FINISH_STATUS] = finish_status

        # DEFENSIVE: Ensure tracing failures never break request processing
        try:
            span.add_event(
                name=f"journey.{event_type.name}",
                attributes=attributes,
                timestamp=ts_epoch_ns,
            )
        except Exception:
            # Tracing must never break request processing
            logger.debug(
                "Failed to emit journey event %s for request %s",
                event_type.name,
                request.request_id,
            )

    # EXISTING: Legacy buffering (unchanged, parallel operation)
    # ... existing buffering code ...

# Update all call sites to pass span:

def add_request(self, request: Request) -> None:
    # ... existing span creation from PR #2 ...

    # NEW: Emit QUEUED event
    if self._enable_journey_tracing:
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.QUEUED,
            scheduler_step=self.scheduler_step_counter,
            span=core_span,
        )

def schedule(self, ...):
    # ... existing scheduling logic ...

    # NEW: Pass span to SCHEDULED event
    if self._enable_journey_tracing and schedule_kind is not None:
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.SCHEDULED,
            scheduler_step=curr_step,
            span=core_span,
            schedule_kind=schedule_kind,
        )

def _update_from_output(self, ...):
    # ... existing logic ...

    # NEW: Emit FIRST_TOKEN (deduped)
    if (
        request.status == RequestStatus.RUNNING
        and num_output_tokens == 1
        and request.request_id not in self._first_token_emitted
    ):
        self._first_token_emitted.add(request.request_id)
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.FIRST_TOKEN,
            scheduler_step=scheduler_output.scheduler_step,
            span=core_span,
        )

    # NEW: Emit FINISHED (before cleanup)
    if stopped:
        try:
            routed_experts = self._get_routed_experts(request)

            # NEW: Emit FINISHED event before cleanup
            if self._enable_journey_tracing:
                core_span = self._core_spans.get(request.request_id)
                try:
                    self._emit_journey_event(
                        request,
                        RequestJourneyEventType.FINISHED,
                        scheduler_step=scheduler_output.scheduler_step,
                        span=core_span,
                        finish_status=_map_finish_status(request.status),
                    )
                except Exception:
                    pass  # Defensive: tracing must never break completion
        except Exception:
            pass
        finally:
            # Cleanup from PR #2 (unchanged)
            self._end_core_span_and_cleanup(request)

# Similar updates for PREEMPTED event
```

### Safety Checklist

- ✅ No new resources created (just event emission)
- ✅ Defensive error handling (try/except around span.add_event)
- ✅ Safe when span is None or not recording
- ✅ Legacy buffering still works (parallel operation)
- ✅ Legacy tracing untouched

### Tests

1. `test_events_emitted_to_span()` - Verify events on span
2. `test_event_attributes_complete()` - Verify all attributes present
3. `test_defensive_error_handling()` - Verify exceptions caught
4. `test_no_events_when_span_none()` - Verify safe when no tracer
5. `test_legacy_buffering_still_works()` - Verify parallel operation
6. `test_first_token_deduped()` - Verify only one FIRST_TOKEN event

**Size**: ~120 lines, 6 tests

---

## PR #5: API - Add Request Metadata Fields (No Spans Yet)

**Branch**: `journey-tracing-05-api-metadata`

**Goal**: Add fields to RequestResponseMetadata without creating any resources.

### Changes

```python
# vllm/entrypoints/openai/engine/protocol.py

class RequestResponseMetadata(BaseModel):
    # Allow arbitrary types (OTEL Span) without validation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    final_usage_info: UsageInfo | None = None

    # NEW: API span tracking fields (no spans created yet)
    api_span: Any | None = None  # OTEL Span for API-level journey tracing
    # Timestamps for latency calculations (monotonic time)
    arrival_time: float | None = None  # time.monotonic() when span created
    first_response_time: float | None = None  # time.monotonic() when first output

# vllm/entrypoints/openai/engine/serving.py

class OpenAIServing:
    async def _get_is_tracing_enabled(self) -> bool:
        """Check if journey tracing is enabled.

        Caches result to avoid repeated async calls.
        """
        if not hasattr(self, '_cached_is_tracing_enabled'):
            self._cached_is_tracing_enabled = (
                await self.engine_client.is_tracing_enabled()
            )
        return self._cached_is_tracing_enabled
```

### Safety Checklist

- ✅ No per-request state introduced
- ✅ No spans created
- ✅ No cleanup needed
- ✅ Legacy tracing untouched
- ✅ Just field definitions (pure metadata)

### Tests

1. `test_metadata_fields_exist()` - Verify new fields
2. `test_metadata_arbitrary_types_allowed()` - Verify Pydantic config
3. `test_is_tracing_enabled_cached()` - Verify caching works

**Size**: ~15 lines, 3 tests

---

## PR #6: API - Parent Span WITH Full Closure (DEPARTED/ABORTED)

**Branch**: `journey-tracing-06-api-spans-full-lifecycle`

**Goal**: Create API parent spans AND ensure they're closed on all exit paths in the same PR.

**CRITICAL**: This PR must include DEPARTED and ABORTED events. No "we'll add them later".

### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

class OpenAIServingChat:
    async def _create_api_span(
        self, request_id: str, raw_request: Request | None
    ) -> Any | None:
        """Create parent span for API-level journey tracing.

        Returns:
            Span object if tracer available, None otherwise
        """
        try:
            from vllm.tracing import SpanAttributes, extract_trace_context
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
        except ImportError:
            return None

        # Get tracer from global provider (set by engine)
        try:
            tracer_provider = trace.get_tracer_provider()
            tracer = tracer_provider.get_tracer("vllm.api")
        except Exception:
            return None

        # Extract incoming trace context (if client provided)
        parent_context = None
        if raw_request:
            trace_headers = await self._get_trace_headers(raw_request.headers)
            if trace_headers:
                parent_context = extract_trace_context(trace_headers)

        # Create parent span
        api_span = tracer.start_span(
            name="llm_request",
            kind=SpanKind.SERVER,
            context=parent_context,
            start_time=time.time_ns(),
        )

        # Set basic attributes
        api_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request_id)

        return api_span

    def _safe_emit_departed_event(
        self,
        api_span: Any,
        request_metadata: RequestResponseMetadata,
    ) -> None:
        """Emit api.DEPARTED event and end span.

        CRITICAL: Idempotent - safe to call even if span already ended.
        """
        if not api_span or not api_span.is_recording():
            return

        try:
            from vllm.tracing import SpanAttributes

            # Emit DEPARTED event (minimal version for this PR)
            api_span.add_event(
                name="api.DEPARTED",
                attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
                timestamp=time.time_ns(),
            )

            # End span
            api_span.end(end_time=time.time_ns())
        except Exception:
            pass  # Defensive: tracing must never break response

    def _safe_emit_aborted_event(
        self,
        api_span: Any,
        error_message: str,
        reason: str | None = None,
    ) -> None:
        """Emit api.ABORTED event and end span.

        CRITICAL: Idempotent - safe to call even if span already ended.
        """
        if not api_span or not api_span.is_recording():
            return

        try:
            from vllm.tracing import SpanAttributes
            from opentelemetry.trace import Status, StatusCode

            # Set error status
            api_span.set_status(Status(StatusCode.ERROR, error_message))

            # Emit ABORTED event
            attributes: dict[str, Any] = {
                SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic(),
                "error": error_message,
            }
            if reason:
                attributes["reason"] = reason

            api_span.add_event(
                name="api.ABORTED",
                attributes=attributes,
                timestamp=time.time_ns(),
            )

            # End span
            api_span.end(end_time=time.time_ns())
        except Exception:
            pass  # Defensive

    async def create_chat_completion(self, ...):
        # ... existing request_id creation ...

        # NEW: Create API span
        api_span = None
        arrival_time = time.monotonic()
        is_tracing_enabled = await self._get_is_tracing_enabled()
        if is_tracing_enabled:
            api_span = await self._create_api_span(request_id, raw_request)

        # NEW: Initialize metadata with span
        request_metadata = RequestResponseMetadata(
            request_id=request_id,
            api_span=api_span,
            arrival_time=arrival_time,
        )

        # ... existing code ...

    async def chat_completion_stream_generator(self, ...):
        """Streaming generator with span closure on all paths."""
        try:
            # ... existing streaming logic ...

            # ... existing code ...
        except GenerationError as e:
            # NEW: Close span on generation error
            self._safe_emit_aborted_event(
                request_metadata.api_span,
                f"Generation error: {e}",
                "generation_error"
            )
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed, safe to return
        except Exception as e:
            # NEW: Close span on unexpected exception
            self._safe_emit_aborted_event(
                request_metadata.api_span, str(e), "exception"
            )
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed, safe to return

        # Send final done message
        yield "data: [DONE]\n\n"

        # NEW: Close span on success
        self._safe_emit_departed_event(request_metadata.api_span, request_metadata)

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        span_closed = False

        # CRITICAL: Outer try/finally ensures span cleanup on ANY exception path
        try:
            # Generator iteration
            try:
                async for res in result_generator:
                    final_res = res
            except asyncio.CancelledError:
                self._safe_emit_aborted_event(
                    request_metadata.api_span,
                    "Client disconnected",
                    "client_disconnect"
                )
                span_closed = True
                return self.create_error_response("Client disconnected")
            except ValueError as e:
                self._safe_emit_aborted_event(
                    request_metadata.api_span, str(e), "validation_error"
                )
                span_closed = True
                return self.create_error_response(e)

            # Response building
            assert final_res is not None

            # ... existing response building code ...

            # NEW: Close span on success
            self._safe_emit_departed_event(request_metadata.api_span, request_metadata)
            span_closed = True

            return response
        finally:
            # CRITICAL: Outer finally catches ALL other exceptions
            if not span_closed and request_metadata.api_span:
                self._safe_emit_aborted_event(
                    request_metadata.api_span,
                    "Unexpected error during request processing",
                )
```

### Safety Checklist

- ✅ Spans created → Spans closed on all paths
  - ✅ Streaming success → DEPARTED + span.end()
  - ✅ Streaming GenerationError → ABORTED + span.end()
  - ✅ Streaming exception → ABORTED + span.end()
  - ✅ Non-streaming success → DEPARTED + span.end()
  - ✅ Non-streaming CancelledError → ABORTED + span.end()
  - ✅ Non-streaming ValueError → ABORTED + span.end()
  - ✅ Non-streaming unexpected exception → ABORTED + span.end() in finally
- ✅ Idempotent span.end() (checks is_recording())
- ✅ Tests prove all paths close span
- ✅ Legacy tracing untouched

### Tests

1. `test_api_span_created()` - Verify span created when tracing enabled
2. `test_api_span_none_when_disabled()` - Verify None when disabled
3. `test_span_closed_on_streaming_success()` - Verify DEPARTED + end
4. `test_span_closed_on_streaming_error()` - Verify ABORTED + end
5. `test_span_closed_on_streaming_exception()` - Verify ABORTED + end
6. `test_span_closed_on_full_success()` - Verify DEPARTED + end
7. `test_span_closed_on_full_cancelled()` - Verify ABORTED + end
8. `test_span_closed_on_full_exception()` - Verify ABORTED + end in finally
9. `test_initialization_order()` - Verify engine init before API span

**Size**: ~150 lines, 9 tests

**CRITICAL**: This PR is larger because it includes full lifecycle. Cannot split into "create" and "close" PRs.

---

## PR #7: API↔Engine - Context Propagation

**Branch**: `journey-tracing-07-context-propagation`

**Goal**: Inject API span context into trace_headers for parent-child linkage.

### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

async def create_chat_completion(self, ...):
    # ... existing code up to trace_headers creation ...

    trace_headers = (
        None
        if raw_request is None
        else await self._get_trace_headers(raw_request.headers)
    )

    # NEW: Inject API span context into trace_headers
    if api_span:
        try:
            from opentelemetry import trace
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            # Set API span as current in context
            ctx = trace.set_span_in_context(api_span)

            # Inject into carrier
            carrier: dict[str, str] = {}
            propagator = TraceContextTextMapPropagator()
            propagator.inject(carrier, context=ctx)

            # Merge with existing trace_headers
            if trace_headers is None:
                trace_headers = carrier
            else:
                trace_headers = {**trace_headers, **carrier}
        except Exception as e:
            logger.debug("Failed to inject trace context: %s", e)

    # ... continue with engine.generate() call ...
```

### Safety Checklist

- ✅ No new resources created
- ✅ Just injects context into existing trace_headers dict
- ✅ Defensive error handling (failures don't break request)
- ✅ Legacy tracing untouched
- ✅ Core span already extracts context (from PR #2)

### Tests

1. `test_context_injection()` - Verify traceparent header created
2. `test_context_extraction_in_scheduler()` - Verify scheduler extracts correctly
3. `test_parent_child_same_trace_id()` - Verify trace linkage
4. `test_injection_failure_graceful()` - Verify failures don't break request

**Size**: ~25 lines, 4 tests

---

## PR #8: API - Emit Additional Events (ARRIVED/HANDOFF/FIRST_RESPONSE)

**Branch**: `journey-tracing-08-api-additional-events`

**Goal**: Add remaining API events. No new resources, just additive event emission.

### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

async def _create_api_span(self, ...):
    # ... existing span creation ...

    # NEW: Emit ARRIVED event immediately after creation
    try:
        from vllm.tracing import SpanAttributes

        api_span.add_event(
            name="api.ARRIVED",
            attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
            timestamp=time.time_ns(),
        )
    except Exception:
        pass  # Defensive

    return api_span

def _set_api_span_request_attributes(
    self,
    api_span: Any,
    model_name: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    """Set request metadata attributes on API span."""
    if not api_span or not api_span.is_recording():
        return

    try:
        from vllm.tracing import SpanAttributes

        api_span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, model_name)
        api_span.set_attribute(
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, len(prompt_token_ids)
        )
        if sampling_params.temperature is not None:
            api_span.set_attribute(
                SpanAttributes.GEN_AI_REQUEST_TEMPERATURE,
                sampling_params.temperature
            )
        # ... other sampling param attributes ...
    except Exception:
        pass  # Defensive

async def create_chat_completion(self, ...):
    # ... after prompt processing ...

    # NEW: Set request attributes on span
    if api_span and isinstance(sampling_params, SamplingParams):
        self._set_api_span_request_attributes(
            api_span,
            model_name,
            engine_prompt["prompt_token_ids"],
            sampling_params,
        )

    # ... after trace_headers and context injection (PR #7) ...

    # ... generate() call ...

    # NEW: Emit HANDOFF_TO_CORE after submitting to engine
    if request_metadata.api_span:
        try:
            from vllm.tracing import SpanAttributes
            request_metadata.api_span.add_event(
                name="api.HANDOFF_TO_CORE",
                attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
                timestamp=time.time_ns(),
            )
        except Exception:
            pass

async def chat_completion_stream_generator(self, ...):
    # ... in main loop, on first iteration ...

    if first_iteration:
        # NEW: Track first response time
        if request_metadata.first_response_time is None:
            request_metadata.first_response_time = time.monotonic()

            # Emit FIRST_RESPONSE_FROM_CORE event
            if request_metadata.api_span:
                try:
                    from vllm.tracing import SpanAttributes
                    request_metadata.api_span.add_event(
                        name="api.FIRST_RESPONSE_FROM_CORE",
                        attributes={
                            SpanAttributes.EVENT_TS_MONOTONIC:
                                request_metadata.first_response_time
                        },
                        timestamp=time.time_ns(),
                    )
                except Exception:
                    pass

# Similar for chat_completion_full_generator
async def chat_completion_full_generator(self, ...):
    # ... in generator iteration loop ...

    async for res in result_generator:
        # NEW: Track first response time
        if request_metadata.first_response_time is None:
            request_metadata.first_response_time = time.monotonic()

            if request_metadata.api_span:
                try:
                    from vllm.tracing import SpanAttributes
                    request_metadata.api_span.add_event(
                        name="api.FIRST_RESPONSE_FROM_CORE",
                        attributes={
                            SpanAttributes.EVENT_TS_MONOTONIC:
                                request_metadata.first_response_time
                        },
                        timestamp=time.time_ns(),
                    )
                except Exception:
                    pass

        final_res = res
```

### Safety Checklist

- ✅ No new resources created (just event emission)
- ✅ All events defensive (try/except)
- ✅ Span closure already handled (PR #6)
- ✅ Legacy tracing untouched

### Tests

1. `test_arrived_event_emitted()` - Verify ARRIVED on span creation
2. `test_handoff_event_emitted()` - Verify HANDOFF_TO_CORE
3. `test_first_response_event_emitted()` - Verify FIRST_RESPONSE_FROM_CORE
4. `test_first_response_only_once()` - Verify deduplication
5. `test_request_attributes_set()` - Verify span attributes

**Size**: ~80 lines, 5 tests

---

## PR #9: Cleanup - Remove Journey Buffering (Keep Legacy Tracing)

**Branch**: `journey-tracing-09-remove-buffering`

**Goal**: Remove journey events buffering now that spans work end-to-end. Do NOT touch OutputProcessor.do_tracing (legacy tracing).

### Changes

```python
# vllm/v1/core/sched/scheduler.py

def _emit_journey_event(self, ...):
    """Emit journey event to span only (buffering removed)."""
    if not self._enable_journey_tracing:
        return

    # Emit to span (from PR #4)
    if span and span.is_recording() and SpanAttributes is not None:
        # ... span emission code (unchanged) ...

    # REMOVED: Legacy buffering code
    # No more:
    # - self._journey_events_buffer_by_client
    # - RequestJourneyEvent creation
    # - Buffering logic

def _update_from_output(self, ...):
    # REMOVED: journey events flushing logic in this method
    # Just span emission (from PR #4) remains

# vllm/v1/engine/output_processor.py

class OutputProcessor:
    def process_outputs(self, ...):
        # REMOVED: journey_events_by_req collection logic
        # REMOVED: journey_events buffering code

        # Keep rest of method unchanged
        # DO NOT TOUCH: do_tracing() method (legacy tracing preserved)

# vllm/v1/core/sched/journey_events.py
# REMOVED: RequestJourneyEvent dataclass (no longer used)
# Keep: RequestJourneyEventType enum (still used)
# Keep: ScheduleKind enum (still used)
```

### Safety Checklist

- ✅ Spans work end-to-end (PRs #2-8)
- ✅ No functionality lost (spans replace buffering)
- ✅ Legacy tracing PRESERVED (do_tracing untouched)
- ✅ Tests verify spans still work

### Tests

1. `test_no_buffering_when_tracing()` - Verify buffer doesn't exist
2. `test_spans_still_work()` - Verify span emission unchanged
3. `test_end_to_end_journey_tracing()` - Verify complete flow
4. `test_legacy_tracing_preserved()` - Verify do_tracing still works

**Size**: ~150 lines removed, 4 tests

---

## Final Checklist (Applied to EVERY PR)

Add this to every PR description:

```markdown
### Resource Safety Checklist

- [ ] If this PR creates spans, it also ends them on all exits (success/error/cancel)
- [ ] If this PR introduces per-request state (dicts/sets), it also cleans them on all termination paths
- [ ] No buffering when tracer/exporter is absent
- [ ] Legacy tracing untouched
- [ ] Tests prove cleanup (no dict/set growth, spans ended)
- [ ] Defensive error handling (tracing never breaks requests)
- [ ] Zero overhead when disabled (early returns, no allocations)

### Termination Paths Covered

- [ ] Natural completion (stopped=True)
- [ ] Explicit abort (finish_requests)
- [ ] Exceptions during processing
- [ ] Client cancellation (if applicable)
- [ ] All paths call cleanup function
```

---

## PR Dependencies

```
PR #1 (Scheduler Tracer Init)
    ↓
PR #2 (Core Span + Cleanup) ← MUST include cleanup in same PR
    ↓
PR #3 (Journey State + Cleanup) ← extends PR #2 cleanup
    ↓
PR #4 (Core Event Emit) ← no new resources, safe
    ↓
PR #5 (API Metadata) ← independent, can be parallel
    ↓
PR #6 (API Span + DEPARTED/ABORTED) ← MUST include all closure paths
    ↓
PR #7 (Context Propagation) ← depends on #2 and #6
    ↓
PR #8 (API Additional Events) ← no new resources, safe
    ↓
PR #9 (Remove Buffering) ← depends on all above working
```

## Timeline Estimate

- **PR #1**: 0.5 day (tiny, foundational)
- **PR #2**: 2 days (most critical, includes cleanup)
- **PR #3**: 1 day (extends cleanup)
- **PR #4**: 1.5 days (event emission)
- **PR #5**: 0.5 day (tiny, just fields)
- **PR #6**: 2 days (critical, includes all closure)
- **PR #7**: 1 day (context propagation)
- **PR #8**: 1 day (additional events)
- **PR #9**: 1 day (removal)

**Total**: ~2 weeks

---

## Benefits of This Disciplined Approach

1. **Every PR is Safe**: No "fix it later" - resources are created and cleaned in same PR
2. **Independent Merging**: Any PR can be merged without waiting for later PRs
3. **Easy Rollback**: Any PR can be reverted without breaking others
4. **Clear Verification**: Each PR has explicit checklist proving safety
5. **No Technical Debt**: No temporary hacks or incomplete implementations
6. **Reviewer Confidence**: Clear evidence that no leaks are introduced

---

## Key Differences from V1

1. **PR #2 now includes cleanup** - span creation and cleanup in same PR
2. **PR #6 now includes DEPARTED/ABORTED** - span creation and all closure paths in same PR
3. **Explicit termination path coverage** - every PR lists all paths that need cleanup
4. **Stricter checklist** - every PR must prove no dangling resources
5. **Fewer PRs** (9 vs 11) - combined creation+cleanup into single PRs

This ensures **no PR depends on a future PR to be safe**.
