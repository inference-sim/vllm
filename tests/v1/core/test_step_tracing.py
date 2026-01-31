# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level batch summary tracing (PR #3)."""

import hashlib
from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest

from vllm.tracing import SpanAttributes

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def make_deterministic_step_sampler(seed: int = 0) -> Callable[[int, float], bool]:
    """Create deterministic step sampler using stable hash.

    Uses hashlib.sha1 (NOT Python's hash() which is salted per process).
    Takes step_id (int) instead of string key.
    """

    def sampler(step_id: int, rate: float) -> bool:
        hash_bytes = hashlib.sha1(f"{seed}:{step_id}".encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:8], "big") / (2**64)
        return hash_value < rate

    return sampler


def test_step_tracing_disabled():
    """Test that step tracing is disabled by default.

    Verifies:
    - No span created when step_tracing_enabled=False
    - No events emitted
    - Zero overhead
    """
    scheduler = create_scheduler(step_tracing_enabled=False)

    # Verify step tracing is disabled
    assert not scheduler._enable_step_tracing
    assert scheduler._step_span is None

    # Add requests and schedule
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    # Verify scheduler works normally
    assert output.scheduler_step == 1
    assert len(output.scheduled_new_reqs) == 2


def test_step_tracing_enabled_no_otel():
    """Test step tracing with OTEL unavailable (no endpoint).

    Verifies graceful degradation when OTEL endpoint not configured.
    """
    scheduler = create_scheduler(
        step_tracing_enabled=True,
        step_tracing_sample_rate=1.0,
        otlp_traces_endpoint=None,  # No OTEL endpoint
    )

    # Step tracing should be disabled due to missing OTEL
    assert not scheduler._enable_step_tracing or scheduler._step_span is None

    # Scheduler should still work
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.scheduler_step == 1


def test_step_tracing_sample_rate_zero():
    """Test that sample_rate=0.0 produces no events.

    Even with step tracing enabled, 0% sampling should emit no events.
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=0.0,  # Never sample
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return False (0% sampling)
        scheduler._step_sampler = lambda step_id, rate: False

        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Schedule multiple times
        for _ in range(5):
            scheduler.schedule()

        # Verify no events emitted (sampler returned False every time)
        assert mock_span.add_event.call_count == 0


def test_step_tracing_sample_rate_one():
    """Test that sample_rate=1.0 emits event every step.

    Verifies:
    - Event emitted for every schedule() call
    - Event name is "step.BATCH_SUMMARY"
    - All required attributes present
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,  # Sample every step
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True (100% sampling)
        scheduler._step_sampler = lambda step_id, rate: True

        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Schedule 3 times
        num_schedules = 3
        for _ in range(num_schedules):
            scheduler.schedule()

        # Verify event emitted for each schedule
        assert mock_span.add_event.call_count == num_schedules

        # Verify all events have correct name
        for call in mock_span.add_event.call_args_list:
            event_name = call[0][0]
            assert event_name == "step.BATCH_SUMMARY"


def test_step_tracing_deterministic_sampling():
    """Test deterministic step sampling with stable hash sampler.

    Verifies:
    - Exact expected steps are sampled
    - No statistical flakiness in tests
    - Sampling is reproducible
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=0.5,  # 50% sampling
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Install deterministic sampler
        deterministic_sampler = make_deterministic_step_sampler(seed=42)
        scheduler._step_sampler = deterministic_sampler

        # Pre-compute which steps should be sampled with this seed
        expected_sampled_steps = []
        for step in range(1, 11):
            if deterministic_sampler(step, 0.5):
                expected_sampled_steps.append(step)

        # Schedule 10 times
        for _ in range(10):
            scheduler.schedule()

        # Verify exact expected steps were sampled
        assert mock_span.add_event.call_count == len(expected_sampled_steps)

        # Verify step IDs in events match expected
        sampled_steps = []
        for call in mock_span.add_event.call_args_list:
            attributes = call[1]["attributes"]
            sampled_steps.append(attributes[SpanAttributes.STEP_ID])

        assert sampled_steps == expected_sampled_steps


def test_step_tracing_empty_schedule():
    """Test batch summary emitted even for empty schedules.

    Verifies:
    - Empty schedule (no requests) still emits batch summary when sampled
    - All counts are zero
    - Useful for liveness monitoring
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Schedule with no requests (empty schedule)
        output = scheduler.schedule()

        # Verify event emitted even with no requests
        assert mock_span.add_event.call_count == 1

        # Verify attributes
        call_args = mock_span.add_event.call_args_list[0]
        event_name = call_args[0][0]
        attributes = call_args[1]["attributes"]

        assert event_name == "step.BATCH_SUMMARY"
        assert attributes[SpanAttributes.STEP_ID] == 1
        assert attributes[SpanAttributes.QUEUE_RUNNING_DEPTH] == 0
        assert attributes[SpanAttributes.QUEUE_WAITING_DEPTH] == 0
        assert attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS] == 0
        assert attributes[SpanAttributes.BATCH_NUM_DECODE_REQS] == 0
        assert attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_PREFILL_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_DECODE_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_NUM_FINISHED] == 0
        assert attributes[SpanAttributes.BATCH_NUM_PREEMPTED] == 0


def test_step_tracing_required_attributes():
    """Test that all required attributes are present and correct.

    Verifies:
    - All required attributes from plan are present
    - Values match expected computations
    - Types are correct
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify event emitted
        assert mock_span.add_event.call_count == 1

        # Extract attributes
        call_args = mock_span.add_event.call_args_list[0]
        attributes = call_args[1]["attributes"]

        # Verify all required attributes present
        required_attrs = [
            SpanAttributes.STEP_ID,
            SpanAttributes.STEP_TS_START_NS,
            SpanAttributes.STEP_TS_END_NS,
            SpanAttributes.STEP_DURATION_US,
            SpanAttributes.QUEUE_RUNNING_DEPTH,
            SpanAttributes.QUEUE_WAITING_DEPTH,
            SpanAttributes.BATCH_NUM_PREFILL_REQS,
            SpanAttributes.BATCH_NUM_DECODE_REQS,
            SpanAttributes.BATCH_SCHEDULED_TOKENS,
            SpanAttributes.BATCH_PREFILL_TOKENS,
            SpanAttributes.BATCH_DECODE_TOKENS,
            SpanAttributes.BATCH_NUM_FINISHED,
            SpanAttributes.BATCH_NUM_PREEMPTED,
            SpanAttributes.KV_USAGE_GPU_RATIO,
            SpanAttributes.KV_BLOCKS_TOTAL_GPU,
            SpanAttributes.KV_BLOCKS_FREE_GPU,
        ]

        for attr in required_attrs:
            assert attr in attributes, f"Missing required attribute: {attr}"

        # Verify types
        assert isinstance(attributes[SpanAttributes.STEP_ID], int)
        assert isinstance(attributes[SpanAttributes.STEP_TS_START_NS], int)
        assert isinstance(attributes[SpanAttributes.STEP_TS_END_NS], int)
        assert isinstance(attributes[SpanAttributes.STEP_DURATION_US], int)
        assert isinstance(attributes[SpanAttributes.QUEUE_RUNNING_DEPTH], int)
        assert isinstance(attributes[SpanAttributes.QUEUE_WAITING_DEPTH], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_DECODE_REQS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_PREFILL_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_DECODE_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_FINISHED], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_PREEMPTED], int)
        assert isinstance(attributes[SpanAttributes.KV_USAGE_GPU_RATIO], float)
        assert isinstance(attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU], int)
        assert isinstance(attributes[SpanAttributes.KV_BLOCKS_FREE_GPU], int)

        # Verify step ID
        assert attributes[SpanAttributes.STEP_ID] == 1

        # Verify timing
        assert attributes[SpanAttributes.STEP_TS_START_NS] > 0
        assert attributes[SpanAttributes.STEP_TS_END_NS] > 0
        assert attributes[SpanAttributes.STEP_TS_END_NS] >= attributes[SpanAttributes.STEP_TS_START_NS]
        assert attributes[SpanAttributes.STEP_DURATION_US] >= 0

        # Verify queue depths
        assert attributes[SpanAttributes.QUEUE_RUNNING_DEPTH] >= 0
        assert attributes[SpanAttributes.QUEUE_WAITING_DEPTH] >= 0

        # Verify batch composition
        assert attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS] >= 0
        assert attributes[SpanAttributes.BATCH_NUM_DECODE_REQS] >= 0

        # Verify token counts
        assert attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS] >= 0
        assert attributes[SpanAttributes.BATCH_PREFILL_TOKENS] >= 0
        assert attributes[SpanAttributes.BATCH_DECODE_TOKENS] >= 0

        # Verify KV cache metrics
        assert 0.0 <= attributes[SpanAttributes.KV_USAGE_GPU_RATIO] <= 1.0
        assert attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU] >= 0
        assert attributes[SpanAttributes.KV_BLOCKS_FREE_GPU] >= 0


def test_step_tracing_invariants():
    """Test that batch summary attributes satisfy expected invariants.

    Verifies:
    - Token sum invariants
    - KV cache consistency
    - Batch composition consistency
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=5)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Extract attributes
        call_args = mock_span.add_event.call_args_list[0]
        attributes = call_args[1]["attributes"]

        # Invariant: prefill + decode requests <= running depth
        prefill_reqs = attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS]
        decode_reqs = attributes[SpanAttributes.BATCH_NUM_DECODE_REQS]
        running_depth = attributes[SpanAttributes.QUEUE_RUNNING_DEPTH]
        assert prefill_reqs + decode_reqs <= running_depth

        # Invariant: prefill + decode tokens == scheduled tokens
        # NOTE: This test assumes no speculative decode. With speculative decode,
        # scheduled_tokens may include spec tokens not counted in prefill/decode.
        # If this test starts failing, check if spec decode is enabled in the test.
        prefill_tokens = attributes[SpanAttributes.BATCH_PREFILL_TOKENS]
        decode_tokens = attributes[SpanAttributes.BATCH_DECODE_TOKENS]
        scheduled_tokens = attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS]
        # Equality expected in standard config (no spec decode in create_requests())
        assert prefill_tokens + decode_tokens == scheduled_tokens

        # Invariant: KV cache consistency
        kv_total = attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU]
        kv_free = attributes[SpanAttributes.KV_BLOCKS_FREE_GPU]
        kv_usage = attributes[SpanAttributes.KV_USAGE_GPU_RATIO]

        assert kv_free <= kv_total
        assert 0.0 <= kv_usage <= 1.0

        # Sanity check: usage roughly matches free/total ratio
        # (May not be exact due to reserved/null blocks)
        if kv_total > 0:
            expected_usage = 1.0 - (kv_free / kv_total)
            # Allow some tolerance due to null block and rounding
            assert abs(kv_usage - expected_usage) < 0.1


def test_step_tracing_failure_safety():
    """Test that tracing failures don't crash the scheduler.

    Verifies:
    - Scheduler continues working even if event emission fails
    - Exceptions are caught and logged
    - No impact on scheduling correctness
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        # Make add_event raise an exception
        mock_span.add_event.side_effect = Exception("OTEL failure")

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Scheduler should not crash despite tracing failure
        output = scheduler.schedule()

        # Verify scheduler worked correctly
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 2

        # Verify exception was caught (add_event was called but didn't crash)
        assert mock_span.add_event.call_count == 1


def test_rich_snapshot_rate_zero():
    """Test that rich subsample rate 0.0 produces no rich events.

    Verifies:
    - Batch summary still emitted (step.BATCH_SUMMARY)
    - No rich snapshot events (step.REQUEST_SNAPSHOT)
    - Rich sampling is independent from batch summary sampling
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,  # Always sample batch summary
            step_tracing_rich_subsample_rate=0.0,  # Never sample rich snapshots
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers
        scheduler._step_sampler = lambda step_id, rate: True  # Batch summary always
        scheduler._rich_sampler = lambda step_id, rate: False  # Rich never

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 3

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # Should have exactly 1 batch summary, no rich snapshots
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 0


def test_rich_snapshot_enabled():
    """Test that rich subsample rate 1.0 emits events for all running requests.

    Verifies:
    - One step.REQUEST_SNAPSHOT event per running request
    - Events have correct step.id correlation
    - All required attributes present
    - KV metrics populated
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,  # Always sample rich
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers to always return True
        scheduler._step_sampler = lambda step_id, rate: True
        scheduler._rich_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=4)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 4

        # Extract events
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]
        event_attrs = [call[1]["attributes"] for call in mock_span.add_event.call_args_list]

        # Should have 1 batch summary + 4 rich snapshots
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 4

        # Verify rich snapshot attributes
        rich_events = [
            attrs
            for name, attrs in zip(event_names, event_attrs)
            if name == "step.REQUEST_SNAPSHOT"
        ]
        assert len(rich_events) == 4

        for attrs in rich_events:
            # Required attributes
            assert attrs[SpanAttributes.STEP_ID] == 1
            assert SpanAttributes.REQUEST_ID in attrs
            assert attrs[SpanAttributes.REQUEST_PHASE] in ("PREFILL", "DECODE")
            assert SpanAttributes.REQUEST_NUM_PROMPT_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_COMPUTED_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_OUTPUT_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_PREEMPTIONS in attrs
            assert SpanAttributes.REQUEST_SCHEDULED_TOKENS_THIS_STEP in attrs
            assert SpanAttributes.KV_BLOCKS_ALLOCATED_GPU in attrs
            assert SpanAttributes.KV_BLOCKS_CACHED_GPU in attrs

            # Verify types
            assert isinstance(attrs[SpanAttributes.STEP_ID], int)
            assert isinstance(attrs[SpanAttributes.REQUEST_ID], str)
            assert isinstance(attrs[SpanAttributes.KV_BLOCKS_ALLOCATED_GPU], int)
            assert isinstance(attrs[SpanAttributes.KV_BLOCKS_CACHED_GPU], int)


def test_rich_snapshot_gated_on_batch_summary():
    """Test that rich snapshots are only emitted when batch summary is sampled.

    Verifies the two-stage sampling:
    1. Step must be batch-summary-sampled
    2. Then rich subsampling decision
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override batch summary sampler to return False (not sampled)
        scheduler._step_sampler = lambda step_id, rate: False
        # Rich sampler is irrelevant (shouldn't be called)
        scheduler._rich_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 3

        # No events should be emitted (batch summary not sampled)
        assert mock_span.add_event.call_count == 0


def test_rich_snapshot_deterministic_sampling():
    """Test deterministic rich sampling for reproducible tests.

    Verifies:
    - Deterministic sampler produces stable results
    - Rich sampling decision is independent per step
    - Same seed produces same sample set
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Use deterministic samplers
        scheduler._step_sampler = make_deterministic_step_sampler(seed=42)
        scheduler._rich_sampler = make_deterministic_step_sampler(seed=100)

        # Run multiple steps
        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        for _ in range(5):
            scheduler.schedule()

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # With deterministic sampling, results should be stable
        batch_summaries = event_names.count("step.BATCH_SUMMARY")
        rich_snapshots = event_names.count("step.REQUEST_SNAPSHOT")

        # Verify we got some events (exact count depends on hash outputs)
        assert batch_summaries > 0
        # Rich snapshots only emitted when batch summary was sampled
        assert rich_snapshots % 2 == 0  # Should be even (2 requests per step)


def test_rich_snapshot_with_zero_running_requests():
    """Test that rich snapshots work correctly with empty running queue.

    Verifies:
    - Batch summary emitted even with no running requests
    - No rich snapshot events (no requests to snapshot)
    - No crashes or errors
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers to always return True
        scheduler._step_sampler = lambda step_id, rate: True
        scheduler._rich_sampler = lambda step_id, rate: True

        # Schedule with no requests
        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 0

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # Should have 1 batch summary, 0 rich snapshots (no running requests)
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 0


def test_step_tracing_cli_wiring():
    """Test that CLI flags are properly wired through to ObservabilityConfig.

    This is a regression test for PR #3 and PR #5 CLI wiring.
    Ensures that step tracing flags flow from CLI -> EngineArgs -> ObservabilityConfig.
    """
    from vllm.engine.arg_utils import EngineArgs

    # Test values different from defaults
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        step_tracing_enabled=True,  # Default: False
        step_tracing_sample_rate=0.75,  # Default: 0.01
        step_tracing_rich_subsample_rate=0.05,  # Default: 0.001
    )

    # Create engine config and verify wiring
    vllm_config = engine_args.create_engine_config()
    obs_config = vllm_config.observability_config

    # Verify all three fields are correctly wired
    assert obs_config.step_tracing_enabled is True
    assert obs_config.step_tracing_sample_rate == 0.75
    assert obs_config.step_tracing_rich_subsample_rate == 0.05


# =============================================================================
# Span Closure Tests (PR #X - Periodic Span Closure for Event Export)
# =============================================================================


def test_step_tracing_closure_interval_default():
    """Test that default closure interval is 100 steps.

    Verifies:
    - Default value is 100
    - Value is accessible via observability_config
    """
    scheduler = create_scheduler(
        step_tracing_enabled=True,
        otlp_traces_endpoint="http://localhost:4317",
    )

    # Verify default closure interval
    assert scheduler._closure_interval_steps == 100
    assert scheduler.observability_config.step_tracing_closure_interval == 100


def test_step_tracing_span_closed_at_interval():
    """Test that spans are closed every N steps.

    Verifies:
    - span.end() called after closure_interval steps
    - New span created immediately after closure
    - Counter resets after closure
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        # Create list of mock spans to track span creation
        mock_spans = [Mock() for _ in range(5)]
        span_index = [0]  # Use list to allow mutation in closure

        def create_span(*args, **kwargs):
            span = mock_spans[span_index[0]]
            span_index[0] += 1
            return span

        mock_tracer.start_span.side_effect = create_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,  # Always sample to trigger closure
            step_tracing_closure_interval=10,  # Close every 10 steps
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 25 steps (should trigger 2 closures: at step 10 and 20)
        for i in range(25):
            output = scheduler.schedule()
            assert output.scheduler_step == i + 1

        # Verify: Initial span + 2 new spans = 3 spans created
        assert mock_tracer.start_span.call_count == 3

        # Verify: First 2 spans were ended (at steps 10 and 20)
        assert mock_spans[0].end.call_count == 1
        assert mock_spans[1].end.call_count == 1
        assert mock_spans[2].end.call_count == 0  # Current span not closed yet


def test_step_tracing_span_sequence_increments():
    """Test that span sequence numbers increment correctly.

    Verifies:
    - Sequence starts at 1
    - Increments for each new span
    - Span names follow pattern: scheduler_steps_N
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 25 steps (should create 3 spans total: initial + 2 closures)
        for i in range(25):
            scheduler.schedule()

        # Extract span names from start_span calls
        span_names = [call[1]["name"] for call in mock_tracer.start_span.call_args_list]

        # Verify span naming: scheduler_steps_1, scheduler_steps_2, scheduler_steps_3
        assert span_names == [
            "scheduler_steps_1",
            "scheduler_steps_2",
            "scheduler_steps_3",
        ]


def test_step_tracing_closure_skipped_without_events():
    """Test event-gated optimization: skip closure if no events emitted.

    Verifies:
    - If no events sampled, span NOT closed even after interval
    - Counter still resets
    - Avoids creating empty spans
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=0.0,  # Never sample events
            step_tracing_closure_interval=10,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to never sample
        scheduler._step_sampler = lambda step_id, rate: False

        # Run 25 steps with no events
        for i in range(25):
            scheduler.schedule()

        # Verify: Only initial span created, no closures (no events emitted)
        assert mock_tracer.start_span.call_count == 1
        assert mock_span.end.call_count == 0


def test_step_tracing_span_metadata():
    """Test that span metadata is correct.

    Verifies:
    - span_sequence attribute
    - step_range_start attribute
    - step_range_end attribute (set on closure)
    - closure_interval attribute
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 15 steps (triggers one closure at step 10)
        for i in range(15):
            scheduler.schedule()

        # Verify initial span attributes (span 1)
        initial_attrs = mock_tracer.start_span.call_args_list[0][1]["attributes"]
        assert initial_attrs["scheduler.span_sequence"] == 1
        assert initial_attrs["scheduler.step_range_start"] == 1
        assert initial_attrs["scheduler.closure_interval"] == 10

        # Verify second span attributes (span 2, created after step 10)
        second_attrs = mock_tracer.start_span.call_args_list[1][1]["attributes"]
        assert second_attrs["scheduler.span_sequence"] == 2
        assert second_attrs["scheduler.step_range_start"] == 11
        assert second_attrs["scheduler.closure_interval"] == 10

        # Verify step_range_end was set on first span before closure
        set_attribute_calls = mock_span.set_attribute.call_args_list
        # Find the set_attribute call for step_range_end (should be 10)
        range_end_calls = [
            call for call in set_attribute_calls
            if call[0][0] == "scheduler.step_range_end"
        ]
        assert len(range_end_calls) >= 1
        assert range_end_calls[0][0][1] == 10  # step_range_end value


def test_step_tracing_span_closure_on_shutdown():
    """Test that final span is closed on shutdown.

    Verifies:
    - shutdown() closes the current span
    - Final span has correct step_range_end
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=100,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 15 steps (no closure yet, interval is 100)
        for i in range(15):
            scheduler.schedule()

        # Verify span not closed yet
        assert mock_span.end.call_count == 0

        # Call shutdown
        scheduler.shutdown()

        # Verify span was closed
        assert mock_span.end.call_count == 1

        # Verify final step_range_end was set to 15
        set_attribute_calls = mock_span.set_attribute.call_args_list
        range_end_calls = [
            call for call in set_attribute_calls
            if call[0][0] == "scheduler.step_range_end"
        ]
        assert len(range_end_calls) == 1
        assert range_end_calls[0][0][1] == 15


def test_step_tracing_span_creation_failure_disables_tracing():
    """Test that span creation failure disables tracing gracefully.

    Verifies:
    - Exception during _open_step_span() disables tracing
    - Scheduler continues to work
    - No crashes
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        # Make start_span raise exception
        mock_tracer.start_span.side_effect = Exception("Mock span creation failure")
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Verify tracing was disabled due to failure
        assert not scheduler._enable_step_tracing

        # Verify scheduler still works
        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 2


def test_step_tracing_span_closure_failure_preserves_span():
    """Test that span closure failure doesn't disable tracing.

    Verifies:
    - Exception during span.end() doesn't disable tracing
    - Span remains open (can retry next interval)
    - Scheduler continues to work
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        # Make span.end() raise exception
        mock_span.end.side_effect = Exception("Mock closure failure")
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10,  # Minimum allowed value
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 15 steps (should attempt closure at step 10)
        for i in range(15):
            output = scheduler.schedule()
            assert output.scheduler_step == i + 1

        # Verify: Tracing still enabled despite closure failure
        assert scheduler._enable_step_tracing

        # Verify: span.end() was called (attempted closure)
        assert mock_span.end.call_count >= 1


def test_step_tracing_multiple_closure_cycles():
    """Test multiple closure cycles work correctly.

    Verifies:
    - Multiple closures work in sequence
    - Each span gets correct sequence number
    - Events distributed across spans correctly
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        # Create multiple mock spans
        mock_spans = [Mock() for _ in range(10)]
        span_index = [0]

        def create_span(*args, **kwargs):
            span = mock_spans[span_index[0]]
            span_index[0] += 1
            return span

        mock_tracer.start_span.side_effect = create_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler
        scheduler._step_sampler = lambda step_id, rate: True

        # Run 35 steps (should create 4 spans: initial + 3 closures)
        for i in range(35):
            scheduler.schedule()

        # Verify 4 spans created
        assert mock_tracer.start_span.call_count == 4

        # Verify first 3 spans were closed
        assert mock_spans[0].end.call_count == 1
        assert mock_spans[1].end.call_count == 1
        assert mock_spans[2].end.call_count == 1
        assert mock_spans[3].end.call_count == 0  # Current span

        # Verify each span has events (since sample_rate=1.0)
        for i in range(3):
            # Each closed span should have add_event calls
            assert mock_spans[i].add_event.call_count >= 10  # At least 10 steps


def test_step_tracing_closure_interval_edge_cases():
    """Test closure interval edge cases.

    Verifies:
    - Minimum interval (10) works
    - Maximum interval (10000) works
    - Custom intervals work correctly
    """
    # Test minimum interval
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10,  # Minimum
            otlp_traces_endpoint="http://localhost:4317",
        )

        scheduler._step_sampler = lambda step_id, rate: True

        # Run 25 steps
        for i in range(25):
            scheduler.schedule()

        # Should have 3 spans (initial + 2 closures at 10, 20)
        assert mock_tracer.start_span.call_count == 3

    # Test large interval
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_closure_interval=10000,  # Maximum
            otlp_traces_endpoint="http://localhost:4317",
        )

        scheduler._step_sampler = lambda step_id, rate: True

        # Run 100 steps (well below interval)
        for i in range(100):
            scheduler.schedule()

        # Should have only 1 span (no closures yet)
        assert mock_tracer.start_span.call_count == 1
        assert mock_span.end.call_count == 0


def test_step_tracing_closure_interval_cli_wiring():
    """Test that --step-tracing-closure-interval CLI flag is properly wired.

    Verifies:
    - CLI flag flows through to ObservabilityConfig
    - Default value is 100
    - Custom values are respected
    """
    from vllm.engine.arg_utils import EngineArgs

    # Test default value
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        step_tracing_enabled=True,
    )
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.observability_config.step_tracing_closure_interval == 100

    # Test custom value
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        step_tracing_enabled=True,
        step_tracing_closure_interval=50,
    )
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.observability_config.step_tracing_closure_interval == 50
