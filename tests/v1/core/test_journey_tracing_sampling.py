# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for journey tracing sampling (PR #2).

Tests verify:
1. Engine compliance with sampling header
2. End-to-end atomicity (core spans exist or not based on header)
3. Backward compatibility

Focus: Engine-side span existence (atomicity via span presence).
API-side sampling logic is tested separately in API integration tests.
"""
import pytest

from vllm.tracing import VLLM_JOURNEY_SAMPLED_HEADER

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_journey_tracing_sample_rate_zero_no_header():
    """Verify no header means no core spans (simulates sample_rate=0.0)."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # Simulate unsampled request (no sampling header)
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {}  # No sampling header

    scheduler.add_request(request)

    # Verify no core span created (engine skips due to missing header)
    assert request.request_id not in scheduler._core_spans


def test_journey_tracing_sample_rate_one_with_header():
    """Verify header present means core spans created (simulates sample_rate=1.0)."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # Simulate sampled request (header present with value "1")
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    scheduler.add_request(request)

    # Verify core span created
    assert request.request_id in scheduler._core_spans


def test_journey_tracing_missing_header():
    """Verify missing header prevents core span creation (conservative behavior)."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",
    )

    # Request without sampling header
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {"other": "header"}  # Other headers but no sampling header

    scheduler.add_request(request)

    # Verify no core span created (conservative: missing header = not sampled)
    assert request.request_id not in scheduler._core_spans


def test_journey_tracing_header_present():
    """Verify header with value "1" creates core span."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # Request with sampling header set to "1"
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    scheduler.add_request(request)

    # Verify core span created
    assert request.request_id in scheduler._core_spans


def test_journey_tracing_none_trace_headers():
    """Verify None trace_headers prevents core span creation."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",
    )

    # Request with None trace_headers
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = None

    scheduler.add_request(request)

    # Verify no core span (conservative: None = not sampled)
    assert request.request_id not in scheduler._core_spans


def test_journey_tracing_wrong_header_value():
    """Verify header with wrong value prevents core span creation."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",
    )

    # Request with sampling header but wrong value
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "0"}  # Wrong value

    scheduler.add_request(request)

    # Verify no core span (conservative: wrong value = not sampled)
    assert request.request_id not in scheduler._core_spans


def test_journey_tracing_multiple_requests_mixed():
    """Verify mixed sampling: some sampled, some not (atomicity)."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # Create 3 requests: sampled, not sampled, sampled
    requests = create_requests(num_requests=3)

    # Request 0: sampled (header = "1")
    requests[0].trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    # Request 1: not sampled (no header)
    requests[1].trace_headers = {}

    # Request 2: sampled (header = "1")
    requests[2].trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Verify spans created only for sampled requests (atomicity check)
    assert requests[0].request_id in scheduler._core_spans
    assert requests[1].request_id not in scheduler._core_spans
    assert requests[2].request_id in scheduler._core_spans


def test_journey_tracing_disabled_master_switch():
    """Verify enable_journey_tracing=False disables all tracing regardless of header."""
    scheduler = create_scheduler(
        enable_journey_tracing=False,  # Master switch OFF
    )

    # Request with sampling header (but tracing disabled)
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    scheduler.add_request(request)

    # Verify no core span (master switch overrides header)
    assert request.request_id not in scheduler._core_spans


def test_journey_tracing_atomicity_span_existence():
    """Verify atomicity: span existence is all-or-nothing based on header."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # Test 1: No header → no span
    request1 = create_requests(num_requests=1)[0]
    request1.trace_headers = {}
    scheduler.add_request(request1)
    assert request1.request_id not in scheduler._core_spans

    # Test 2: Header present → span exists
    request2 = create_requests(num_requests=1)[0]
    request2.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}
    scheduler.add_request(request2)
    assert request2.request_id in scheduler._core_spans


def test_journey_tracing_backward_compatibility_default():
    """Verify default behavior is backward compatible."""
    # Create scheduler with default sample_rate (should be 1.0)
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        otlp_traces_endpoint="http://localhost:4317",  # Enable tracer initialization
    )

    # With header set (simulating API with sample_rate=1.0 behavior)
    requests = create_requests(num_requests=1)
    request = requests[0]
    request.trace_headers = {VLLM_JOURNEY_SAMPLED_HEADER: "1"}

    scheduler.add_request(request)

    # Should behave identically to before (span created)
    assert request.request_id in scheduler._core_spans
