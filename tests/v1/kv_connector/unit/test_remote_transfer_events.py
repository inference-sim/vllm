# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for remote transfer events (Phase 3).

Tests verify the behavioral contracts for RemoteTransferInitiated and
RemoteTransferCompleted events as specified in the plan.

Behavioral contracts tested:
- C1b: Correct emission timing (NIXL: after transfer; P2P: bracket)
- C4: Every Initiated has exactly one Completed (no-disappear guarantee)
- C6b: Failures produce success=False
- C7: transfer_id uniquely pairs events
- Granularity: One event pair per request
"""

import pytest

from vllm.distributed.kv_events import (
    CONNECTOR_TYPE_NIXL,
    CONNECTOR_TYPE_P2P,
    RemoteTransferCompleted,
    RemoteTransferEventTracker,
    RemoteTransferInitiated,
)


# =============================================================================
# RemoteTransferEventTracker Mixin Tests
# =============================================================================


class MockConnectorWithTracker(RemoteTransferEventTracker):
    """Mock connector class that uses the RemoteTransferEventTracker mixin."""

    def __init__(self, rank: int = 0, connector_type: str = CONNECTOR_TYPE_NIXL):
        self._init_remote_transfer_events(rank=rank, connector_type=connector_type)


class TestRemoteTransferEventTracker:
    """Tests for RemoteTransferEventTracker mixin behavior."""

    def test_emit_initiated_creates_event(self):
        """Test that _emit_remote_transfer_initiated creates an Initiated event."""
        tracker = MockConnectorWithTracker(rank=0)

        transfer_id = tracker._emit_remote_transfer_initiated(
            request_id="req-001",
            source_rank=1,
            dest_rank=0,
            block_count=10,
        )

        events = tracker._take_remote_transfer_events()
        assert len(events) == 1
        assert isinstance(events[0], RemoteTransferInitiated)
        assert events[0].transfer_id == transfer_id
        assert events[0].request_id == "req-001"
        assert events[0].source_rank == 1
        assert events[0].dest_rank == 0
        assert events[0].block_count == 10

    def test_emit_completed_creates_event(self):
        """Test that _emit_remote_transfer_completed creates a Completed event."""
        tracker = MockConnectorWithTracker(rank=0)

        # First emit Initiated to store metadata
        transfer_id = tracker._emit_remote_transfer_initiated(
            request_id="req-001",
            source_rank=1,
            dest_rank=0,
            block_count=10,
        )

        # Then emit Completed
        tracker._emit_remote_transfer_completed(request_id="req-001", success=True)

        events = tracker._take_remote_transfer_events()
        assert len(events) == 2
        assert isinstance(events[1], RemoteTransferCompleted)
        assert events[1].transfer_id == transfer_id
        assert events[1].request_id == "req-001"
        assert events[1].success is True

    def test_c7_transfer_id_uniquely_pairs_events(self):
        """C7: transfer_id from Initiated is preserved in Completed."""
        tracker = MockConnectorWithTracker(rank=5)

        transfer_id = tracker._emit_remote_transfer_initiated(
            request_id="req-001",
            source_rank=1,
            dest_rank=5,
            block_count=20,
        )
        tracker._emit_remote_transfer_completed(request_id="req-001", success=True)

        events = tracker._take_remote_transfer_events()
        initiated = events[0]
        completed = events[1]

        # Both events must have the same transfer_id
        assert isinstance(initiated, RemoteTransferInitiated)
        assert isinstance(completed, RemoteTransferCompleted)
        assert initiated.transfer_id == completed.transfer_id == transfer_id

    def test_c4_completed_without_initiated_is_noop(self):
        """C4: _emit_remote_transfer_completed is idempotent (no-op if no Initiated)."""
        tracker = MockConnectorWithTracker(rank=0)

        # Emit Completed without Initiated - should be no-op
        tracker._emit_remote_transfer_completed(request_id="req-unknown", success=True)

        events = tracker._take_remote_transfer_events()
        assert len(events) == 0

    def test_c4_every_initiated_has_exactly_one_completed(self):
        """C4: Multiple Initiated events each get exactly one Completed."""
        tracker = MockConnectorWithTracker(rank=0)

        # Emit multiple Initiated events
        id1 = tracker._emit_remote_transfer_initiated(
            request_id="req-001", source_rank=1, dest_rank=0, block_count=10
        )
        id2 = tracker._emit_remote_transfer_initiated(
            request_id="req-002", source_rank=2, dest_rank=0, block_count=20
        )
        id3 = tracker._emit_remote_transfer_initiated(
            request_id="req-003", source_rank=3, dest_rank=0, block_count=30
        )

        # Emit Completed for each
        tracker._emit_remote_transfer_completed(request_id="req-001", success=True)
        tracker._emit_remote_transfer_completed(request_id="req-002", success=False)
        tracker._emit_remote_transfer_completed(request_id="req-003", success=True)

        events = tracker._take_remote_transfer_events()
        assert len(events) == 6  # 3 Initiated + 3 Completed

        # Verify pairing
        initiated_events = [e for e in events if isinstance(e, RemoteTransferInitiated)]
        completed_events = [e for e in events if isinstance(e, RemoteTransferCompleted)]

        assert len(initiated_events) == 3
        assert len(completed_events) == 3

        # Verify each transfer_id appears exactly once in Initiated and once in Completed
        initiated_ids = {e.transfer_id for e in initiated_events}
        completed_ids = {e.transfer_id for e in completed_events}
        assert initiated_ids == completed_ids == {id1, id2, id3}

    def test_c4_double_completed_is_noop(self):
        """C4: Calling Completed twice for same request is no-op on second call."""
        tracker = MockConnectorWithTracker(rank=0)

        tracker._emit_remote_transfer_initiated(
            request_id="req-001", source_rank=1, dest_rank=0, block_count=10
        )
        tracker._emit_remote_transfer_completed(request_id="req-001", success=True)

        # Second call should be no-op
        tracker._emit_remote_transfer_completed(request_id="req-001", success=False)

        events = tracker._take_remote_transfer_events()
        assert len(events) == 2  # Only 1 Initiated + 1 Completed

    def test_c6b_failure_produces_success_false(self):
        """C6b: Failed transfers result in success=False."""
        tracker = MockConnectorWithTracker(rank=0)

        tracker._emit_remote_transfer_initiated(
            request_id="req-fail", source_rank=1, dest_rank=0, block_count=10
        )
        tracker._emit_remote_transfer_completed(request_id="req-fail", success=False)

        events = tracker._take_remote_transfer_events()
        completed = events[1]
        assert isinstance(completed, RemoteTransferCompleted)
        assert completed.success is False

    def test_connector_type_preserved(self):
        """Test that connector_type is correctly set in events."""
        tracker_nixl = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_NIXL)
        tracker_p2p = MockConnectorWithTracker(rank=1, connector_type=CONNECTOR_TYPE_P2P)

        tracker_nixl._emit_remote_transfer_initiated(
            request_id="req-nixl", source_rank=1, dest_rank=0, block_count=10
        )
        tracker_p2p._emit_remote_transfer_initiated(
            request_id="req-p2p", source_rank=2, dest_rank=1, block_count=20
        )

        nixl_events = tracker_nixl._take_remote_transfer_events()
        p2p_events = tracker_p2p._take_remote_transfer_events()

        assert nixl_events[0].connector_type == CONNECTOR_TYPE_NIXL
        assert p2p_events[0].connector_type == CONNECTOR_TYPE_P2P

    def test_transfer_id_uniqueness_across_requests(self):
        """Test that transfer_ids are unique across different requests."""
        tracker = MockConnectorWithTracker(rank=0)

        ids = []
        for i in range(100):
            transfer_id = tracker._emit_remote_transfer_initiated(
                request_id=f"req-{i}",
                source_rank=1,
                dest_rank=0,
                block_count=i,
            )
            ids.append(transfer_id)

        # All IDs should be unique
        assert len(set(ids)) == 100

    def test_transfer_id_encodes_rank(self):
        """Test that transfer_id encodes the worker rank."""
        from vllm.distributed.kv_events import TransferIdGenerator

        tracker = MockConnectorWithTracker(rank=42)

        transfer_id = tracker._emit_remote_transfer_initiated(
            request_id="req-001",
            source_rank=1,
            dest_rank=42,
            block_count=10,
        )

        # Extract rank from transfer_id
        extracted_rank = TransferIdGenerator.extract_rank(transfer_id)
        assert extracted_rank == 42

    def test_take_events_clears_pending(self):
        """Test that _take_remote_transfer_events clears pending events."""
        tracker = MockConnectorWithTracker(rank=0)

        tracker._emit_remote_transfer_initiated(
            request_id="req-001", source_rank=1, dest_rank=0, block_count=10
        )

        # First take gets events
        events1 = tracker._take_remote_transfer_events()
        assert len(events1) == 1

        # Second take should be empty
        events2 = tracker._take_remote_transfer_events()
        assert len(events2) == 0

    def test_metadata_fields_preserved_in_completed(self):
        """Test that all metadata from Initiated is preserved in Completed."""
        tracker = MockConnectorWithTracker(rank=7)

        tracker._emit_remote_transfer_initiated(
            request_id="req-meta",
            source_rank=3,
            dest_rank=7,
            block_count=100,
        )
        tracker._emit_remote_transfer_completed(request_id="req-meta", success=True)

        events = tracker._take_remote_transfer_events()
        initiated = events[0]
        completed = events[1]

        # All fields from Initiated should be in Completed
        assert completed.request_id == initiated.request_id == "req-meta"
        assert completed.source_rank == initiated.source_rank == 3
        assert completed.dest_rank == initiated.dest_rank == 7
        assert completed.block_count == initiated.block_count == 100
        assert completed.connector_type == initiated.connector_type == CONNECTOR_TYPE_NIXL


# =============================================================================
# Event Container Tests
# =============================================================================


class TestEventContainers:
    """Tests for KVEvents container classes."""

    def test_nixl_kv_events_add_and_get(self):
        """Test NixlKVEvents can add and retrieve events."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlKVEvents,
        )

        events_container = NixlKVEvents(num_workers=1)

        event = RemoteTransferInitiated(
            transfer_id=123,
            request_id="req-001",
            connector_type=CONNECTOR_TYPE_NIXL,
            source_rank=0,
            dest_rank=1,
            block_count=10,
        )
        events_container.add_events([event])

        retrieved = events_container.get_all_events()
        assert len(retrieved) == 1
        assert retrieved[0] == event

    def test_p2p_nccl_kv_events_add_and_get(self):
        """Test P2pNcclKVEvents can add and retrieve events."""
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
            P2pNcclKVEvents,
        )

        events_container = P2pNcclKVEvents(num_workers=1)

        event = RemoteTransferInitiated(
            transfer_id=456,
            request_id="req-002",
            connector_type=CONNECTOR_TYPE_P2P,
            source_rank=0,
            dest_rank=1,
            block_count=20,
        )
        events_container.add_events([event])

        retrieved = events_container.get_all_events()
        assert len(retrieved) == 1
        assert retrieved[0] == event

    def test_event_container_clear(self):
        """Test that clear_events properly clears the container."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlKVEvents,
        )

        events_container = NixlKVEvents(num_workers=1)

        event = RemoteTransferInitiated(
            transfer_id=789,
            request_id="req-003",
            connector_type=CONNECTOR_TYPE_NIXL,
            source_rank=0,
            dest_rank=1,
            block_count=30,
        )
        events_container.add_events([event])
        assert len(events_container.get_all_events()) == 1

        events_container.clear_events()
        assert len(events_container.get_all_events()) == 0


# =============================================================================
# Integration-style Behavioral Tests (without actual NIXL/P2P dependencies)
# =============================================================================


class TestNixlConnectorEventBehavior:
    """Tests for NIXL connector event emission behavior using mock tracker."""

    def test_granularity_one_event_pair_per_request(self):
        """Test that even with multiple handles, only one event pair per request."""
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_NIXL)

        # Simulate NIXL behavior: emit Initiated once, even if multiple handles
        request_id = "req-multi-handle"
        tracker._emit_remote_transfer_initiated(
            request_id=request_id,
            source_rank=1,
            dest_rank=0,
            block_count=50,
        )

        # After all handles complete, emit one Completed
        tracker._emit_remote_transfer_completed(request_id=request_id, success=True)

        events = tracker._take_remote_transfer_events()
        initiated_count = sum(1 for e in events if isinstance(e, RemoteTransferInitiated))
        completed_count = sum(1 for e in events if isinstance(e, RemoteTransferCompleted))

        assert initiated_count == 1
        assert completed_count == 1


class TestP2pConnectorEventBehavior:
    """Tests for P2P connector event emission behavior using mock tracker."""

    def test_consumer_bracket_pattern(self):
        """Test P2P consumer bracket pattern: Initiated before, Completed after."""
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        # Simulate P2P consumer behavior
        request_id = "req-p2p-consumer"

        # Emit Initiated BEFORE recv loop
        tracker._emit_remote_transfer_initiated(
            request_id=request_id,
            source_rank=1,
            dest_rank=0,
            block_count=30,
        )

        # (Simulate recv operations here)

        # Emit Completed AFTER recv loop
        tracker._emit_remote_transfer_completed(request_id=request_id, success=True)

        events = tracker._take_remote_transfer_events()
        assert len(events) == 2
        assert isinstance(events[0], RemoteTransferInitiated)
        assert isinstance(events[1], RemoteTransferCompleted)

    def test_producer_first_layer_pattern(self):
        """Test P2P producer pattern: Initiated on first layer, Completed on wait."""
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        # Simulate P2P producer behavior
        requests = ["req-1", "req-2", "req-3"]
        initiated_requests: set[str] = set()

        # Simulate save_kv_layer being called multiple times (once per layer)
        for layer_idx in range(10):
            for request_id in requests:
                # Only emit Initiated on first layer
                if request_id not in initiated_requests:
                    initiated_requests.add(request_id)
                    tracker._emit_remote_transfer_initiated(
                        request_id=request_id,
                        source_rank=0,
                        dest_rank=1,
                        block_count=10,
                    )
                # (Simulate send operation here)

        # Simulate wait_for_save - emit Completed for all
        for request_id in initiated_requests:
            tracker._emit_remote_transfer_completed(request_id=request_id, success=True)

        events = tracker._take_remote_transfer_events()

        # Should have exactly 3 Initiated and 3 Completed (one per request)
        initiated_events = [e for e in events if isinstance(e, RemoteTransferInitiated)]
        completed_events = [e for e in events if isinstance(e, RemoteTransferCompleted)]

        assert len(initiated_events) == 3
        assert len(completed_events) == 3


# =============================================================================
# Exception Safety Tests (C4 No-Disappear Guarantee)
# =============================================================================


class TestExceptionSafety:
    """Tests verifying C4 No-Disappear Guarantee under exception conditions.

    These tests verify that RemoteTransferCompleted is always emitted after
    RemoteTransferInitiated, even when exceptions occur during transfer.
    """

    def test_consumer_exception_safety_pattern(self):
        """C4: Consumer must emit Completed even when recv loop raises exception.

        This test verifies the pattern used in P2P consumer (start_load_kv):
        - Initiated is emitted before recv loop
        - Completed MUST be emitted in finally block
        - success=False when exception occurs
        """
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        request_id = "req-exception"
        transfer_success = True

        # Emit Initiated BEFORE the operation
        tracker._emit_remote_transfer_initiated(
            request_id=request_id,
            source_rank=1,
            dest_rank=0,
            block_count=10,
        )

        # Simulate the try/finally pattern used in actual connector
        try:
            # Simulate recv operations that raise an exception
            raise RuntimeError("Simulated recv_tensor failure")
        except Exception:
            transfer_success = False
            # In actual code, we'd re-raise here
        finally:
            # C4: Completed MUST be emitted in finally block
            tracker._emit_remote_transfer_completed(
                request_id=request_id,
                success=transfer_success,
            )

        events = tracker._take_remote_transfer_events()

        # Verify both events are emitted despite exception
        assert len(events) == 2
        assert isinstance(events[0], RemoteTransferInitiated)
        assert isinstance(events[1], RemoteTransferCompleted)

        # Verify failure is recorded
        completed = events[1]
        assert completed.success is False
        assert completed.request_id == request_id

    def test_producer_exception_safety_pattern(self):
        """C4: Producer must emit Completed even when wait_for_sent raises.

        This test verifies the pattern used in P2P producer (wait_for_save):
        - Initiated is emitted on first layer
        - Completed MUST be emitted in finally block of wait_for_save
        - success=False when exception occurs
        """
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        # Track requests that had Initiated events emitted (like the connector does)
        producer_reqs_with_initiated: set[str] = set()

        # Simulate save_kv_layer being called - emit Initiated on first layer
        requests = ["req-1", "req-2", "req-3"]
        for request_id in requests:
            if request_id not in producer_reqs_with_initiated:
                producer_reqs_with_initiated.add(request_id)
                tracker._emit_remote_transfer_initiated(
                    request_id=request_id,
                    source_rank=0,
                    dest_rank=1,
                    block_count=10,
                )

        # Simulate wait_for_save with try/finally pattern
        success = True
        try:
            # Simulate wait_for_sent() failure
            raise RuntimeError("Simulated wait_for_sent failure")
        except Exception:
            success = False
            # In actual code, we'd re-raise here
        finally:
            # C4: Completed MUST be emitted in finally block for ALL requests
            for request_id in producer_reqs_with_initiated:
                tracker._emit_remote_transfer_completed(
                    request_id=request_id,
                    success=success,
                )
            producer_reqs_with_initiated.clear()

        events = tracker._take_remote_transfer_events()

        # Verify all events are emitted despite exception
        initiated_events = [e for e in events if isinstance(e, RemoteTransferInitiated)]
        completed_events = [e for e in events if isinstance(e, RemoteTransferCompleted)]

        assert len(initiated_events) == 3
        assert len(completed_events) == 3

        # Verify all failures are recorded
        for completed in completed_events:
            assert completed.success is False

    def test_partial_failure_in_multi_request_batch(self):
        """C4/C6b: Each request gets its own Completed even with mixed results.

        Tests that in a batch of requests, each gets a Completed event with
        the appropriate success status, even if some succeed and some fail.
        """
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_NIXL)

        # Emit Initiated for multiple requests
        request_ids = ["req-success-1", "req-fail", "req-success-2"]
        for request_id in request_ids:
            tracker._emit_remote_transfer_initiated(
                request_id=request_id,
                source_rank=1,
                dest_rank=0,
                block_count=10,
            )

        # Emit Completed with mixed success statuses
        tracker._emit_remote_transfer_completed("req-success-1", success=True)
        tracker._emit_remote_transfer_completed("req-fail", success=False)
        tracker._emit_remote_transfer_completed("req-success-2", success=True)

        events = tracker._take_remote_transfer_events()
        completed_events = [e for e in events if isinstance(e, RemoteTransferCompleted)]

        # Verify each request has correct success status
        success_by_req = {e.request_id: e.success for e in completed_events}
        assert success_by_req["req-success-1"] is True
        assert success_by_req["req-fail"] is False
        assert success_by_req["req-success-2"] is True

    def test_exception_during_initiated_does_not_require_completed(self):
        """C4: If Initiated is never emitted, no Completed is required.

        Verifies that if an exception occurs BEFORE Initiated is emitted,
        we don't need to emit a Completed (no orphaned Completed events).
        """
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        request_id = "req-early-fail"

        # Simulate exception BEFORE Initiated is emitted
        try:
            # Simulated early failure (e.g., metadata parsing error)
            raise ValueError("Simulated early failure")
        except ValueError:
            pass  # Exception handled, no Initiated was emitted

        # Since no Initiated was emitted, Completed should be no-op
        tracker._emit_remote_transfer_completed(request_id=request_id, success=False)

        events = tracker._take_remote_transfer_events()

        # No events should be emitted (Completed without Initiated is no-op)
        assert len(events) == 0

    def test_reraise_preserves_exception_after_completed(self):
        """Verify that re-raising exception still works after emitting Completed.

        This tests that the try/except/finally pattern correctly re-raises
        the original exception after emitting the Completed event.
        """
        tracker = MockConnectorWithTracker(rank=0, connector_type=CONNECTOR_TYPE_P2P)

        request_id = "req-reraise"
        original_exception = RuntimeError("Original error")

        tracker._emit_remote_transfer_initiated(
            request_id=request_id,
            source_rank=1,
            dest_rank=0,
            block_count=10,
        )

        caught_exception = None
        try:
            try:
                raise original_exception
            except Exception:
                # Emit completed before re-raising
                tracker._emit_remote_transfer_completed(
                    request_id=request_id,
                    success=False,
                )
                raise  # Re-raise the original exception
        except RuntimeError as e:
            caught_exception = e

        # Verify the original exception was re-raised
        assert caught_exception is original_exception

        # Verify Completed was still emitted
        events = tracker._take_remote_transfer_events()
        assert len(events) == 2
        assert isinstance(events[1], RemoteTransferCompleted)
        assert events[1].success is False
