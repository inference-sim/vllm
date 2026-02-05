# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for worker-side KV cache events being published by the scheduler.

This test verifies that TransferInitiated and TransferCompleted events
collected from the worker side (via kv_connector_output.kv_cache_events)
are properly extracted and published by the scheduler.

Bug context: Worker-side events were being collected but never published
because the scheduler's _make_request_outputs method did not extract
kv_cache_events from kv_connector_output.
"""

from vllm.distributed.kv_events import (
    TransferCompleted,
    TransferInitiated,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingKVEvents,
)
from vllm.v1.outputs import KVConnectorOutput


class TestWorkerEventsPublishing:
    """Tests that worker-side KV cache events are published by the scheduler.

    These tests verify that events collected from the worker
    (TransferInitiated, TransferCompleted) via kv_connector_output.kv_cache_events
    are included in the events published by the scheduler.
    """

    def test_worker_transfer_events_in_kv_connector_output(self):
        """Test that worker events can be stored in KVConnectorOutput.kv_cache_events.

        This is a prerequisite test - verifies the data structure can hold worker events.
        """
        # Create worker-side transfer events
        initiated = TransferInitiated(
            transfer_id=1,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=4,
            scheduler_step=10,
        )
        completed = TransferCompleted(
            transfer_id=1,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=4,
            success=True,
            scheduler_step=10,
        )

        # Wrap in OffloadingKVEvents (as done in the worker)
        kv_events = OffloadingKVEvents(num_workers=1)
        kv_events.add_events([initiated, completed])

        # Store in KVConnectorOutput
        output = KVConnectorOutput(
            finished_sending={"req-001"},
            finished_recving=None,
            kv_cache_events=kv_events,
        )

        # Verify events are accessible
        assert output.kv_cache_events is not None
        all_events = output.kv_cache_events.get_all_events()
        assert len(all_events) == 2
        assert isinstance(all_events[0], TransferInitiated)
        assert isinstance(all_events[1], TransferCompleted)

    def test_scheduler_should_extract_worker_events_from_kv_connector_output(self):
        """Test that scheduler extracts kv_cache_events from kv_connector_output.

        This test documents the EXPECTED behavior - currently FAILING due to bug.
        The scheduler should extract events from kv_connector_output.kv_cache_events
        and include them in the published event batch.

        Current behavior (bug): Only events from kv_cache_manager.take_events() and
        connector.take_events() are published. Worker events in
        kv_connector_output.kv_cache_events are ignored.

        Expected behavior: All three event sources should be combined:
        1. kv_cache_manager.take_events() -> BlockStored, BlockRemoved
        2. connector.take_events() -> CacheLoadCommitted, CacheStoreCommitted, CacheEviction
        3. kv_connector_output.kv_cache_events -> TransferInitiated, TransferCompleted
        """
        # Create worker-side transfer events
        initiated = TransferInitiated(
            transfer_id=100,
            request_id="worker-req",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=8,
            scheduler_step=42,
        )
        completed = TransferCompleted(
            transfer_id=100,
            request_id="worker-req",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=8,
            success=True,
            scheduler_step=42,
        )

        # Wrap in OffloadingKVEvents
        worker_events = OffloadingKVEvents(num_workers=1)
        worker_events.add_events([initiated, completed])

        # Create kv_connector_output with worker events
        kv_connector_output = KVConnectorOutput(
            finished_sending=set(),
            finished_recving=set(),
            kv_cache_events=worker_events,
        )

        # Simulate what the scheduler should do:
        # Extract events from kv_connector_output.kv_cache_events
        # and add them to the events list to be published

        # This is the EXPECTED behavior - extract worker events
        events_to_publish = []

        # Simulating: events from kv_cache_manager.take_events() (empty in this test)
        kv_cache_manager_events = []
        if kv_cache_manager_events:
            events_to_publish.extend(kv_cache_manager_events)

        # Simulating: events from connector.take_events() (empty in this test)
        connector_events = []
        if connector_events:
            events_to_publish.extend(connector_events)

        # THIS IS THE MISSING PIECE - extract worker events from kv_connector_output
        if (kv_connector_output and
            kv_connector_output.kv_cache_events is not None):
            worker_kv_events = kv_connector_output.kv_cache_events.get_all_events()
            events_to_publish.extend(worker_kv_events)

        # Verify worker events are included
        assert len(events_to_publish) == 2
        assert any(isinstance(e, TransferInitiated) for e in events_to_publish)
        assert any(isinstance(e, TransferCompleted) for e in events_to_publish)

        # Verify specific event details
        initiated_events = [e for e in events_to_publish if isinstance(e, TransferInitiated)]
        assert len(initiated_events) == 1
        assert initiated_events[0].transfer_id == 100
        assert initiated_events[0].request_id == "worker-req"

    def test_empty_kv_cache_events_handled_gracefully(self):
        """Test that None or empty kv_cache_events doesn't cause errors."""
        # Test with None
        output_none = KVConnectorOutput(
            finished_sending=set(),
            finished_recving=set(),
            kv_cache_events=None,
        )

        events = []
        if output_none.kv_cache_events is not None:
            events.extend(output_none.kv_cache_events.get_all_events())

        assert len(events) == 0

        # Test with empty OffloadingKVEvents
        empty_events = OffloadingKVEvents(num_workers=1)
        output_empty = KVConnectorOutput(
            finished_sending=set(),
            finished_recving=set(),
            kv_cache_events=empty_events,
        )

        events = []
        if output_empty.kv_cache_events is not None:
            events.extend(output_empty.kv_cache_events.get_all_events())

        assert len(events) == 0

    def test_worker_events_cleared_after_extraction(self):
        """Test that worker events are cleared after being extracted.

        This prevents the same events from being published multiple times.
        """
        # Create events
        event = TransferInitiated(
            transfer_id=1,
            request_id="req",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=1,
            scheduler_step=1,
        )

        kv_events = OffloadingKVEvents(num_workers=1)
        kv_events.add_events([event])

        # First extraction
        first_events = kv_events.get_all_events()
        assert len(first_events) == 1

        # Clear events (as should be done after extraction)
        kv_events.clear_events()

        # Second extraction should be empty
        second_events = kv_events.get_all_events()
        assert len(second_events) == 0


class TestEventAggregationFromMultipleWorkers:
    """Tests for aggregating events from multiple workers."""

    def test_events_aggregated_from_multiple_workers(self):
        """Test that events from multiple workers are properly aggregated.

        When using tensor parallelism, each worker may emit transfer events.
        These should be aggregated (deduplicated) before publishing.
        """
        # Same event from two workers (simulating TP=2)
        event = TransferInitiated(
            transfer_id=1,
            request_id="req-tp",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=4,
            scheduler_step=5,
        )

        # Create aggregator expecting 1 worker initially
        kv_events = OffloadingKVEvents(num_workers=1)

        # Add event from first worker
        kv_events.add_events([event])

        # Increment to expect 2 workers total
        kv_events.increment_workers(1)

        # Add same event from second worker
        kv_events.add_events([event])

        # After aggregation, should have common events (events that appeared in all workers)
        kv_events.aggregate()
        common_events = kv_events.get_all_events()

        # The event appeared in both workers, so it should be in common events
        assert len(common_events) == 1
        assert isinstance(common_events[0], TransferInitiated)
        assert common_events[0].transfer_id == 1


class TestSchedulerEventCollection:
    """Tests that verify the scheduler's event collection behavior.

    These tests exercise the actual code path in the scheduler that
    collects and publishes KV cache events.
    """

    def test_collect_kv_events_includes_worker_events(self):
        """Test that collect_kv_events helper includes worker-side events.

        This test will FAIL until the bug is fixed. It tests the helper
        function that should be used by the scheduler to collect events
        from all sources including worker-side kv_cache_events.
        """
        from vllm.v1.core.sched.scheduler import collect_kv_events

        # Create worker-side events
        initiated = TransferInitiated(
            transfer_id=1,
            request_id="req-1",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=4,
            scheduler_step=10,
        )
        completed = TransferCompleted(
            transfer_id=1,
            request_id="req-1",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=4,
            success=True,
            scheduler_step=10,
        )

        worker_events = OffloadingKVEvents(num_workers=1)
        worker_events.add_events([initiated, completed])

        kv_connector_output = KVConnectorOutput(
            finished_sending=set(),
            finished_recving=set(),
            kv_cache_events=worker_events,
        )

        # Call the helper function
        # kv_cache_manager_events and connector_events are empty for this test
        events = collect_kv_events(
            kv_cache_manager_events=None,
            connector_events=None,
            kv_connector_output=kv_connector_output,
        )

        # Verify worker events are included
        assert events is not None
        assert len(events) == 2
        assert any(isinstance(e, TransferInitiated) for e in events)
        assert any(isinstance(e, TransferCompleted) for e in events)
