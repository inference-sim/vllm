# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for worker-side KV cache events publishing.

These tests verify the behavioral contract:
  "When the scheduler processes model_runner_output containing
   kv_connector_output.kv_cache_events, those events are published
   via kv_event_publisher.publish()"

This contract ensures that worker-side transfer events (TransferInitiated,
TransferCompleted, RemoteTransferInitiated, RemoteTransferCompleted) flow
from the worker through the scheduler to the event publisher.
"""

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_events import (
    TransferCompleted,
    TransferInitiated,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingKVEvents,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


@pytest.fixture
def worker_events():
    """Create sample worker-side transfer events."""
    initiated = TransferInitiated(
        transfer_id=42,
        request_id="req-test",
        source_medium="GPU",
        dest_medium="CPU",
        block_count=8,
        scheduler_step=100,
    )
    completed = TransferCompleted(
        transfer_id=42,
        request_id="req-test",
        source_medium="GPU",
        dest_medium="CPU",
        block_count=8,
        success=True,
        scheduler_step=100,
    )
    return [initiated, completed]


@pytest.fixture
def kv_connector_output_with_events(worker_events):
    """Create KVConnectorOutput containing worker events."""
    kv_events = OffloadingKVEvents(num_workers=1)
    kv_events.add_events(worker_events)
    return KVConnectorOutput(kv_cache_events=kv_events)


@pytest.fixture
def mock_kv_event_publisher():
    """Create a mock publisher that captures published batches."""
    publisher = MagicMock()
    publisher.published_batches = []

    def capture_publish(batch):
        publisher.published_batches.append(batch)

    publisher.publish.side_effect = capture_publish
    return publisher


@pytest.fixture
def empty_scheduler_output():
    """Create minimal SchedulerOutput for testing."""
    return SchedulerOutput.make_empty()


def make_model_runner_output(kv_connector_output):
    """Create ModelRunnerOutput with the given kv_connector_output."""
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )


class TestSchedulerPublishesWorkerEvents:
    """Behavioral tests for scheduler publishing worker-side KV events.

    Contract: Worker events in model_runner_output.kv_connector_output.kv_cache_events
    are published via kv_event_publisher when the scheduler processes output.
    """

    def test_worker_events_are_published(
        self,
        mock_kv_event_publisher,
        kv_connector_output_with_events,
        empty_scheduler_output,
    ):
        """Worker events in kv_connector_output are published by scheduler.

        This test creates a scheduler with a mock publisher,
        then verifies that worker events flow through to publication.
        """
        from tests.v1.core.utils import create_scheduler

        # Given: A scheduler with mocked kv_event_publisher
        scheduler = create_scheduler(enable_prefix_caching=True)
        scheduler.kv_event_publisher = mock_kv_event_publisher

        # When: Scheduler processes output containing worker events
        model_runner_output = make_model_runner_output(
            kv_connector_output_with_events
        )
        scheduler.update_from_output(empty_scheduler_output, model_runner_output)

        # Then: Publisher received batch containing worker events
        assert mock_kv_event_publisher.publish.called, (
            "kv_event_publisher.publish() should have been called"
        )

        # Find all published events
        all_published_events = []
        for batch in mock_kv_event_publisher.published_batches:
            all_published_events.extend(batch.events)

        # Verify worker events are in published events
        transfer_initiated_found = any(
            isinstance(e, TransferInitiated) and e.transfer_id == 42
            for e in all_published_events
        )
        transfer_completed_found = any(
            isinstance(e, TransferCompleted) and e.transfer_id == 42
            for e in all_published_events
        )

        assert transfer_initiated_found, (
            "TransferInitiated event should be published"
        )
        assert transfer_completed_found, (
            "TransferCompleted event should be published"
        )

    def test_no_crash_when_kv_connector_output_is_none(
        self,
        mock_kv_event_publisher,
        empty_scheduler_output,
    ):
        """No crash when kv_connector_output is None."""
        from tests.v1.core.utils import create_scheduler

        # Given: A scheduler with mocked publisher
        scheduler = create_scheduler(enable_prefix_caching=True)
        scheduler.kv_event_publisher = mock_kv_event_publisher

        # When: Scheduler processes output with no kv_connector_output
        model_runner_output = make_model_runner_output(None)
        scheduler.update_from_output(empty_scheduler_output, model_runner_output)

        # Then: No crash occurred (implicit)

    def test_no_crash_when_kv_cache_events_is_none(
        self,
        mock_kv_event_publisher,
        empty_scheduler_output,
    ):
        """No crash when kv_connector_output.kv_cache_events is None."""
        from tests.v1.core.utils import create_scheduler

        # Given: A scheduler and kv_connector_output with no events
        scheduler = create_scheduler(enable_prefix_caching=True)
        scheduler.kv_event_publisher = mock_kv_event_publisher

        kv_connector_output = KVConnectorOutput(kv_cache_events=None)
        model_runner_output = make_model_runner_output(kv_connector_output)

        # When: Scheduler processes output
        scheduler.update_from_output(empty_scheduler_output, model_runner_output)

        # Then: No crash occurred (implicit)

    def test_events_not_published_twice(
        self,
        mock_kv_event_publisher,
        kv_connector_output_with_events,
        empty_scheduler_output,
    ):
        """Worker events are cleared after extraction (no duplicates)."""
        from tests.v1.core.utils import create_scheduler

        # Given: A scheduler with worker events
        scheduler = create_scheduler(enable_prefix_caching=True)
        scheduler.kv_event_publisher = mock_kv_event_publisher

        model_runner_output = make_model_runner_output(
            kv_connector_output_with_events
        )

        # When: Scheduler processes output twice with same kv_connector_output
        for _ in range(2):
            scheduler.update_from_output(
                empty_scheduler_output, model_runner_output
            )

        # Then: Worker events appear only once (not duplicated)
        all_transfer_initiated = [
            e for batch in mock_kv_event_publisher.published_batches
            for e in batch.events
            if isinstance(e, TransferInitiated) and e.transfer_id == 42
        ]

        assert len(all_transfer_initiated) == 1, (
            f"TransferInitiated should appear exactly once, "
            f"found {len(all_transfer_initiated)}"
        )
