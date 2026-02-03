# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV events subscriber.

Phase 4 of KV Offloading Tracing: Production-ready subscriber.
"""

import json
import os
import signal
import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_events import (
    BlockStored,
    CacheEviction,
    CacheLoadCommitted,
    KVEventBatch,
    TransferCompleted,
    TransferInitiated,
)

# Import the subscriber module
from examples.online_serving.kv_events_subscriber import (
    DEFAULT_REPLAY_ADDR,
    DEFAULT_SUB_ADDR,
    DEFAULT_TOPIC,
    ENV_OUTPUT_FILE,
    ENV_REPLAY_ADDR,
    ENV_SUB_ADDR,
    ENV_TOPIC,
    KVEventsSubscriber,
)


# =============================================================================
# P2: Environment Variable Configuration Tests
# =============================================================================


class TestEnvironmentVariables:
    """Tests for environment variable configuration (P2)."""

    def test_default_addresses(self):
        """Test that default addresses are used when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            sub_addr = os.environ.get(ENV_SUB_ADDR, DEFAULT_SUB_ADDR)
            replay_addr = os.environ.get(ENV_REPLAY_ADDR, DEFAULT_REPLAY_ADDR)
            output_file = os.environ.get(ENV_OUTPUT_FILE)
            topic = os.environ.get(ENV_TOPIC, DEFAULT_TOPIC)

            assert sub_addr == "tcp://localhost:5557"
            assert replay_addr == "tcp://localhost:5558"
            assert output_file is None
            assert topic == ""

    def test_custom_addresses_from_env(self):
        """Test that custom addresses are read from env vars."""
        custom_env = {
            ENV_SUB_ADDR: "tcp://custom-host:6000",
            ENV_REPLAY_ADDR: "tcp://custom-host:6001",
            ENV_OUTPUT_FILE: "/tmp/events.jsonl",
            ENV_TOPIC: "kv-events",
        }
        with patch.dict(os.environ, custom_env, clear=True):
            sub_addr = os.environ.get(ENV_SUB_ADDR, DEFAULT_SUB_ADDR)
            replay_addr = os.environ.get(ENV_REPLAY_ADDR, DEFAULT_REPLAY_ADDR)
            output_file = os.environ.get(ENV_OUTPUT_FILE)
            topic = os.environ.get(ENV_TOPIC, DEFAULT_TOPIC)

            assert sub_addr == "tcp://custom-host:6000"
            assert replay_addr == "tcp://custom-host:6001"
            assert output_file == "/tmp/events.jsonl"
            assert topic == "kv-events"

    def test_env_var_names_correct(self):
        """Test that env var names match the documented names."""
        assert ENV_SUB_ADDR == "VLLM_KV_EVENTS_SUB_ADDR"
        assert ENV_REPLAY_ADDR == "VLLM_KV_EVENTS_REPLAY_ADDR"
        assert ENV_OUTPUT_FILE == "VLLM_KV_EVENTS_OUTPUT_FILE"
        assert ENV_TOPIC == "VLLM_KV_EVENTS_TOPIC"


# =============================================================================
# P1: File Output Tests
# =============================================================================


class TestFileOutput:
    """Tests for JSONL file output (P1)."""

    def test_jsonl_encoding(self):
        """Test that event batches are encoded as valid JSONL."""
        encoder = msgspec.json.Encoder()

        event = TransferInitiated(
            transfer_id=12345,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=10,
            scheduler_step=42,
        )
        batch = KVEventBatch(ts=1234567890.123, events=[event], scheduler_step=42)

        json_bytes = encoder.encode(batch)
        json_str = json_bytes.decode("utf-8")

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert "ts" in parsed or isinstance(parsed, list)

    def test_write_to_temp_file(self):
        """Test writing events to a temporary file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            temp_path = f.name

        try:
            subscriber = KVEventsSubscriber(
                sub_addr="tcp://localhost:5557",
                replay_addr="tcp://localhost:5558",
                output_file=temp_path,
            )
            subscriber._open_output()

            # Create and write a batch
            event = CacheLoadCommitted(
                request_id="req-test",
                medium="CPU",
                block_count=5,
                scheduler_step=1,
            )
            batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)
            subscriber._write_event_batch(batch)

            subscriber._close_output()

            # Verify file contents
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert content.endswith("\n")
                # Verify it's valid JSON
                line = content.strip()
                parsed = json.loads(line)
                assert parsed[0] == 1234567890.0  # ts field (array_like format)
        finally:
            os.unlink(temp_path)

    def test_append_mode(self):
        """Test that file output uses append mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            temp_path = f.name
            f.write('{"existing": "data"}\n')

        try:
            subscriber = KVEventsSubscriber(
                sub_addr="tcp://localhost:5557",
                replay_addr="tcp://localhost:5558",
                output_file=temp_path,
            )
            subscriber._open_output()

            event = CacheEviction(
                medium="CPU",
                blocks_evicted=3,
                eviction_reason="lru",
                scheduler_step=1,
            )
            batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)
            subscriber._write_event_batch(batch)
            subscriber._close_output()

            # Verify both lines exist
            with open(temp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert lines[0] == '{"existing": "data"}\n'
        finally:
            os.unlink(temp_path)

    def test_stdout_when_no_file(self):
        """Test that stdout is used when no file is specified."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
            output_file=None,
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            subscriber._open_output()
            event = TransferCompleted(
                transfer_id=1,
                request_id="req-1",
                source_medium="GPU",
                dest_medium="CPU",
                block_count=1,
                success=True,
                scheduler_step=1,
            )
            batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)
            subscriber._write_event_batch(batch)

            output = sys.stdout.getvalue()
            assert len(output) > 0
            assert output.endswith("\n")
        finally:
            sys.stdout = old_stdout


# =============================================================================
# P3: Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown (P3)."""

    def test_shutdown_sets_running_false(self):
        """Test that shutdown() sets _running to False."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )
        subscriber._running = True
        subscriber.shutdown()
        assert subscriber._running is False

    def test_signal_handler_calls_shutdown(self):
        """Test that signal handler triggers shutdown."""
        from examples.online_serving.kv_events_subscriber import _signal_handler

        # Create a mock subscriber
        mock_subscriber = MagicMock()
        mock_subscriber.shutdown = MagicMock()

        # Patch the global _subscriber
        with patch(
            "examples.online_serving.kv_events_subscriber._subscriber",
            mock_subscriber,
        ):
            from examples.online_serving import kv_events_subscriber

            kv_events_subscriber._subscriber = mock_subscriber
            _signal_handler(signal.SIGTERM, None)
            mock_subscriber.shutdown.assert_called_once()


# =============================================================================
# P5: Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling (P5)."""

    def test_decode_error_does_not_crash(self):
        """Test that decode errors are logged but don't crash."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )

        # Try to process invalid payload
        invalid_payload = b"not valid msgpack data"

        # This should log an error but not raise
        subscriber._process_message(invalid_payload, seq=1)

        # Subscriber should still be functional
        assert subscriber._last_seq == -1  # Not updated on error

    def test_file_open_error_exits(self):
        """Test that file open error causes exit."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
            output_file="/nonexistent/directory/file.jsonl",
        )

        with pytest.raises(SystemExit) as exc_info:
            subscriber._open_output()

        assert exc_info.value.code == 1


# =============================================================================
# KVEventsSubscriber Class Tests
# =============================================================================


class TestKVEventsSubscriber:
    """Tests for KVEventsSubscriber class."""

    def test_init_stores_parameters(self):
        """Test that __init__ stores parameters correctly."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://test:1234",
            replay_addr="tcp://test:1235",
            output_file="/tmp/test.jsonl",
            topic="kv-events",
        )

        assert subscriber._sub_addr == "tcp://test:1234"
        assert subscriber._replay_addr == "tcp://test:1235"
        assert subscriber._output_file_path == "/tmp/test.jsonl"
        assert subscriber._topic == "kv-events"
        assert subscriber._running is False
        assert subscriber._last_seq == -1

    def test_init_default_output_file(self):
        """Test that output_file defaults to None."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://test:1234",
            replay_addr="tcp://test:1235",
        )

        assert subscriber._output_file_path is None

    def test_init_default_topic(self):
        """Test that topic defaults to empty string (all topics)."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://test:1234",
            replay_addr="tcp://test:1235",
        )

        assert subscriber._topic == ""

    def test_close_output_flushes_file(self):
        """Test that _close_output flushes the file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            temp_path = f.name

        try:
            subscriber = KVEventsSubscriber(
                sub_addr="tcp://localhost:5557",
                replay_addr="tcp://localhost:5558",
                output_file=temp_path,
            )
            subscriber._open_output()

            # Write something
            event = BlockStored(
                block_hashes=[b"test"],
                parent_block_hash=None,
                token_ids=[1, 2, 3],
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            )
            batch = KVEventBatch(ts=1234567890.0, events=[event])
            subscriber._write_event_batch(batch)

            # Close should flush
            subscriber._close_output()

            # File should have content
            with open(temp_path, "r") as f:
                content = f.read()
                assert len(content) > 0
        finally:
            os.unlink(temp_path)


# =============================================================================
# Integration Tests (without actual ZMQ connections)
# =============================================================================


class TestMessageProcessing:
    """Tests for message processing logic."""

    def test_process_message_updates_last_seq(self):
        """Test that processing a message updates last_seq."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )

        # Create a valid encoded batch
        encoder = msgspec.msgpack.Encoder()
        event = CacheLoadCommitted(
            request_id="req-1",
            medium="CPU",
            block_count=1,
            scheduler_step=1,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)
        payload = encoder.encode(batch)

        # Mock the output file
        subscriber._output_file = StringIO()

        # Process the message
        subscriber._process_message(payload, seq=5)

        assert subscriber._last_seq == 5

    def test_process_message_writes_to_output(self):
        """Test that processing a message writes to output."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )

        encoder = msgspec.msgpack.Encoder()
        event = TransferInitiated(
            transfer_id=1,
            request_id="req-1",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=1,
            scheduler_step=1,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)
        payload = encoder.encode(batch)

        output = StringIO()
        subscriber._output_file = output

        subscriber._process_message(payload, seq=1)

        output_content = output.getvalue()
        assert len(output_content) > 0
        assert output_content.endswith("\n")

    def test_handle_replay_not_called_for_sequential(self):
        """Test that replay is not triggered for sequential messages."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )
        subscriber._last_seq = 5

        # Mock replay socket to track if called
        subscriber._replay = MagicMock()
        subscriber._poller = MagicMock()

        # This should NOT trigger replay (6 is sequential after 5)
        subscriber._handle_replay(current_seq=6)

        # Replay socket should not be used
        subscriber._replay.send.assert_not_called()

    def test_handle_replay_triggered_for_gap(self):
        """Test that replay is triggered when there's a gap."""
        subscriber = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )
        subscriber._last_seq = 5

        # Mock replay socket
        mock_replay = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []  # No replay data available

        subscriber._replay = mock_replay
        subscriber._poller = mock_poller

        # This SHOULD trigger replay (10 is not sequential after 5)
        subscriber._handle_replay(current_seq=10)

        # Replay socket should be used
        mock_replay.send.assert_called_once()


# =============================================================================
# Event Type Handling Tests
# =============================================================================


class TestEventTypeHandling:
    """Tests to verify all event types can be processed."""

    @pytest.fixture
    def subscriber(self):
        """Create a subscriber with mocked output."""
        sub = KVEventsSubscriber(
            sub_addr="tcp://localhost:5557",
            replay_addr="tcp://localhost:5558",
        )
        sub._output_file = StringIO()
        return sub

    @pytest.fixture
    def encoder(self):
        return msgspec.msgpack.Encoder()

    def test_transfer_initiated_processing(self, subscriber, encoder):
        """Test processing TransferInitiated events."""
        event = TransferInitiated(
            transfer_id=12345,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=10,
            scheduler_step=42,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=42)
        payload = encoder.encode(batch)

        subscriber._process_message(payload, seq=1)

        output = subscriber._output_file.getvalue()
        assert len(output) > 0
        # Verify JSON is valid
        parsed = json.loads(output.strip())
        assert parsed is not None

    def test_cache_eviction_processing(self, subscriber, encoder):
        """Test processing CacheEviction events."""
        event = CacheEviction(
            medium="CPU",
            blocks_evicted=5,
            eviction_reason="lru",
            scheduler_step=100,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=100)
        payload = encoder.encode(batch)

        subscriber._process_message(payload, seq=1)

        output = subscriber._output_file.getvalue()
        assert len(output) > 0

    def test_mixed_events_processing(self, subscriber, encoder):
        """Test processing a batch with multiple event types."""
        events = [
            BlockStored(
                block_hashes=[b"hash1"],
                parent_block_hash=None,
                token_ids=[1, 2, 3],
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            ),
            TransferInitiated(
                transfer_id=1,
                request_id="req-1",
                source_medium="GPU",
                dest_medium="CPU",
                block_count=1,
                scheduler_step=1,
            ),
            CacheEviction(
                medium="CPU",
                blocks_evicted=1,
                eviction_reason="capacity",
                scheduler_step=1,
            ),
        ]
        batch = KVEventBatch(ts=1234567890.0, events=events, scheduler_step=1)
        payload = encoder.encode(batch)

        subscriber._process_message(payload, seq=1)

        output = subscriber._output_file.getvalue()
        assert len(output) > 0
        # Should be a single line
        assert output.count("\n") == 1
