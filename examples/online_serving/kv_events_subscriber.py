# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Cache Events Subscriber

A production-ready subscriber for vLLM KV cache events. Connects to the
ZMQ event publisher and writes events to JSONL format for downstream
processing (e.g., simulator training, analytics).

Configuration via environment variables:
    VLLM_KV_EVENTS_SUB_ADDR: ZMQ SUB socket address (default: tcp://localhost:5557)
    VLLM_KV_EVENTS_REPLAY_ADDR: ZMQ REQ socket for replay (default: tcp://localhost:5558)
    VLLM_KV_EVENTS_OUTPUT_FILE: Output file path (default: None, prints to stdout)
    VLLM_KV_EVENTS_TOPIC: Topic to subscribe to (default: "" for all topics)

Supports graceful shutdown via SIGTERM/SIGINT signals.

Output format (JSONL - one JSON object per line):
    {"ts": <float>, "events": [<event>, ...], "scheduler_step": <int|null>, ...}

    Each event has a "type" field indicating the event class:
    - "BlockStored": Block added to cache
    - "BlockRemoved": Block removed from cache
    - "AllBlocksCleared": All blocks cleared
    - "TransferInitiated": Local DMA transfer started
    - "TransferCompleted": Local DMA transfer finished
    - "RemoteTransferInitiated": Cross-machine transfer started
    - "RemoteTransferCompleted": Cross-machine transfer finished
    - "CacheLoadCommitted": Scheduler committed to cache load
    - "CacheStoreCommitted": Scheduler committed to cache store
    - "CacheEviction": Blocks evicted from cache

Example usage:
    # Output to stdout
    python kv_events_subscriber.py

    # Output to file
    VLLM_KV_EVENTS_OUTPUT_FILE=/var/log/kv_events.jsonl python kv_events_subscriber.py

    # Custom addresses
    VLLM_KV_EVENTS_SUB_ADDR=tcp://vllm-server:5557 python kv_events_subscriber.py
"""

import logging
import os
import signal
import sys
from types import FrameType

import msgspec
import zmq
from msgspec.msgpack import Decoder

from vllm.distributed.kv_events import KVEventBatch

# Event types available in KVEventBatch.events (for reference):
# - BlockStored, BlockRemoved, AllBlocksCleared (cache state events)
# - TransferInitiated, TransferCompleted (local DMA transfers)
# - RemoteTransferInitiated, RemoteTransferCompleted (cross-machine transfers)
# - CacheLoadCommitted, CacheStoreCommitted, CacheEviction (cache operations)

# Configure logging to stderr so stdout remains available for event output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Environment variable names
ENV_SUB_ADDR = "VLLM_KV_EVENTS_SUB_ADDR"
ENV_REPLAY_ADDR = "VLLM_KV_EVENTS_REPLAY_ADDR"
ENV_OUTPUT_FILE = "VLLM_KV_EVENTS_OUTPUT_FILE"
ENV_TOPIC = "VLLM_KV_EVENTS_TOPIC"

# Default values
DEFAULT_SUB_ADDR = "tcp://localhost:5557"
DEFAULT_REPLAY_ADDR = "tcp://localhost:5558"
DEFAULT_TOPIC = ""  # Empty string subscribes to all topics


class KVEventsSubscriber:
    """Subscriber for KV cache events with file output and graceful shutdown."""

    def __init__(
        self,
        sub_addr: str,
        replay_addr: str,
        output_file: str | None = None,
        topic: str = "",
    ) -> None:
        """Initialize the subscriber.

        Args:
            sub_addr: ZMQ SUB socket address to connect to.
            replay_addr: ZMQ REQ socket address for replay requests.
            output_file: Path to output file (None for stdout).
            topic: Topic to subscribe to (empty string for all topics).
        """
        self._sub_addr = sub_addr
        self._replay_addr = replay_addr
        self._output_file_path = output_file
        self._topic = topic
        self._output_file = None
        self._running = False
        self._last_seq = -1

        # JSON encoder for JSONL output
        self._json_encoder = msgspec.json.Encoder()

        # msgpack decoder for incoming events
        self._decoder = Decoder(type=KVEventBatch)

        # ZMQ context and sockets (initialized in run())
        self._context: zmq.Context | None = None
        self._sub: zmq.Socket | None = None
        self._replay: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None

    def _open_output(self) -> None:
        """Open the output file or use stdout."""
        if self._output_file_path:
            try:
                # Open in append mode with UTF-8 encoding
                self._output_file = open(
                    self._output_file_path, "a", encoding="utf-8"
                )
                logger.info("Writing events to file: %s", self._output_file_path)
            except OSError as e:
                logger.error("Failed to open output file %s: %s",
                           self._output_file_path, e)
                sys.exit(1)
        else:
            self._output_file = sys.stdout
            logger.info("Writing events to stdout")

    def _close_output(self) -> None:
        """Close the output file if not stdout."""
        if self._output_file and self._output_file is not sys.stdout:
            try:
                self._output_file.flush()
                self._output_file.close()
                logger.info("Output file closed")
            except OSError as e:
                logger.error("Error closing output file: %s", e)

    def _setup_sockets(self) -> None:
        """Initialize ZMQ sockets."""
        self._context = zmq.Context()

        # Set up the main subscription socket
        self._sub = self._context.socket(zmq.SUB)
        self._sub.connect(self._sub_addr)
        # Subscribe to configured topic (empty string = all topics)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, self._topic)

        # Initialize replay socket
        self._replay = self._context.socket(zmq.REQ)
        self._replay.connect(self._replay_addr)

        self._poller = zmq.Poller()
        self._poller.register(self._replay, zmq.POLLIN)

    def _cleanup_sockets(self) -> None:
        """Clean up ZMQ sockets.

        Uses linger=0 to ensure fast shutdown (important for k8s pod termination).
        This may drop in-flight messages, but the output file is flushed before
        socket cleanup, so all processed events are persisted.
        """
        if self._sub:
            self._sub.close(linger=0)
        if self._replay:
            self._replay.close(linger=0)
        if self._context:
            self._context.term()

    def _write_event_batch(self, event_batch: KVEventBatch) -> None:
        """Write an event batch as a single JSONL line.

        Args:
            event_batch: The event batch to write.
        """
        if not self._output_file:
            return

        try:
            # Encode to JSON bytes, then decode to string for file writing
            json_bytes = self._json_encoder.encode(event_batch)
            self._output_file.write(json_bytes.decode("utf-8"))
            self._output_file.write("\n")
            self._output_file.flush()
        except OSError as e:
            logger.error("Error writing to output file: %s", e)

    def _handle_replay(self, current_seq: int) -> None:
        """Request and process missed messages via replay.

        Args:
            current_seq: The current sequence number received.
        """
        if self._last_seq < 0 or current_seq <= self._last_seq + 1:
            return

        missed = current_seq - self._last_seq - 1
        logger.warning(
            "Missed %d messages (last: %d, current: %d). Requesting replay.",
            missed, self._last_seq, current_seq
        )

        try:
            self._replay.send((self._last_seq + 1).to_bytes(8, "big"))

            while self._poller.poll(timeout=200):
                seq_bytes, replay_payload = self._replay.recv_multipart()
                if not replay_payload:
                    # End of replay marker
                    break

                replay_seq = int.from_bytes(seq_bytes, "big")

                if replay_seq > self._last_seq:
                    try:
                        event_batch = self._decoder.decode(replay_payload)
                        self._write_event_batch(event_batch)
                        self._last_seq = replay_seq
                    except msgspec.DecodeError as e:
                        logger.error("Error decoding replay message: %s", e)

                    if replay_seq >= current_seq - 1:
                        break

            logger.info("Replay complete")
        except zmq.ZMQError as e:
            logger.error("ZMQ error during replay: %s", e)

    def _process_message(self, payload: bytes, seq: int) -> None:
        """Process a received message.

        Args:
            payload: The message payload.
            seq: The sequence number.
        """
        # Handle any missed messages first
        self._handle_replay(seq)

        try:
            event_batch = self._decoder.decode(payload)
            self._write_event_batch(event_batch)
            self._last_seq = seq
        except msgspec.DecodeError as e:
            logger.error("Error decoding message (seq=%d): %s", seq, e)
            # Continue processing - don't crash on decode errors

    def shutdown(self) -> None:
        """Signal the subscriber to stop."""
        logger.info("Shutting down")
        self._running = False

    def run(self) -> None:
        """Run the subscriber main loop."""
        self._open_output()
        self._setup_sockets()
        self._running = True

        topic_desc = f"topic '{self._topic}'" if self._topic else "all topics"
        logger.info(
            "Listening for KV cache events on %s (%s, replay: %s)",
            self._sub_addr, topic_desc, self._replay_addr
        )

        try:
            while self._running:
                try:
                    if self._sub.poll(50):
                        parts = self._sub.recv_multipart()
                        # Publisher sends 3-part messages: [topic, seq_bytes, payload]
                        if len(parts) != 3:
                            logger.warning(
                                "Unexpected message format: expected 3 parts, got %d",
                                len(parts)
                            )
                            continue
                        _, seq_bytes, payload = parts
                        seq = int.from_bytes(seq_bytes, "big")
                        self._process_message(payload, seq)
                except zmq.ZMQError as e:
                    if self._running:
                        logger.error("ZMQ error: %s", e)
        finally:
            self._close_output()
            self._cleanup_sockets()


# Global subscriber instance for signal handlers
_subscriber: KVEventsSubscriber | None = None


def _signal_handler(signum: int, _: FrameType | None) -> None:
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s", sig_name)
    if _subscriber:
        _subscriber.shutdown()


def main() -> None:
    """Main entry point."""
    global _subscriber

    # Read configuration from environment variables
    sub_addr = os.environ.get(ENV_SUB_ADDR, DEFAULT_SUB_ADDR)
    replay_addr = os.environ.get(ENV_REPLAY_ADDR, DEFAULT_REPLAY_ADDR)
    output_file = os.environ.get(ENV_OUTPUT_FILE)
    topic = os.environ.get(ENV_TOPIC, DEFAULT_TOPIC)

    # Create subscriber
    _subscriber = KVEventsSubscriber(
        sub_addr=sub_addr,
        replay_addr=replay_addr,
        output_file=output_file,
        topic=topic,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Run the subscriber
    _subscriber.run()

    logger.info("Subscriber stopped")


if __name__ == "__main__":
    main()
