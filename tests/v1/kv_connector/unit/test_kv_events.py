# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache event types and serialization.

Phase 1 of KV Offloading Tracing: Event type definitions and serialization.
"""

import pytest
from msgspec.msgpack import Decoder, Encoder

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheEviction,
    CacheLoadCommitted,
    CacheStoreCommitted,
    KVEventBatch,
    RemoteTransferCompleted,
    RemoteTransferInitiated,
    TransferCompleted,
    TransferIdGenerator,
    TransferInitiated,
)


# =============================================================================
# TransferIdGenerator Tests
# =============================================================================


class TestTransferIdGenerator:
    """Tests for TransferIdGenerator."""

    def test_basic_generation(self):
        """Test basic ID generation."""
        gen = TransferIdGenerator(rank=0)
        id1 = gen.next_id()
        id2 = gen.next_id()
        id3 = gen.next_id()

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_rank_encoding(self):
        """Test that rank is encoded in high 32 bits."""
        gen = TransferIdGenerator(rank=5)
        id1 = gen.next_id()

        # rank=5 should be in high 32 bits: (5 << 32) | 1
        expected = (5 << 32) | 1
        assert id1 == expected

    def test_extract_rank(self):
        """Test rank extraction from transfer ID."""
        gen = TransferIdGenerator(rank=42)
        transfer_id = gen.next_id()

        assert TransferIdGenerator.extract_rank(transfer_id) == 42

    def test_extract_counter(self):
        """Test counter extraction from transfer ID."""
        gen = TransferIdGenerator(rank=10)
        gen.next_id()  # counter = 1
        gen.next_id()  # counter = 2
        transfer_id = gen.next_id()  # counter = 3

        assert TransferIdGenerator.extract_counter(transfer_id) == 3

    def test_roundtrip_extraction(self):
        """Test that extract functions are inverses of encoding."""
        rank = 12345
        gen = TransferIdGenerator(rank=rank)

        for i in range(1, 100):
            transfer_id = gen.next_id()
            assert TransferIdGenerator.extract_rank(transfer_id) == rank
            assert TransferIdGenerator.extract_counter(transfer_id) == i

    def test_uniqueness_across_ranks(self):
        """Test that IDs from different ranks don't collide."""
        gen0 = TransferIdGenerator(rank=0)
        gen1 = TransferIdGenerator(rank=1)
        gen2 = TransferIdGenerator(rank=2)

        ids = set()
        for gen in [gen0, gen1, gen2]:
            for _ in range(100):
                transfer_id = gen.next_id()
                assert transfer_id not in ids
                ids.add(transfer_id)

    def test_invalid_rank_negative(self):
        """Test that negative rank raises error."""
        with pytest.raises(ValueError, match="rank must be in"):
            TransferIdGenerator(rank=-1)

    def test_invalid_rank_too_large(self):
        """Test that rank >= 2^32 raises error."""
        with pytest.raises(ValueError, match="rank must be in"):
            TransferIdGenerator(rank=1 << 32)

    def test_max_valid_rank(self):
        """Test maximum valid rank."""
        max_rank = (1 << 32) - 1
        gen = TransferIdGenerator(rank=max_rank)
        transfer_id = gen.next_id()
        assert TransferIdGenerator.extract_rank(transfer_id) == max_rank


# =============================================================================
# Event Construction Tests
# =============================================================================


class TestTransferEvents:
    """Tests for transfer event construction."""

    def test_transfer_initiated_construction(self):
        """Test TransferInitiated event construction."""
        event = TransferInitiated(
            transfer_id=12345,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=10,
            scheduler_step=42,
        )

        assert event.transfer_id == 12345
        assert event.request_id == "req-001"
        assert event.source_medium == "GPU"
        assert event.dest_medium == "CPU"
        assert event.block_count == 10
        assert event.scheduler_step == 42

    def test_transfer_completed_construction(self):
        """Test TransferCompleted event construction."""
        event = TransferCompleted(
            transfer_id=12345,
            request_id="req-001",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=10,
            success=True,
            scheduler_step=42,
        )

        assert event.transfer_id == 12345
        assert event.success is True

        # Test failure case
        event_failed = TransferCompleted(
            transfer_id=12346,
            request_id="req-002",
            source_medium="CPU",
            dest_medium="DISK",
            block_count=5,
            success=False,
            scheduler_step=43,
        )
        assert event_failed.success is False


class TestRemoteTransferEvents:
    """Tests for remote transfer event construction."""

    def test_remote_transfer_initiated_construction(self):
        """Test RemoteTransferInitiated event construction."""
        event = RemoteTransferInitiated(
            transfer_id=67890,
            request_id="req-003",
            connector_type="NIXL",
            source_rank=0,
            dest_rank=1,
            block_count=20,
        )

        assert event.transfer_id == 67890
        assert event.connector_type == "NIXL"
        assert event.source_rank == 0
        assert event.dest_rank == 1

    def test_remote_transfer_completed_construction(self):
        """Test RemoteTransferCompleted event construction."""
        event = RemoteTransferCompleted(
            transfer_id=67890,
            request_id="req-003",
            connector_type="P2P",
            source_rank=0,
            dest_rank=1,
            block_count=20,
            success=True,
        )

        assert event.success is True
        assert event.connector_type == "P2P"


class TestCacheOperationEvents:
    """Tests for cache operation event construction."""

    def test_cache_load_committed_construction(self):
        """Test CacheLoadCommitted event construction."""
        event = CacheLoadCommitted(
            request_id="req-004",
            medium="CPU",
            block_count=15,
            scheduler_step=100,
        )

        assert event.request_id == "req-004"
        assert event.medium == "CPU"
        assert event.block_count == 15
        assert event.scheduler_step == 100

    def test_cache_store_committed_construction(self):
        """Test CacheStoreCommitted event construction."""
        event = CacheStoreCommitted(
            request_id="req-005",
            medium="DISK",
            block_count=8,
            scheduler_step=101,
        )

        assert event.medium == "DISK"
        assert event.scheduler_step == 101

    def test_cache_eviction_construction(self):
        """Test CacheEviction event construction."""
        event = CacheEviction(
            medium="CPU",
            blocks_evicted=5,
            eviction_reason="lru",
            scheduler_step=102,
        )

        assert event.medium == "CPU"
        assert event.blocks_evicted == 5
        assert event.eviction_reason == "lru"
        assert event.block_hashes is None  # Default

    def test_cache_eviction_with_hashes(self):
        """Test CacheEviction event with block hashes."""
        hashes = [b"hash1", b"hash2", b"hash3"]
        event = CacheEviction(
            medium="GPU",
            blocks_evicted=3,
            eviction_reason="capacity",
            scheduler_step=103,
            block_hashes=hashes,
        )

        assert event.block_hashes == hashes

    def test_cache_eviction_reasons(self):
        """Test different eviction reasons."""
        for reason in ["lru", "capacity", "preemption"]:
            event = CacheEviction(
                medium="CPU",
                blocks_evicted=1,
                eviction_reason=reason,
                scheduler_step=1,
            )
            assert event.eviction_reason == reason


# =============================================================================
# Serialization Tests
# =============================================================================


class TestEventSerialization:
    """Tests for msgspec serialization/deserialization."""

    @pytest.fixture
    def encoder(self):
        return Encoder()

    @pytest.fixture
    def decoder(self):
        return Decoder(type=KVEventBatch)

    def test_transfer_initiated_roundtrip(self, encoder, decoder):
        """Test TransferInitiated serialization roundtrip."""
        event = TransferInitiated(
            transfer_id=(5 << 32) | 42,
            request_id="req-roundtrip",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=100,
            scheduler_step=999,
        )
        batch = KVEventBatch(ts=1234567890.123, events=[event], scheduler_step=999)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert len(decoded.events) == 1
        assert isinstance(decoded.events[0], TransferInitiated)
        assert decoded.events[0].transfer_id == event.transfer_id
        assert decoded.events[0].request_id == event.request_id
        assert decoded.events[0].scheduler_step == 999
        assert decoded.scheduler_step == 999

    def test_transfer_completed_roundtrip(self, encoder, decoder):
        """Test TransferCompleted serialization roundtrip."""
        event = TransferCompleted(
            transfer_id=12345,
            request_id="req-complete",
            source_medium="CPU",
            dest_medium="GPU",
            block_count=50,
            success=True,
            scheduler_step=500,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=500)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert decoded.events[0].success is True

    def test_remote_transfer_roundtrip(self, encoder, decoder):
        """Test remote transfer events serialization roundtrip."""
        initiated = RemoteTransferInitiated(
            transfer_id=99999,
            request_id="req-remote",
            connector_type="MOONCAKE",
            source_rank=0,
            dest_rank=3,
            block_count=200,
        )
        completed = RemoteTransferCompleted(
            transfer_id=99999,
            request_id="req-remote",
            connector_type="MOONCAKE",
            source_rank=0,
            dest_rank=3,
            block_count=200,
            success=True,
        )
        # Remote-only batches have scheduler_step=None
        batch = KVEventBatch(ts=1234567890.0, events=[initiated, completed])

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert len(decoded.events) == 2
        assert isinstance(decoded.events[0], RemoteTransferInitiated)
        assert isinstance(decoded.events[1], RemoteTransferCompleted)
        assert decoded.events[0].connector_type == "MOONCAKE"
        assert decoded.scheduler_step is None

    def test_cache_operation_events_roundtrip(self, encoder, decoder):
        """Test cache operation events serialization roundtrip."""
        load = CacheLoadCommitted(
            request_id="req-load",
            medium="CPU",
            block_count=10,
            scheduler_step=1,
        )
        store = CacheStoreCommitted(
            request_id="req-store",
            medium="DISK",
            block_count=20,
            scheduler_step=1,
        )
        eviction = CacheEviction(
            medium="CPU",
            blocks_evicted=5,
            eviction_reason="lru",
            scheduler_step=1,
        )
        batch = KVEventBatch(
            ts=1234567890.0, events=[load, store, eviction], scheduler_step=1
        )

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert len(decoded.events) == 3
        assert isinstance(decoded.events[0], CacheLoadCommitted)
        assert isinstance(decoded.events[1], CacheStoreCommitted)
        assert isinstance(decoded.events[2], CacheEviction)

    def test_cache_eviction_with_hashes_roundtrip(self, encoder, decoder):
        """Test CacheEviction with block_hashes serialization."""
        hashes = [b"block_hash_1", b"block_hash_2"]
        event = CacheEviction(
            medium="GPU",
            blocks_evicted=2,
            eviction_reason="capacity",
            scheduler_step=42,
            block_hashes=hashes,
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=42)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert decoded.events[0].block_hashes == hashes

    def test_mixed_event_batch(self, encoder, decoder):
        """Test batch with mix of existing and new event types."""
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
            BlockRemoved(
                block_hashes=[b"hash2"],
                medium="GPU",
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
                eviction_reason="lru",
                scheduler_step=1,
            ),
        ]
        batch = KVEventBatch(ts=1234567890.0, events=events, scheduler_step=1)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert len(decoded.events) == 4
        assert isinstance(decoded.events[0], BlockStored)
        assert isinstance(decoded.events[1], BlockRemoved)
        assert isinstance(decoded.events[2], TransferInitiated)
        assert isinstance(decoded.events[3], CacheEviction)

    def test_all_blocks_cleared_in_batch(self, encoder, decoder):
        """Test AllBlocksCleared event in batch with new events."""
        events = [
            AllBlocksCleared(),
            CacheEviction(
                medium="GPU",
                blocks_evicted=100,
                eviction_reason="preemption",
                scheduler_step=5,
            ),
        ]
        batch = KVEventBatch(ts=1234567890.0, events=events, scheduler_step=5)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert isinstance(decoded.events[0], AllBlocksCleared)
        assert isinstance(decoded.events[1], CacheEviction)


# =============================================================================
# KVEventBatch Tests
# =============================================================================


class TestKVEventBatch:
    """Tests for KVEventBatch class."""

    def test_scheduler_step_default_none(self):
        """Test that scheduler_step defaults to None."""
        batch = KVEventBatch(ts=1234567890.0, events=[])
        assert batch.scheduler_step is None

    def test_scheduler_step_explicit_value(self):
        """Test explicit scheduler_step value."""
        batch = KVEventBatch(ts=1234567890.0, events=[], scheduler_step=42)
        assert batch.scheduler_step == 42

    def test_scheduler_step_zero(self):
        """Test scheduler_step=0 is valid (not confused with None)."""
        batch = KVEventBatch(ts=1234567890.0, events=[], scheduler_step=0)
        assert batch.scheduler_step == 0

    def test_data_parallel_rank_preserved(self):
        """Test data_parallel_rank field is preserved."""
        batch = KVEventBatch(
            ts=1234567890.0,
            events=[],
            data_parallel_rank=3,
            scheduler_step=10,
        )
        assert batch.data_parallel_rank == 3
        assert batch.scheduler_step == 10

    def test_ts_field(self):
        """Test timestamp field."""
        ts = 1706886400.123456
        batch = KVEventBatch(ts=ts, events=[])
        assert batch.ts == ts


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_request_id(self):
        """Test events with empty request_id."""
        event = TransferInitiated(
            transfer_id=1,
            request_id="",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=1,
            scheduler_step=1,
        )
        assert event.request_id == ""

    def test_large_transfer_id(self):
        """Test with maximum valid transfer_id."""
        max_rank = (1 << 32) - 1
        max_counter = (1 << 32) - 1
        transfer_id = (max_rank << 32) | max_counter

        event = TransferInitiated(
            transfer_id=transfer_id,
            request_id="req-max",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=1,
            scheduler_step=1,
        )
        assert event.transfer_id == transfer_id

    def test_large_block_count(self):
        """Test with large block counts."""
        event = TransferInitiated(
            transfer_id=1,
            request_id="req-large",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=1_000_000,
            scheduler_step=1,
        )
        assert event.block_count == 1_000_000

    def test_large_scheduler_step(self):
        """Test with large scheduler step values."""
        large_step = 2**60
        event = CacheLoadCommitted(
            request_id="req-1",
            medium="CPU",
            block_count=1,
            scheduler_step=large_step,
        )
        assert event.scheduler_step == large_step

    def test_storage_mediums(self):
        """Test all documented storage mediums."""
        mediums = ["GPU", "CPU", "DISK", "LMCACHE"]
        for medium in mediums:
            event = CacheLoadCommitted(
                request_id="req-1",
                medium=medium,
                block_count=1,
                scheduler_step=1,
            )
            assert event.medium == medium

    def test_connector_types(self):
        """Test all documented connector types."""
        connector_types = ["NIXL", "P2P", "MOONCAKE"]
        for ctype in connector_types:
            event = RemoteTransferInitiated(
                transfer_id=1,
                request_id="req-1",
                connector_type=ctype,
                source_rank=0,
                dest_rank=1,
                block_count=1,
            )
            assert event.connector_type == ctype


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    @pytest.fixture
    def encoder(self):
        return Encoder()

    @pytest.fixture
    def decoder(self):
        return Decoder(type=KVEventBatch)

    def test_existing_events_still_work(self, encoder, decoder):
        """Test that existing event types still serialize correctly."""
        events = [
            BlockStored(
                block_hashes=[b"h1", b"h2"],
                parent_block_hash=b"parent",
                token_ids=[100, 200, 300],
                block_size=32,
                lora_id=1,
                medium="GPU",
                lora_name="my-lora",
            ),
            BlockRemoved(
                block_hashes=[b"h3"],
                medium="CPU",
            ),
            AllBlocksCleared(),
        ]
        # No scheduler_step (None) to simulate old-style batch
        batch = KVEventBatch(ts=1234567890.0, events=events)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert len(decoded.events) == 3
        assert decoded.scheduler_step is None  # Default preserved

    def test_batch_without_scheduler_step(self, encoder, decoder):
        """Test that batches without scheduler_step decode correctly."""
        batch = KVEventBatch(ts=1234567890.0, events=[AllBlocksCleared()])

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        # scheduler_step should be None when not provided
        assert decoded.scheduler_step is None

    def test_cache_eviction_without_block_hashes(self, encoder, decoder):
        """Test CacheEviction works without optional block_hashes."""
        event = CacheEviction(
            medium="CPU",
            blocks_evicted=10,
            eviction_reason="lru",
            scheduler_step=1,
            # block_hashes not provided, defaults to None
        )
        batch = KVEventBatch(ts=1234567890.0, events=[event], scheduler_step=1)

        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

        assert decoded.events[0].block_hashes is None
