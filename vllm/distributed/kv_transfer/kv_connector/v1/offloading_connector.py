# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    CacheEviction,
    CacheLoadCommitted,
    CacheStoreCommitted,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
    TransferCompleted,
    TransferIdGenerator,
    TransferInitiated,
)
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


class OffloadingKVEvents(KVConnectorKVEvents):
    """Concrete implementation of KVConnectorKVEvents for offloading connector.

    Collects transfer events (TransferInitiated, TransferCompleted) from
    the worker side for publishing via the event system.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "OffloadingKVEvents":
        """Aggregate KV events and retain only common events."""
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<OffloadingKVEvents events={self.get_all_events()}>"


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    scheduler_step: int = 0
    """Scheduler step for correlating transfer events."""


class OffloadingConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        assert self.connector_worker is not None
        self.connector_worker.handle_preemptions(preempted_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    def set_scheduler_step(self, step: int) -> None:
        """Set the current scheduler step for event correlation.

        Called by the scheduler before build_connector_meta() to ensure
        events emitted during this scheduling round have the correct step.

        Args:
            step: The current scheduler step counter value.
        """
        if self.connector_scheduler is not None:
            self.connector_scheduler.set_scheduler_step(step)

    def get_kv_connector_kv_cache_events(self) -> OffloadingKVEvents | None:
        """Get the KV connector events collected during the last interval.

        Returns pending worker-side events (TransferInitiated, TransferCompleted)
        wrapped in an OffloadingKVEvents container for aggregation.

        Returns:
            OffloadingKVEvents if there are pending events, None otherwise.
        """
        if self.connector_worker is not None:
            return self.connector_worker.get_kv_connector_kv_cache_events()
        return None


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[BlockHash] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        # Event tracking for KV cache tracing (Phase 2)
        self._pending_events: list[KVCacheEvent] = []
        self._current_scheduler_step: int = 0

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def set_scheduler_step(self, step: int) -> None:
        """Set the current scheduler step for event correlation.

        Called by the scheduler before build_connector_meta() to ensure
        events emitted during this scheduling round have the correct step.

        Args:
            step: The current scheduler step counter value.
        """
        self._current_scheduler_step = step

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        if self._blocks_being_loaded:
            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=start_block_idx + hits
            )

            if any(
                block_hash in self._blocks_being_loaded for block_hash in block_hashes
            ):
                # hit blocks are being loaded, delay request
                logger.debug(
                    "Delaying request %s since some of its blocks are already"
                    " being loaded",
                    request.request_id,
                )
                return None, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor >= num_blocks
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        # C1: Emit CacheLoadCommitted AFTER prepare_load() returns successfully
        # C2: block_count = length of input to prepare_load()
        load_block_count = num_blocks - start_block_idx
        self._pending_events.append(
            CacheLoadCommitted(
                request_id=request.request_id,
                medium=src_spec.medium(),
                block_count=load_block_count,
                scheduler_step=self._current_scheduler_step,
            )
        )

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(block_hashes)
        self._next_stored_block_idx[request.request_id] = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            # Calculate num_blocks as the minimum of:
            # 1. Blocks with hashes available (from req.block_hashes)
            # 2. Blocks actually allocated (from block_ids)
            #
            # During chunked prefill, req.block_hashes contains hashes for ALL
            # prompt tokens (computed at request creation), but block_ids only
            # contains blocks allocated so far. We must not attempt to store
            # blocks that haven't been allocated yet.
            num_gpu_block_hashes = len(req.block_hashes)
            num_allocated_gpu_blocks = len(block_ids)
            num_blocks = min(
                num_gpu_block_hashes // self.block_size_factor,
                num_allocated_gpu_blocks // self.block_size_factor,
            )
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                # C6: prepare_store() returns None -> No CacheStoreCommitted
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            # C1: Emit events AFTER prepare_store() returns non-None
            # C9: CacheEviction emitted BEFORE CacheStoreCommitted (same step)
            if store_output.block_hashes_evicted:
                evicted_count = len(store_output.block_hashes_evicted)
                logger.debug(
                    "CacheEviction: medium=%s, blocks_evicted=%d, step=%d",
                    store_output.store_spec.medium(),
                    evicted_count,
                    self._current_scheduler_step,
                )
                self._pending_events.append(
                    CacheEviction(
                        medium=store_output.store_spec.medium(),
                        blocks_evicted=evicted_count,
                        eviction_reason="lru",
                        scheduler_step=self._current_scheduler_step,
                    )
                )

            if not store_output.block_hashes_to_store:
                continue

            # C2: block_count = len(store_output.block_hashes_to_store)
            block_count = len(store_output.block_hashes_to_store)
            logger.debug(
                "CacheStoreCommitted: req=%s, medium=%s, blocks=%d, step=%d",
                req_id,
                store_output.store_spec.medium(),
                block_count,
                self._current_scheduler_step,
            )
            self._pending_events.append(
                CacheStoreCommitted(
                    request_id=req_id,
                    medium=store_output.store_spec.medium(),
                    block_count=block_count,
                    scheduler_step=self._current_scheduler_step,
                )
            )
            block_hashes_to_store = set(store_output.block_hashes_to_store)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(src_block_ids)

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
            scheduler_step=self._current_scheduler_step,
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            block_hashes = self._reqs_being_stored.get(req_id)
            if block_hashes:
                self.manager.complete_store(block_hashes)
                block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns pending scheduler-side events (CacheLoadCommitted,
        CacheStoreCommitted, CacheEviction) followed by manager events
        (BlockStored, BlockRemoved).

        Returns:
            A list of KV cache events.
        """
        # Yield pending scheduler-side events first
        if self._pending_events:
            event_counts: dict[str, int] = {}
            for e in self._pending_events:
                event_type = type(e).__name__
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            logger.debug(
                "OffloadingConnectorScheduler.take_events: yielding %d events: %s",
                len(self._pending_events),
                event_counts,
            )
            yield from self._pending_events
            self._pending_events = []

        # Then yield manager events (BlockStored, BlockRemoved)
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        # Transfer ID generator replaces _job_counter
        # Uses (rank << 32) | counter encoding for global uniqueness
        self._transfer_id_gen = TransferIdGenerator(rank=0)

        # transfer_id -> (req_id, is_store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active transfer IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active transfer IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # list of store jobs pending submission (transfer_id, req_id, transfer_spec)
        self._unsubmitted_store_jobs: list[tuple[int, ReqId, TransferSpec]] = []

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

        # Event tracking for KV cache tracing (Phase 2)
        self._pending_events: list[KVCacheEvent] = []
        # transfer_id -> event data for emitting TransferCompleted
        self._job_to_event_data: dict[int, dict[str, Any]] = {}
        # Current scheduler step from metadata
        self._current_scheduler_step: int = 0

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    def _submit_deferred_stores(self) -> None:
        """Submit deferred store transfers and emit TransferInitiated events.

        Stores are deferred by one scheduler step (C3) to avoid blocking token
        generation. This method submits the deferred stores and emits events.
        """
        if self._unsubmitted_store_jobs:
            logger.debug(
                "Submitting %d deferred store jobs at scheduler_step=%d",
                len(self._unsubmitted_store_jobs),
                self._current_scheduler_step,
            )

        for transfer_id, req_id, transfer_spec in self._unsubmitted_store_jobs:
            src_spec, dst_spec = transfer_spec
            success = self.worker.transfer_async(transfer_id, transfer_spec)
            if success:
                # C1: Emit TransferInitiated AFTER transfer_async() returns True
                # C2: block_count = len(src_spec.block_ids)
                block_count = len(src_spec.block_ids)  # type: ignore[attr-defined]
                logger.debug(
                    "TransferInitiated: transfer_id=%d, req=%s, %s->%s, "
                    "blocks=%d, step=%d",
                    transfer_id,
                    req_id,
                    src_spec.medium(),
                    dst_spec.medium(),
                    block_count,
                    self._current_scheduler_step,
                )
                self._pending_events.append(
                    TransferInitiated(
                        transfer_id=transfer_id,
                        request_id=req_id,
                        source_medium=src_spec.medium(),
                        dest_medium=dst_spec.medium(),
                        block_count=block_count,
                        scheduler_step=self._current_scheduler_step,
                    )
                )
                # Store event data for TransferCompleted
                self._job_to_event_data[transfer_id] = {
                    "request_id": req_id,
                    "source_medium": src_spec.medium(),
                    "dest_medium": dst_spec.medium(),
                    "block_count": block_count,
                    "scheduler_step": self._current_scheduler_step,
                }
            else:
                # C6: transfer_async() returns False -> No TransferInitiated
                logger.error(
                    "Failed to submit deferred store transfer for request %s",
                    req_id,
                )
                # Clean up tracking state
                self._jobs.pop(transfer_id, None)
                req_jobs = self._store_jobs.get(req_id)
                if req_jobs:
                    req_jobs.discard(transfer_id)
        self._unsubmitted_store_jobs.clear()

    def handle_preemptions(self, preempted_req_ids: set[str]):
        # Submit deferred stores - these are stores prepared in the previous step
        # C3: Stores are deferred by one step, so TransferInitiated uses current step
        self._submit_deferred_stores()

        for req_id in preempted_req_ids:
            transfer_ids = self._store_jobs.get(req_id)
            if transfer_ids:
                self.worker.wait(transfer_ids)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        # Update current scheduler step from metadata
        self._current_scheduler_step = metadata.scheduler_step

        # Submit deferred stores - these are stores prepared in the previous step
        # C3: Stores are deferred by one step, so TransferInitiated uses current step
        self._submit_deferred_stores()

        # Submit loads for current step
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            src_spec, dst_spec = transfer_spec
            transfer_id = self._transfer_id_gen.next_id()
            self._jobs[transfer_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = transfer_id
            success = self.worker.transfer_async(transfer_id, transfer_spec)
            if success:
                # C1: Emit TransferInitiated AFTER transfer_async() returns True
                block_count = len(src_spec.block_ids)  # type: ignore[attr-defined]
                self._pending_events.append(
                    TransferInitiated(
                        transfer_id=transfer_id,
                        request_id=req_id,
                        source_medium=src_spec.medium(),
                        dest_medium=dst_spec.medium(),
                        block_count=block_count,
                        scheduler_step=self._current_scheduler_step,
                    )
                )
                # Store event data for TransferCompleted
                self._job_to_event_data[transfer_id] = {
                    "request_id": req_id,
                    "source_medium": src_spec.medium(),
                    "dest_medium": dst_spec.medium(),
                    "block_count": block_count,
                    "scheduler_step": self._current_scheduler_step,
                }
            else:
                # C6: transfer_async() returns False -> No TransferInitiated
                logger.error(
                    "Failed to submit load transfer for request %s",
                    req_id,
                )
                # Clean up tracking state
                self._jobs.pop(transfer_id, None)
                del self._load_job[req_id]

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            transfer_id = self._transfer_id_gen.next_id()
            self._jobs[transfer_id] = (req_id, True)
            self._store_jobs[req_id].add(transfer_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            # C3: Stores are deferred - TransferInitiated emitted in next step
            self._unsubmitted_store_jobs.append((transfer_id, req_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for transfer_id, success in self.worker.get_finished():
            # C4: No-Disappear Guarantee - emit TransferCompleted for ALL completions
            # C1: Emit TransferCompleted AFTER get_finished() returns a job
            event_data = self._job_to_event_data.pop(transfer_id, None)
            if event_data:
                logger.debug(
                    "TransferCompleted: transfer_id=%d, req=%s, %s->%s, "
                    "blocks=%d, success=%s, step=%d",
                    transfer_id,
                    event_data["request_id"],
                    event_data["source_medium"],
                    event_data["dest_medium"],
                    event_data["block_count"],
                    success,
                    event_data["scheduler_step"],
                )
                self._pending_events.append(
                    TransferCompleted(
                        transfer_id=transfer_id,
                        request_id=event_data["request_id"],
                        source_medium=event_data["source_medium"],
                        dest_medium=event_data["dest_medium"],
                        block_count=event_data["block_count"],
                        success=success,
                        scheduler_step=event_data["scheduler_step"],
                    )
                )

            if not success:
                # C6: get_finished() returns failure -> TransferCompleted(success=False)
                logger.error(
                    "Transfer %d failed for request %s",
                    transfer_id,
                    event_data["request_id"] if event_data else "unknown",
                )

            req_id, store = self._jobs.pop(transfer_id)
            if store:
                req_transfers = self._store_jobs[req_id]
                req_transfers.remove(transfer_id)
                if req_transfers:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_transfer = self._load_job[req_id]
                assert transfer_id == req_transfer
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_transfers = self._store_jobs.get(req_id)
            if pending_req_transfers:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_transfers is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving

    def get_kv_connector_kv_cache_events(self) -> OffloadingKVEvents | None:
        """Get the KV connector events collected during the last interval.

        Returns pending worker-side events (TransferInitiated, TransferCompleted)
        wrapped in an OffloadingKVEvents container for aggregation.

        Returns:
            OffloadingKVEvents if there are pending events, None otherwise.
        """
        if not self._pending_events:
            logger.debug(
                "OffloadingConnectorWorker.get_kv_connector_kv_cache_events: "
                "no pending events"
            )
            return None

        event_counts = {}
        for e in self._pending_events:
            event_type = type(e).__name__
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        logger.debug(
            "OffloadingConnectorWorker.get_kv_connector_kv_cache_events: "
            "returning %d events: %s",
            len(self._pending_events),
            event_counts,
        )

        events = OffloadingKVEvents(num_workers=1)
        events.add_events(self._pending_events)
        self._pending_events = []
        return events
