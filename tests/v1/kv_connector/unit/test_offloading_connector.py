# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import Request

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_scheduler,
    create_vllm_config,
)


class MockLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[BlockHash]):
        self.block_hashes: list[BlockHash] = list(block_hashes)
        # Add block_ids for TransferInitiated event emission (Phase 2)
        self.block_ids = list(range(len(self.block_hashes)))

    @staticmethod
    def medium() -> str:
        return "Mock"

    def __repr__(self) -> str:
        return repr(self.block_hashes)


class MockOffloadingHandler(OffloadingHandler):
    def __init__(self):
        self.transfer_specs: dict[int, TransferSpec] = {}
        self.completed_transfers: list[TransferResult] = []
        self.waiting_jobs: set[int] = set()
        self.completed_jobs: list[int] = []
        self.flushed_jobs: set[int] = set()

    def get_finished(self) -> list[TransferResult]:
        finished = self.completed_transfers
        self.completed_transfers = []
        return finished

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        self.transfer_specs[job_id] = spec
        self.waiting_jobs.add(job_id)
        return True

    def complete_jobs(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            if job_id in self.waiting_jobs:
                self.waiting_jobs.remove(job_id)
                self.completed_jobs.append(job_id)
                self.completed_transfers.append((job_id, True))

    def wait(self, job_ids: set[int]) -> None:
        self.flushed_jobs |= job_ids
        self.complete_jobs(job_ids)


class MockOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.manager = MagicMock(spec=OffloadingManager)
        self.manager.lookup.return_value = 0
        self.manager.prepare_load = lambda block_hashes: (
            MockLoadStoreSpec(block_hashes)
        )
        self.handler = MockOffloadingHandler()

    def get_manager(self) -> OffloadingManager:
        return self.manager

    def get_handlers(
        self, _, __
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        yield GPULoadStoreSpec, MockLoadStoreSpec, self.handler
        yield MockLoadStoreSpec, GPULoadStoreSpec, self.handler

    def complete_transfers(self):
        self.handler.complete_jobs(self.handler.waiting_jobs.copy())

    def get_completed_transfers(self) -> list[TransferSpec]:
        specs = [
            self.handler.transfer_specs[job_id]
            for job_id in self.handler.completed_jobs
        ]
        self.handler.completed_jobs.clear()
        return specs

    def get_flushed_transfers(self):
        specs = [
            self.handler.transfer_specs[job_id] for job_id in self.handler.flushed_jobs
        ]
        self.handler.flushed_jobs.clear()
        return specs


@dataclass
class TransferSummary:
    gpu_block_indices: list[int]
    offload_addresses: list[Any]


class RequestRunner:
    def __init__(
        self, offloaded_block_size: int, gpu_block_size: int, num_gpu_blocks: int
    ):
        self.offloaded_block_size: int = offloaded_block_size
        self.gpu_block_size: int = gpu_block_size
        self.num_gpu_blocks: int = num_gpu_blocks

        self.req_id: int = -1

        vllm_config = create_vllm_config(
            block_size=gpu_block_size, max_num_batched_tokens=1000
        )
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "MockOffloadingSpec",
                "spec_module_path": "tests.v1.kv_connector.unit.test_offloading_connector",  # noqa: E501
                "block_size": offloaded_block_size,
            },
        )

        self.scheduler: Scheduler = create_scheduler(
            vllm_config, num_blocks=num_gpu_blocks
        )
        self.worker_connector = OffloadingConnector(vllm_config, KVConnectorRole.WORKER)

        # register worker kv_caches to enable OffloadingWorker creations
        self.worker_connector.register_cross_layers_kv_cache(
            kv_cache=torch.empty(0),
            attn_backend=FlashAttentionBackend,
        )

        # extract connector of scheduler
        scheduler_connector = self.scheduler.connector
        assert scheduler_connector is not None
        assert isinstance(scheduler_connector, OffloadingConnector)
        self.scheduler_connector: OffloadingConnector = scheduler_connector

        # extract mocked OffloadingManager of scheduler connector
        connector_scheduler = scheduler_connector.connector_scheduler
        assert connector_scheduler is not None
        manager = connector_scheduler.manager
        assert isinstance(manager, MagicMock)
        self.manager: MagicMock = manager

        assert connector_scheduler.gpu_block_size == gpu_block_size
        assert connector_scheduler.offloaded_block_size == offloaded_block_size

        # extract OffloadingSpec of worker_connector
        connector_worker = self.worker_connector.connector_worker
        assert connector_worker is not None
        offloading_spec = connector_worker.spec
        assert isinstance(offloading_spec, MockOffloadingSpec)
        self.offloading_spec: MockOffloadingSpec = offloading_spec

        # mapping (offloading address) -> gpu_block_index
        self.offloaded: dict[Any, int] = {}

        self.completed_loads: list[TransferSummary] = []
        self.completed_stores: list[TransferSummary] = []
        self.flushed_gpu_block_indexes: set[int] = set()

        # maps {block_id: block_offset}
        self.gpu_block_index: dict[int, int] = {}

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(gpu_block_size, sha256)

        self._dummy_ctx: ForwardContext = ForwardContext(
            no_compile_layers={}, attn_metadata={}, virtual_engine=0
        )

    def new_request(self, token_ids: list[int]):
        self.req_id += 1

        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1000),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=self._block_hasher,
        )

        self.scheduler.add_request(req)

    def _parse_transfers(self):
        for transfer_spec in self.offloading_spec.get_flushed_transfers():
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, GPULoadStoreSpec)

            for block_id in src_spec.block_ids:
                self.flushed_gpu_block_indexes.add(
                    self.gpu_block_index[block_id.item()]
                )

        block_size_factor = self.offloaded_block_size // self.gpu_block_size

        for transfer_spec in self.offloading_spec.get_completed_transfers():
            src_spec, dst_spec = transfer_spec

            if isinstance(src_spec, GPULoadStoreSpec):
                store = True
                gpu_spec = src_spec
                offload_spec = dst_spec
            else:
                store = False
                gpu_spec = dst_spec
                offload_spec = src_spec

            assert isinstance(offload_spec, MockLoadStoreSpec)
            assert isinstance(gpu_spec, GPULoadStoreSpec)

            gpu_block_indices: list[int] = []
            for block_id in gpu_spec.block_ids:
                gpu_block_indices.append(self.gpu_block_index[block_id.item()])

            # list of (block_hash, sub_block_offset)
            offload_addresses: list[Any] = []
            for block_hash in offload_spec.block_hashes:
                for sub_block_idx in range(block_size_factor):
                    offload_addresses.append((block_hash, sub_block_idx))

            if store:
                assert len(gpu_block_indices) == len(offload_addresses)

                self.completed_stores.append(
                    TransferSummary(gpu_block_indices, offload_addresses)
                )
            else:
                remainder_sub_block_count = len(offload_addresses) - len(
                    gpu_block_indices
                )
                assert remainder_sub_block_count >= 0
                assert remainder_sub_block_count < block_size_factor
                offload_addresses = offload_addresses[remainder_sub_block_count:]

                self.completed_loads.append(
                    TransferSummary(gpu_block_indices, offload_addresses)
                )

    def _update_gpu_block_idx(self):
        for blocks in self.scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks.values():
            for block_idx, block in enumerate(blocks):
                self.gpu_block_index[block.block_id] = block_idx

    def _run(self, decoded_tokens: list[int], complete_transfers: bool):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
        """

        tokens_iter = iter(decoded_tokens)
        token_id = next(tokens_iter, None)
        while True:
            assert self.scheduler.requests

            scheduler_output = self.scheduler.schedule()
            self._update_gpu_block_idx()

            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)

            if scheduler_output.preempted_req_ids:
                self.worker_connector.handle_preemptions(
                    scheduler_output.preempted_req_ids
                )

            self.worker_connector.bind_connector_metadata(kv_connector_metadata)
            self.worker_connector.start_load_kv(self._dummy_ctx)

            if scheduler_output.total_num_scheduled_tokens > 0:
                self.worker_connector.wait_for_save()

            if complete_transfers:
                self.offloading_spec.complete_transfers()

            finished_sending, finished_recving = self.worker_connector.get_finished(
                scheduler_output.finished_req_ids
            )

            self.worker_connector.clear_connector_metadata()

            model_runner_output = create_model_runner_output(
                reqs=self.scheduler.running,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                token_id=token_id or 0,
            )

            prev_token_id = token_id
            if self.scheduler.running:
                token_id = next(tokens_iter, None)

            self.scheduler.update_from_output(scheduler_output, model_runner_output)

            if (
                prev_token_id is EOS_TOKEN_ID
                and prev_token_id != token_id
                and self.scheduler.requests
            ):
                # continue for one more step to allow offloading to kick off
                continue

            if token_id is None:
                break

        self._parse_transfers()

        # run one more step to update finished stored
        if EOS_TOKEN_ID in decoded_tokens:
            assert not self.scheduler.running

            while self.scheduler.requests:
                scheduler_output = self.scheduler.schedule()

                finished_sending, finished_recving = self.worker_connector.get_finished(
                    scheduler_output.finished_req_ids
                )

                assert not finished_recving

                model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
                model_runner_output.kv_connector_output = KVConnectorOutput(
                    finished_sending=finished_sending
                )

                self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def run(
        self,
        decoded_tokens: list[int],
        complete_transfers: bool = True,
        expected_stored_gpu_block_indexes: tuple[int, ...] = (),
        expected_loaded_gpu_block_indexes: tuple[int, ...] = (),
        expected_flushed_gpu_block_indexes: tuple[int, ...] = (),
    ):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
            expected_stored_gpu_block_indexes: GPU block indexes
                that are expected to be written during the run.
            expected_loaded_gpu_block_indexes: GPU block indexes
                that are expected to be loaded during the run.
            expected_flushed_gpu_block_indexes: GPU block indexes
                that are expected to be flushed during the run.
        """

        self.manager.reset_mock()
        self._run(decoded_tokens, complete_transfers)

        loaded_gpu_block_indexes: set[int] = set()
        for transfer in self.completed_loads:
            for gpu_block_idx, offloaded_address in zip(
                transfer.gpu_block_indices, transfer.offload_addresses
            ):
                loaded_gpu_block_indexes.add(gpu_block_idx)
                assert gpu_block_idx == self.offloaded[offloaded_address]

        assert set(expected_loaded_gpu_block_indexes) == loaded_gpu_block_indexes
        self.completed_loads.clear()

        stored_gpu_block_indexes: set[int] = set()
        for transfer in self.completed_stores:
            for gpu_block_idx, offloaded_address in zip(
                transfer.gpu_block_indices, transfer.offload_addresses
            ):
                stored_gpu_block_indexes.add(gpu_block_idx)
                self.offloaded[offloaded_address] = gpu_block_idx

        assert set(expected_stored_gpu_block_indexes) == stored_gpu_block_indexes
        self.completed_stores.clear()

        assert set(expected_flushed_gpu_block_indexes) == self.flushed_gpu_block_indexes
        self.flushed_gpu_block_indexes.clear()


@pytest.fixture
def request_runner():
    runners = []

    def runner_factory(offloaded_block_size, gpu_block_size, num_gpu_blocks):
        runner = RequestRunner(
            offloaded_block_size=offloaded_block_size,
            gpu_block_size=gpu_block_size,
            num_gpu_blocks=num_gpu_blocks,
        )
        runners.append(runner)
        return runner

    yield runner_factory  # pass factory to the test


def generate_store_output(block_hashes: Iterable[BlockHash]):
    block_hashes = list(block_hashes)
    return PrepareStoreOutput(
        block_hashes_to_store=list(block_hashes),
        store_spec=MockLoadStoreSpec(block_hashes),
        block_hashes_evicted=[],
    )


def test_offloading_connector(request_runner):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100
    block_size_factor = offloaded_block_size // gpu_block_size

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # 3 blocks, store just the middle block (skip first and last)
    # blocks = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(list(block_hashes)[1:2])
    )
    runner.run(decoded_tokens=[0])

    # add block missing 1 token -> no offload
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size - 1),
        expected_stored_gpu_block_indexes=(3, 4, 5),
    )
    runner.manager.prepare_store.assert_not_called()

    # +1 token -> single block, fail prepare_store
    runner.manager.prepare_store.side_effect = lambda block_hashes: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # 1 more block, now set block_hashes_to_store = []
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.run(decoded_tokens=[0] * offloaded_block_size)

    # 1 more block, now check touch was called with all 6 blocks
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(decoded_tokens=[0] * offloaded_block_size)
    runner.manager.touch.assert_called()
    block_hashes1 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes1) == 6

    # terminate request
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(15, 16, 17),
    )

    # create a new request differing only on the last token
    runner.new_request(token_ids=[0] * (offloaded_block_size * 6 - 1) + [1])
    runner.run(decoded_tokens=[0])
    runner.manager.touch.assert_called()
    block_hashes2 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes2) == 6

    # verify hashes are the same, except for the last block
    assert block_hashes1[:5] == block_hashes2[:5]
    assert block_hashes1[5] != block_hashes2[5]

    # terminate request
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=tuple(range(6 * block_size_factor)),
    )

    # full_block_tokens - num_computed_tokens < offloaded_block_size
    runner.new_request(
        token_ids=[0] * gpu_block_size + [1] * (offloaded_block_size - gpu_block_size)
    )
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_not_called()

    # single block lookup with no hits
    runner.new_request(token_ids=[1] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_called()
    assert len(list(runner.manager.lookup.call_args.args[0])) == 1

    # single block lookup with a hit
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.manager.lookup.return_value = 1
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID], expected_loaded_gpu_block_indexes=(0, 1, 2)
    )

    # single block lookup with a hit in a middle block
    runner.new_request(
        token_ids=[0] * offloaded_block_size * 2 + [1] * offloaded_block_size
    )
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.manager.lookup.return_value = 1
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID], expected_loaded_gpu_block_indexes=(3, 4, 5)
    )

    # test take_events
    def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
        return [BlockHash(str(i).encode()) for i in int_hashes]

    def take_events() -> Iterable[OffloadingEvent]:
        yield OffloadingEvent(
            block_hashes=to_hashes([1, 2, 3]), block_size=16, medium="A", removed=False
        )
        yield OffloadingEvent(
            block_hashes=to_hashes([4, 5, 6]), block_size=32, medium="B", removed=True
        )

    runner.manager.take_events.side_effect = take_events
    events = list(runner.scheduler_connector.take_events())
    assert len(events) == 2
    event = events[0]
    assert isinstance(event, BlockStored)
    assert event.block_hashes == to_hashes([1, 2, 3])
    assert event.block_size == 16
    assert event.medium == "A"
    assert event.token_ids == []
    assert event.parent_block_hash is None
    assert event.lora_id is None
    assert event.lora_name is None
    event = events[1]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == to_hashes([4, 5, 6])
    assert event.medium == "B"


def test_request_preemption(request_runner):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    # 2 blocks, store all, without flushing
    # blocks = [0, 1, 2], [3, 4, 5]
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
    )

    # decode 2 more blocks - 1 gpu block, storing [6, 7, 8] (no flush)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - gpu_block_size),
        complete_transfers=False,
    )

    # simulate KV cache running out of space
    free_block_queue.num_free_blocks = 0

    # request should be preempted now
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    # restore KV cache space and reset GPU prefix cache
    free_block_queue.num_free_blocks = num_free_blocks_empty
    runner.scheduler.reset_prefix_cache()

    # request should now return from preemption
    # re-load [0, ..., 8] from the CPU and store [9, 10, 11]
    runner.manager.lookup.return_value = 3
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(
        decoded_tokens=[0] * gpu_block_size,
        expected_loaded_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(9, 10, 11),
    )


def test_concurrent_lookups_of_the_same_prefix(request_runner):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0, 1, 2),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.lookup.return_value = 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # start a new request to load the same first block
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.lookup.return_value = 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request did not trigger a load
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # complete transfers
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded_gpu_block_indexes=(0, 1, 2),
    )

    # second request will use the GPU prefix cache
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)


# =============================================================================
# Phase 2 Tests: KV Cache Offloading Tracing Events
# =============================================================================


class TestPhase2Events:
    """Tests for Phase 2 KV cache offloading tracing events.

    These tests verify the behavioral contracts (C1-C9) defined in the plan.
    Tests call actual methods and verify events via take_events().
    """

    def test_cache_store_committed_emission(self):
        """Test C1: CacheStoreCommitted emitted when _get_reqs_to_store() succeeds.

        Behavioral test: calls actual _get_reqs_to_store() method and verifies
        CacheStoreCommitted event is emitted with correct fields.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_events import CacheStoreCommitted
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorScheduler,
        )

        # Create mock spec
        mock_spec = MagicMock()
        mock_spec.gpu_block_size = 4
        mock_spec.offloaded_block_size = 8
        mock_manager = MagicMock()
        mock_manager.take_events.return_value = []
        mock_spec.get_manager.return_value = mock_manager
        mock_spec.vllm_config.cache_config.enable_prefix_caching = False

        # Mock prepare_store to return a valid output
        block_hash = BlockHash(b"test_hash")
        mock_store_output = MagicMock()
        mock_store_output.block_hashes_to_store = [block_hash]
        mock_store_output.block_hashes_evicted = []
        mock_store_output.store_spec = MockLoadStoreSpec([block_hash])
        mock_manager.prepare_store.return_value = mock_store_output

        scheduler = OffloadingConnectorScheduler(mock_spec)
        scheduler.set_scheduler_step(5)

        # Create mock request with enough tokens for 1 offloaded block
        mock_request = MagicMock()
        mock_request.request_id = "test-req"
        mock_request.num_computed_tokens = 0
        mock_request.block_hashes = [block_hash, block_hash]  # 2 GPU blocks = 1 offload block

        # Register the request
        scheduler._requests["test-req"] = mock_request
        scheduler._request_block_ids["test-req"] = [0, 1]  # 2 GPU block IDs
        scheduler._next_stored_block_idx["test-req"] = 0

        # Create mock new request data (what yield_req_data yields)
        mock_new_req_data = MagicMock()
        mock_new_req_data.req_id = "test-req"
        mock_new_req_data.block_ids = ([0, 1],)  # tuple of block_id lists per group

        # Create mock scheduler_output with proper structure
        mock_cached_reqs = MagicMock()
        mock_cached_reqs.req_ids = []
        mock_cached_reqs.new_block_ids = []
        mock_cached_reqs.resumed_req_ids = set()

        mock_scheduler_output = MagicMock()
        mock_scheduler_output.scheduled_new_reqs = [mock_new_req_data]
        mock_scheduler_output.scheduled_cached_reqs = mock_cached_reqs
        mock_scheduler_output.num_scheduled_tokens = {"test-req": 8}  # 1 offloaded block
        mock_scheduler_output.preempted_req_ids = set()

        # Call the actual method
        scheduler._get_reqs_to_store(mock_scheduler_output)

        # Verify CacheStoreCommitted was emitted via take_events()
        events = list(scheduler.take_events())
        store_events = [e for e in events if isinstance(e, CacheStoreCommitted)]

        assert len(store_events) == 1
        event = store_events[0]
        assert event.request_id == "test-req"
        assert event.medium == "Mock"  # From MockLoadStoreSpec.medium()
        assert event.block_count == 1
        assert event.scheduler_step == 5

    def test_cache_load_committed_emission(self):
        """Test C1: CacheLoadCommitted emitted when update_state_after_alloc() loads.

        Behavioral test: calls actual update_state_after_alloc() method and verifies
        CacheLoadCommitted event is emitted with correct fields.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_events import CacheLoadCommitted
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorScheduler,
        )
        from vllm.v1.core.kv_cache_utils import BlockHash

        # Create mock spec
        mock_spec = MagicMock()
        mock_spec.gpu_block_size = 4
        mock_spec.offloaded_block_size = 8
        mock_manager = MagicMock()
        mock_manager.take_events.return_value = []
        mock_manager.prepare_load.return_value = MockLoadStoreSpec([BlockHash(b"hash1")])
        mock_spec.get_manager.return_value = mock_manager
        mock_spec.vllm_config.cache_config.enable_prefix_caching = False

        scheduler = OffloadingConnectorScheduler(mock_spec)
        scheduler.set_scheduler_step(3)

        # Create mock request
        mock_request = MagicMock()
        mock_request.request_id = "load-req"
        mock_request.num_tokens = 8  # 1 offloaded block
        mock_request.block_hashes = [BlockHash(b"hash1"), BlockHash(b"hash1")]

        # Create mock blocks
        mock_block = MagicMock()
        mock_block.block_hash = None  # Not computed yet
        mock_blocks = MagicMock()
        mock_blocks.get_block_ids.return_value = [[10, 11]]  # 2 GPU block IDs
        mock_blocks.blocks = [[mock_block, mock_block]]

        # Call the actual method with num_external_tokens > 0
        scheduler.update_state_after_alloc(mock_request, mock_blocks, num_external_tokens=8)

        # Verify CacheLoadCommitted was emitted via take_events()
        events = list(scheduler.take_events())
        load_events = [e for e in events if isinstance(e, CacheLoadCommitted)]

        assert len(load_events) == 1
        event = load_events[0]
        assert event.request_id == "load-req"
        assert event.medium == "Mock"  # From MockLoadStoreSpec.medium()
        assert event.block_count == 1  # 1 offloaded block
        assert event.scheduler_step == 3

    def test_eviction_before_store_ordering(self):
        """Test C9: CacheEviction emitted BEFORE CacheStoreCommitted.

        Behavioral test: sets up prepare_store() to return evicted blocks and
        verifies CacheEviction comes before CacheStoreCommitted in take_events().
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_events import CacheEviction, CacheStoreCommitted
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorScheduler,
        )

        # Create mock spec
        mock_spec = MagicMock()
        mock_spec.gpu_block_size = 4
        mock_spec.offloaded_block_size = 8
        mock_manager = MagicMock()
        mock_manager.take_events.return_value = []
        mock_spec.get_manager.return_value = mock_manager
        mock_spec.vllm_config.cache_config.enable_prefix_caching = False

        # Mock prepare_store to return evicted blocks (triggers CacheEviction)
        block_hash = BlockHash(b"test_hash")
        evicted_hash = BlockHash(b"evicted_hash")
        mock_store_output = MagicMock()
        mock_store_output.block_hashes_to_store = [block_hash]
        mock_store_output.block_hashes_evicted = [evicted_hash, evicted_hash]  # 2 evicted
        mock_store_output.store_spec = MockLoadStoreSpec([block_hash])
        mock_manager.prepare_store.return_value = mock_store_output

        scheduler = OffloadingConnectorScheduler(mock_spec)
        scheduler.set_scheduler_step(7)

        # Create mock request
        mock_request = MagicMock()
        mock_request.request_id = "store-req"
        mock_request.num_computed_tokens = 0
        mock_request.block_hashes = [block_hash, block_hash]

        # Register the request
        scheduler._requests["store-req"] = mock_request
        scheduler._request_block_ids["store-req"] = [0, 1]
        scheduler._next_stored_block_idx["store-req"] = 0

        # Create mock new request data (what yield_req_data yields)
        mock_new_req_data = MagicMock()
        mock_new_req_data.req_id = "store-req"
        mock_new_req_data.block_ids = ([0, 1],)  # tuple of block_id lists per group

        # Create mock scheduler_output with proper structure
        mock_cached_reqs = MagicMock()
        mock_cached_reqs.req_ids = []
        mock_cached_reqs.new_block_ids = []
        mock_cached_reqs.resumed_req_ids = set()

        mock_scheduler_output = MagicMock()
        mock_scheduler_output.scheduled_new_reqs = [mock_new_req_data]
        mock_scheduler_output.scheduled_cached_reqs = mock_cached_reqs
        mock_scheduler_output.num_scheduled_tokens = {"store-req": 8}
        mock_scheduler_output.preempted_req_ids = set()

        # Call the actual method
        scheduler._get_reqs_to_store(mock_scheduler_output)

        # Verify ordering via take_events()
        events = list(scheduler.take_events())

        # Filter to just eviction and store events
        relevant_events = [e for e in events
                          if isinstance(e, (CacheEviction, CacheStoreCommitted))]

        assert len(relevant_events) == 2
        assert isinstance(relevant_events[0], CacheEviction), \
            "C9: CacheEviction must come BEFORE CacheStoreCommitted"
        assert isinstance(relevant_events[1], CacheStoreCommitted), \
            "C9: CacheStoreCommitted must come AFTER CacheEviction"

        # Verify eviction fields
        assert relevant_events[0].blocks_evicted == 2
        assert relevant_events[0].eviction_reason == "lru"
        assert relevant_events[0].scheduler_step == 7

    def test_worker_transfer_id_generator(self):
        """Test TransferIdGenerator integration in worker."""
        from vllm.distributed.kv_events import TransferIdGenerator

        gen = TransferIdGenerator(rank=0)

        # Generate IDs
        id1 = gen.next_id()
        id2 = gen.next_id()
        id3 = gen.next_id()

        # Verify monotonically increasing
        assert id1 < id2 < id3

        # Verify uniqueness
        assert len({id1, id2, id3}) == 3

        # Verify rank extraction
        assert TransferIdGenerator.extract_rank(id1) == 0
        assert TransferIdGenerator.extract_counter(id1) == 1
        assert TransferIdGenerator.extract_counter(id2) == 2
        assert TransferIdGenerator.extract_counter(id3) == 3

    def test_offloading_kv_events_class(self):
        """Test OffloadingKVEvents class for event collection."""
        from vllm.distributed.kv_events import TransferInitiated
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingKVEvents,
        )

        events_container = OffloadingKVEvents(num_workers=1)

        # Add events
        event1 = TransferInitiated(
            transfer_id=1,
            request_id="req-1",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=10,
            scheduler_step=1,
        )
        event2 = TransferInitiated(
            transfer_id=2,
            request_id="req-2",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=5,
            scheduler_step=1,
        )

        events_container.add_events([event1, event2])

        # Verify events are collected
        all_events = events_container.get_all_events()
        assert len(all_events) == 2

        # Verify clear works
        events_container.clear_events()
        assert len(events_container.get_all_events()) == 0

    def test_scheduler_step_set_correctly(self):
        """Test that set_scheduler_step updates _current_scheduler_step."""
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorScheduler,
        )

        # Create a minimal mock spec (no spec= to allow nested attribute creation)
        mock_spec = MagicMock()
        mock_spec.gpu_block_size = 4
        mock_spec.offloaded_block_size = 8
        mock_manager = MagicMock()
        mock_spec.get_manager.return_value = mock_manager
        mock_spec.vllm_config.cache_config.enable_prefix_caching = False

        scheduler = OffloadingConnectorScheduler(mock_spec)

        # Initial value should be 0
        assert scheduler._current_scheduler_step == 0

        # Set to various values
        scheduler.set_scheduler_step(1)
        assert scheduler._current_scheduler_step == 1

        scheduler.set_scheduler_step(100)
        assert scheduler._current_scheduler_step == 100

    def test_metadata_scheduler_step(self):
        """Test that OffloadingConnectorMetadata includes scheduler_step."""
        # Directly test the metadata dataclass structure
        metadata = OffloadingConnectorMetadata(
            reqs_to_load={},
            reqs_to_store={},
            scheduler_step=42,
        )

        # Verify scheduler_step field is present and correct
        assert hasattr(metadata, "scheduler_step")
        assert isinstance(metadata.scheduler_step, int)
        assert metadata.scheduler_step == 42

        # Test default value
        metadata_default = OffloadingConnectorMetadata(
            reqs_to_load={},
            reqs_to_store={},
        )
        assert metadata_default.scheduler_step == 0

    def test_transfer_initiated_event_fields(self):
        """Test TransferInitiated event field values (C2)."""
        from vllm.distributed.kv_events import TransferInitiated

        event = TransferInitiated(
            transfer_id=12345,
            request_id="req-abc",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=8,
            scheduler_step=42,
        )

        assert event.transfer_id == 12345
        assert event.request_id == "req-abc"
        assert event.source_medium == "GPU"
        assert event.dest_medium == "CPU"
        assert event.block_count == 8
        assert event.scheduler_step == 42

        # Verify hashability
        assert hash(event) is not None

    def test_transfer_completed_event_fields(self):
        """Test TransferCompleted event field values (C2)."""
        from vllm.distributed.kv_events import TransferCompleted

        event = TransferCompleted(
            transfer_id=12345,
            request_id="req-abc",
            source_medium="GPU",
            dest_medium="CPU",
            block_count=8,
            success=True,
            scheduler_step=42,
        )

        assert event.transfer_id == 12345
        assert event.request_id == "req-abc"
        assert event.source_medium == "GPU"
        assert event.dest_medium == "CPU"
        assert event.block_count == 8
        assert event.success is True
        assert event.scheduler_step == 42

        # Test failure case
        failure_event = TransferCompleted(
            transfer_id=12346,
            request_id="req-def",
            source_medium="CPU",
            dest_medium="GPU",
            block_count=4,
            success=False,
            scheduler_step=43,
        )
        assert failure_event.success is False

    def test_no_events_when_pending_empty(self):
        """Test that take_events returns empty when no events pending."""
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorScheduler,
        )

        # Create a minimal mock spec (no spec= to allow nested attribute creation)
        mock_spec = MagicMock()
        mock_spec.gpu_block_size = 4
        mock_spec.offloaded_block_size = 8
        mock_manager = MagicMock()
        mock_manager.take_events.return_value = []
        mock_spec.get_manager.return_value = mock_manager
        mock_spec.vllm_config.cache_config.enable_prefix_caching = False

        scheduler = OffloadingConnectorScheduler(mock_spec)

        # No events should be returned when none were added
        events = list(scheduler.take_events())
        assert len(events) == 0

    def test_worker_transfer_initiated_completed_pairing(self):
        """Test C4: Every TransferInitiated has matching TransferCompleted.

        Behavioral test: calls actual worker methods and verifies that
        TransferInitiated events are paired with TransferCompleted events.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_events import TransferCompleted, TransferInitiated
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnectorMetadata,
            OffloadingConnectorWorker,
        )

        # Create mock spec with mock handler
        mock_spec = MagicMock()
        mock_handler = MockOffloadingHandler()

        def mock_get_handlers(kv_caches, attn_backends):
            yield GPULoadStoreSpec, MockLoadStoreSpec, mock_handler
            yield MockLoadStoreSpec, GPULoadStoreSpec, mock_handler

        mock_spec.get_handlers = mock_get_handlers

        worker = OffloadingConnectorWorker(mock_spec)

        # Register handlers (required before transfers)
        worker._register_handlers({}, {})

        # Create transfer spec for a load operation
        src_spec = MockLoadStoreSpec([BlockHash(b"hash1")])
        dst_spec = GPULoadStoreSpec([0, 1])
        transfer_spec = (src_spec, dst_spec)

        # Create metadata with a load request
        metadata = OffloadingConnectorMetadata(
            reqs_to_load={"req-1": transfer_spec},
            reqs_to_store={},
            scheduler_step=10,
        )

        # Call start_kv_transfers (should emit TransferInitiated)
        worker.start_kv_transfers(metadata)

        # Get events - should have TransferInitiated
        events_container = worker.get_kv_connector_kv_cache_events()
        assert events_container is not None
        events = events_container.get_all_events()
        initiated_events = [e for e in events if isinstance(e, TransferInitiated)]
        assert len(initiated_events) == 1
        transfer_id = initiated_events[0].transfer_id

        # Complete the transfer
        mock_handler.complete_jobs(mock_handler.waiting_jobs.copy())

        # Call get_finished (should emit TransferCompleted)
        worker.get_finished(set())

        # Get events again - should have TransferCompleted
        events_container = worker.get_kv_connector_kv_cache_events()
        assert events_container is not None
        events = events_container.get_all_events()
        completed_events = [e for e in events if isinstance(e, TransferCompleted)]

        assert len(completed_events) == 1
        assert completed_events[0].transfer_id == transfer_id, \
            "C4: TransferCompleted must have same transfer_id as TransferInitiated"
        assert completed_events[0].success is True


def test_phase2_store_events_integration(request_runner):
    """Integration test: Verify store flow works with Phase 2 event infrastructure.

    This test verifies that the store flow completes successfully with the
    Phase 2 event emission code in place. The events are consumed by the
    scheduler's internal processing, so we verify the flow completes without
    errors and that the expected blocks are stored.
    """
    offloaded_block_size = 8
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # Configure manager to store blocks
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )

    # Run a request that will trigger store
    runner.new_request(token_ids=[0] * offloaded_block_size)

    # This should complete without errors, verifying Phase 2 code doesn't break flow
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0, 1),
    )

    # Verify manager.prepare_store was called (store flow executed)
    assert runner.manager.prepare_store.called, \
        "prepare_store should be called during store flow"


def test_phase2_load_events_integration(request_runner):
    """Integration test: Verify load flow works with Phase 2 event infrastructure.

    This test verifies that the load flow completes successfully with the
    Phase 2 event emission code in place.
    """
    offloaded_block_size = 8
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # First, store some blocks
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0, 1),
    )

    # Reset prefix cache and configure for load hit
    runner.scheduler.reset_prefix_cache()
    runner.manager.lookup.return_value = 1  # 1 block hit
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )

    # Create new request that should load from cache
    runner.new_request(token_ids=[0] * offloaded_block_size)

    # This should complete without errors, verifying Phase 2 code doesn't break flow
    # The expected_loaded_gpu_block_indexes assertion verifies the load flow executed
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded_gpu_block_indexes=(0, 1),
    )
