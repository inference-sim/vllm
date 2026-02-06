# KV Cache Offloading: Investigation Document

This document captures concerns, required reviews, and test strategies for the KV cache CPU offloading feature, specifically around the chunked prefill bug fix.

## Background

### The Bug
During chunked prefill with CPU offloading enabled, an `IndexError` occurs at `offloading_connector.py:505`:
```
IndexError: list index out of range
  src_block_ids.append(block_ids[gpu_block_idx + i])
```

**Root cause**: `req.block_hashes` is populated at request creation for ALL prompt tokens, but `_request_block_ids` only contains blocks allocated for tokens scheduled so far. During chunked prefill, `len(req.block_hashes) > len(block_ids)`.

### The Fix
```python
num_blocks = min(
    len(req.block_hashes) // self.block_size_factor,
    len(block_ids) // self.block_size_factor,
)
```

### Key Files
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `vllm/v1/request.py` (block_hashes initialization)
- `tests/v1/kv_connector/unit/test_offloading_connector.py`

---

## Open Concerns

### 1. Timing Dependencies
The fix relies on the **store deferral mechanism** (stores prepared in step N are submitted in step N+1). This ensures KV is computed before transfer. However:
- This is an implicit dependency, not explicitly documented or tested
- If deferral timing changes, the fix could break silently
- Need to verify: Is deferral guaranteed for all code paths?

### 2. Edge Cases Not Verified
- **Preemption during chunked prefill**: What happens if a request is preempted mid-chunked-prefill? Are `block_ids` and `block_hashes` properly reset/synchronized?
- **Block boundary alignment**: What if `num_allocated_gpu_blocks % block_size_factor != 0`? Does truncation cause issues?
- **Very long sequences**: Behavior with 10+ chunks not tested
- **Concurrent chunked prefill**: Multiple requests chunking simultaneously

### 3. Hash vs Computation Timing
- `req.block_hashes` reflects token IDs, not computed KV
- For prefill: hashes exist before computation (from request creation)
- For decode: hashes are added after computation
- Need to verify: Does the fix correctly handle both cases?

### 4. Test Quality Concerns
Current tests:
- Use mocked `OffloadingManager` and `OffloadingHandler`
- Don't test actual GPU→CPU transfers
- Hardcode expected block indices (fragile to timing changes)
- Don't test the deferral mechanism directly

---

## Code Review Checklist

### 1. Trace All Code Paths
For each scenario, trace through `_get_reqs_to_store()`:
- [ ] Normal prefill (all tokens in one batch)
- [ ] Chunked prefill (2 chunks)
- [ ] Chunked prefill (3+ chunks)
- [ ] Decode without new block
- [ ] Decode completing a new block
- [ ] Preemption during chunked prefill
- [ ] Resume after preemption

### 2. Verify Invariants
At each point where `block_ids[idx]` is accessed:
- [ ] Is `idx < len(block_ids)` guaranteed?
- [ ] What is the source of the index calculation?
- [ ] Could any race condition invalidate the index?

### 3. Review Deferral Mechanism
- [ ] Where is deferral implemented? (`_unsubmitted_store_jobs`, `_submit_deferred_stores`)
- [ ] Is deferral guaranteed for all store operations?
- [ ] What happens if deferral is bypassed (e.g., during flush/preemption)?

### 4. Review Block Hash Lifecycle
- [ ] When is `req.block_hashes` populated for prefill?
- [ ] When is it extended for decode?
- [ ] Is it ever truncated or reset?
- [ ] How does it interact with prefix caching?

---

## Kubernetes E2E Test Plan

### Test Environment
```yaml
# Key configuration
--model facebook/opt-125m
--kv-offloading-size 8.0
--max-num-batched-tokens 16  # Small to force chunking
--gpu-memory-utilization 0.1
--enable-chunked-prefill
```

### Test Cases

#### TC1: Basic Chunked Prefill
```bash
# Prompt longer than max_num_batched_tokens
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model": "opt-125m", "prompt": "<32+ tokens>", "max_tokens": 10}'
```
**Expected**: No crash, response returned

#### TC2: Multiple Concurrent Chunked Requests
```bash
# Send 10+ concurrent requests with long prompts
for i in {1..10}; do
  curl -s http://localhost:8000/v1/completions \
    -d '{"model": "opt-125m", "prompt": "Request '$i': <long prompt>", "max_tokens": 50}' &
done
wait
```
**Expected**: All requests complete, no crashes

#### TC3: Trigger CPU Offloading
```bash
# Exceed GPU KV cache capacity to force offloading
for i in {1..100}; do
  curl -s http://localhost:8000/v1/completions \
    -d '{"model": "opt-125m", "prompt": "Request '$i': <~500 tokens>", "max_tokens": 100}' &
done
wait
```
**Expected**: KV events show `CacheStoreCommitted`, `TransferInitiated`, `TransferCompleted`

#### TC4: Verify KV Events
```bash
# Check kv_events.jsonl for correct event sequence
kubectl cp pvc-debug:/mnt/data/kv_events.jsonl ./kv_events.jsonl
# Verify: CacheStoreCommitted count > 0
# Verify: Each TransferInitiated has matching TransferCompleted
```

#### TC5: Stress Test with Preemption
```bash
# Many concurrent requests to trigger preemption
for i in {1..300}; do
  curl -s http://localhost:8000/v1/completions \
    -d '{"model": "opt-125m", "prompt": "Request '$i': <long unique prompt>", "max_tokens": 200}' &
done
wait
```
**Expected**: No crashes, some requests may timeout but server stays healthy

---

## Behavioral Contracts

### Contract 1: No Index Out of Bounds
**Statement**: `_get_reqs_to_store()` must never access `block_ids[idx]` where `idx >= len(block_ids)`.

**Invariant**: `num_blocks * block_size_factor <= len(block_ids)`

### Contract 2: Only Store Computed Blocks
**Statement**: A store transfer must only be initiated for blocks whose KV cache has been computed.

**Mechanism**: Store preparation happens in step N, actual transfer in step N+1 (deferral).

**Invariant**: At transfer time, all referenced blocks have computed KV.

### Contract 3: Only Store Hashed Blocks
**Statement**: A store transfer must only reference blocks for which hashes exist.

**Invariant**: `num_blocks <= len(req.block_hashes) // block_size_factor`

### Contract 4: Chunked Prefill Completeness
**Statement**: After all prefill chunks complete, all prompt blocks should be stored to CPU (if offloading is enabled and capacity allows).

**Invariant**: Eventually, stored blocks cover the full prompt.

### Contract 5: Idempotent Preemption
**Statement**: Preempting and resuming a request must not corrupt block tracking state.

**Invariant**: After preemption, `_request_block_ids[req_id]` is reset; after resume, it's rebuilt correctly.

---

## Behavioral Test Strategy

### Principle: Test Outcomes, Not Implementation

Instead of:
```python
# Bad: Tests implementation detail
assert num_blocks == min(hashes // factor, allocs // factor)
```

Do:
```python
# Good: Tests observable behavior
assert no_crash_during_chunked_prefill()
assert all_prompt_blocks_eventually_stored()
```

### Test Categories

#### Category A: Crash Prevention
- Create scenarios that would crash without the fix
- Assert: No exception raised
- Example: Chunked prefill with various prompt lengths

#### Category B: Correctness
- Verify correct blocks are stored
- Assert: Stored block indices match expected set
- Don't hardcode indices; derive from config

#### Category C: Completeness
- Verify all blocks are eventually stored
- Assert: After request completes, all blocks accounted for
- May require multiple scheduler steps

#### Category D: Timing Independence
- Tests should not depend on specific step timing
- Assert invariants that hold regardless of when stores happen
- Example: "Blocks are stored in allocation order" vs "Block 3 stored in step 2"

### Proposed Test Structure

```python
def test_chunked_prefill_contract(config):
    """
    Given: A request with tokens > max_num_batched_tokens
    When: Request is processed through chunked prefill
    Then:
      - No crash occurs
      - All prompt blocks are eventually stored
      - Stores happen only for allocated blocks
    """
    prompt_tokens = config.max_num_batched_tokens * 2
    expected_gpu_blocks = prompt_tokens // config.gpu_block_size

    runner = create_runner(config)
    runner.new_request(token_ids=[0] * prompt_tokens)
    runner.run_to_completion()

    # Verify contract, not implementation
    assert runner.stored_block_count >= expected_gpu_blocks
    assert runner.no_index_errors_occurred()
```

---

## Next Steps

1. **Code Review**: Walk through `_get_reqs_to_store()` for each scenario in the checklist
2. **Verify Deferral**: Confirm store deferral is guaranteed and documented
3. **K8s Testing**: Run TC1-TC5 on actual cluster
4. **Improve Tests**: Refactor tests to be more behavioral, less implementation-coupled
5. **Document Contracts**: Add contract assertions to the codebase as runtime checks or comments

---

## References

- Bug reproduction: `testk8s/README.md`
- Original bug analysis: `/tmp/originalbugs.md`
- Test file: `tests/v1/kv_connector/unit/test_offloading_connector.py`
- Fix location: `offloading_connector.py:429-438`
