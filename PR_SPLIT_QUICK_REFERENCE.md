# Journey Tracing PR Split - Quick Reference

## The Iron Rule: No Incomplete Resources

**If you create it, you clean it - IN THE SAME PR**

- ✅ Create span → End span on all paths
- ✅ Add to dict → Remove from dict on all paths
- ✅ Add to set → Discard from set on all paths
- ❌ **NO** "we'll add cleanup later"
- ❌ **NO** "we'll add ABORTED in the next PR"

---

## 9 PRs, ~2 Weeks

### Phase 1: Core Layer (4 PRs, ~1 week)

| PR | Branch | What | Size | Key Point |
|----|--------|------|------|-----------|
| #1 | `...-01-scheduler-init` | Init tracer | 25 lines | No per-request state |
| #2 | `...-02-core-spans-lifecycle` | Create + cleanup spans | 100 lines | **Cleanup in same PR** ✅ |
| #3 | `...-03-journey-state-cleanup` | Add state + cleanup | 50 lines | Extends PR #2 cleanup |
| #4 | `...-04-journey-events-emit` | Emit events | 120 lines | No new resources |

**After Phase 1**: Core spans work end-to-end, legacy buffering still active

---

### Phase 2: API Layer (4 PRs, ~1 week)

| PR | Branch | What | Size | Key Point |
|----|--------|------|------|-----------|
| #5 | `...-05-api-metadata` | Add metadata fields | 15 lines | No resources |
| #6 | `...-06-api-spans-full-lifecycle` | Create + close API spans | 150 lines | **All closure paths in same PR** ✅ |
| #7 | `...-07-context-propagation` | Link parent-child | 25 lines | No new resources |
| #8 | `...-08-api-additional-events` | Emit API events | 80 lines | No new resources |

**After Phase 2**: Full dual-stream works, legacy buffering still active

---

### Phase 3: Cleanup (1 PR, ~1 day)

| PR | Branch | What | Size | Key Point |
|----|--------|------|------|-----------|
| #9 | `...-09-remove-buffering` | Remove legacy buffer | 150 lines removed | Keep do_tracing() |

**After Phase 3**: Complete! Only spans, no buffering.

---

## PR #2 and #6 Are Critical

These PRs create resources, so they MUST include cleanup:

### PR #2: Core Spans
**Creates**: `_core_spans[request_id]`, span objects

**Must also include**:
- ✅ `_end_core_span_and_cleanup()` function
- ✅ Called from `finish_requests()`
- ✅ Called from `_update_from_output()` finally block
- ✅ Tests proving span.end() called
- ✅ Tests proving dict cleaned

### PR #6: API Spans
**Creates**: `api_span` objects

**Must also include**:
- ✅ `_safe_emit_departed_event()` - ends span on success
- ✅ `_safe_emit_aborted_event()` - ends span on errors
- ✅ Covers all paths: success, GenerationError, Exception, CancelledError, ValueError
- ✅ Outer finally in `chat_completion_full_generator` catches ALL exceptions
- ✅ Tests proving span ended on all paths

---

## Every PR Checklist

```markdown
### Resource Safety Checklist

- [ ] If this PR creates spans, it also ends them on all exits
- [ ] If this PR introduces dicts/sets, it also cleans them on all termination paths
- [ ] No buffering when tracer absent
- [ ] Legacy tracing untouched
- [ ] Tests prove cleanup (no dict/set growth, spans ended)
- [ ] Defensive error handling (tracing never breaks requests)
- [ ] Zero overhead when disabled

### Termination Paths Covered

- [ ] Natural completion
- [ ] Explicit abort (finish_requests)
- [ ] Exceptions during processing
- [ ] Client cancellation (if applicable)
- [ ] All paths call cleanup function
```

---

## What Makes Each PR Safe?

**PR #1**: No per-request state → Safe ✅

**PR #2**: Creates spans + cleanup in same PR → Safe ✅
- Cleanup function created
- Called on all termination paths
- Tests prove no leaks

**PR #3**: Extends PR #2 cleanup → Safe ✅
- Uses same cleanup function
- Tests prove no leaks

**PR #4**: Just event emission, no resources → Safe ✅

**PR #5**: Just field definitions, no resources → Safe ✅

**PR #6**: Creates spans + DEPARTED/ABORTED in same PR → Safe ✅
- All closure paths included
- Outer finally catches all exceptions
- Tests prove span ended on all paths

**PR #7**: Just context injection, no resources → Safe ✅

**PR #8**: Just event emission, no resources → Safe ✅

**PR #9**: Removes code, no new resources → Safe ✅

---

## Review Time Per PR

- PR #1: 10 min (tiny)
- PR #2: 25 min (critical, includes cleanup)
- PR #3: 15 min (extends cleanup)
- PR #4: 20 min (event emission)
- PR #5: 5 min (trivial)
- PR #6: 30 min (critical, all closure paths)
- PR #7: 15 min (context propagation)
- PR #8: 15 min (events)
- PR #9: 15 min (removal)

**Total**: ~2.5 hours vs many hours for single large PR

---

## Key Differences from V1 Plan

| V1 (Bad) | V2 (Good) |
|----------|-----------|
| PR creates spans, separate PR cleans them | Same PR creates + cleans |
| PR creates API spans, later PR adds ABORTED | Same PR creates + adds DEPARTED/ABORTED |
| 11 PRs with dependencies | 9 PRs, each self-consistent |
| "We'll fix leaks later" | No leaks possible |
| Reviewer must trust future PRs | Reviewer can verify each PR independently |

---

## Full Documentation

See `PR_SPLIT_PLAN_V2.md` for:
- Complete code examples for each PR
- Detailed safety checklists
- Test requirements
- Verification procedures
- Rollback strategies
