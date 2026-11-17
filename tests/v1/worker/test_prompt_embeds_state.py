# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Model Runner V2 prompt-embeds overlay (PromptEmbedsState).

Each request's prompt embeddings stay on the host; every step copies the rows
it computes into one GPU staging buffer, and the overlay kernel reads them
through a per-request pointer table and writes them into `inputs_embeds`,
honoring chunked prefill (`num_computed_tokens` offset), the
prompt/decode boundary (rows past the embeds length untouched), the rows not
sent for prefix-cache hits (`prompt_embeds_offset`), and the mixed-mode
`prompt_is_token_ids` mask (token-id rows keep the base embedding). Host rows
are released once a request's prompt has been computed.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from vllm.platforms import current_platform

if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "An accelerator is required for prompt-embeds overlay tests",
        allow_module_level=True,
    )

from vllm.v1.worker.gpu.model_states.prompt_embeds import PromptEmbedsState

HIDDEN = 24
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 64
DEVICE = torch.device(current_platform.device_type)


@dataclass
class _NewReqData:
    req_id: str
    prompt_embeds: torch.Tensor | None
    prompt_is_token_ids: list[bool] | None = None
    prompt_embeds_offset: int = 0
    num_computed_tokens: int = 0


@dataclass
class _Batch:
    req_ids: list[str]
    num_reqs: int
    num_scheduled_tokens: np.ndarray
    idx_mapping_np: np.ndarray
    idx_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    # Filled in by _apply from num_computed.
    num_computed_prefill_tokens_np: np.ndarray | None = None


def _make_state() -> PromptEmbedsState:
    return PromptEmbedsState(
        MAX_NUM_REQS, MAX_NUM_TOKENS, HIDDEN, torch.float32, DEVICE
    )


def _batch(
    num_scheduled: list[int], idx_mapping: list[int], req_ids: list[str]
) -> _Batch:
    query_start_loc = [0]
    for n in num_scheduled:
        query_start_loc.append(query_start_loc[-1] + n)
    return _Batch(
        req_ids=req_ids,
        num_reqs=len(num_scheduled),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        idx_mapping_np=np.array(idx_mapping, dtype=np.intp),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=DEVICE),
    )


def _apply(
    state: PromptEmbedsState,
    batch: _Batch,
    num_computed: list[int],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the overlay on a fresh base buffer; return (result, base)."""
    num_computed_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    for batch_idx, req_index in enumerate(batch.idx_mapping.tolist()):
        num_computed_tokens[req_index] = num_computed[batch_idx]
    # The model runner's host-side prefill progress is the computed-token
    # count clipped to the prompt length. The state compares it only with the
    # prompt-embeds end, so the unclipped count picks the same rows.
    batch.num_computed_prefill_tokens_np = np.array(num_computed, dtype=np.int32)
    base = torch.randn(num_tokens, HIDDEN, dtype=torch.float32, device=DEVICE)
    inputs_embeds = base.clone()
    state.apply(batch, num_computed_tokens, inputs_embeds)
    torch.accelerator.synchronize()
    return inputs_embeds, base


def test_overlay_chunked_prefill_and_decode():
    """Rows within the embeds range come from prompt_embeds at the
    num_computed offset; requests without embeds and requests past their
    embeds length (decode) keep the base embedding."""
    state = _make_state()
    embeds_a = torch.randn(6, HIDDEN, dtype=torch.float32)
    embeds_b = torch.randn(5, HIDDEN, dtype=torch.float32)
    state.add_request(0, _NewReqData("a", embeds_a))
    state.add_request(1, _NewReqData("b", embeds_b))
    state.add_request(2, _NewReqData("c", None))
    state.apply_staged_writes()

    # a: chunk [2, 6) of its embeds; b: fully decoded; c: no embeds.
    batch = _batch(
        num_scheduled=[4, 1, 3], idx_mapping=[0, 1, 2], req_ids=["a", "b", "c"]
    )
    out, base = _apply(state, batch, num_computed=[2, 7, 1], num_tokens=8)

    torch.testing.assert_close(out[0:4], embeds_a[2:6].to(DEVICE))
    torch.testing.assert_close(out[4:8], base[4:8])


def test_overlay_clamps_to_embeds_length():
    """A window straddling the end of the prompt embeds writes only the
    in-range rows (e.g. final prefill chunk + sampled token)."""
    state = _make_state()
    embeds = torch.randn(4, HIDDEN, dtype=torch.float32)
    state.add_request(3, _NewReqData("a", embeds))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[3], idx_mapping=[3], req_ids=["a"])
    out, base = _apply(state, batch, num_computed=[2], num_tokens=3)

    torch.testing.assert_close(out[0:2], embeds[2:4].to(DEVICE))
    torch.testing.assert_close(out[2:3], base[2:3])


def test_overlay_respects_is_token_ids_mask():
    """Mixed mode: positions marked as real token ids keep the base
    embedding; embed positions read the compact rows (one per embed position,
    in order), also across a chunked-prefill boundary."""
    state = _make_state()
    is_token_ids = [True, False, False, True, False]
    embeds = torch.randn(is_token_ids.count(False), HIDDEN, dtype=torch.float32)
    state.add_request(0, _NewReqData("a", embeds, is_token_ids))
    state.apply_staged_writes()

    embeds_gpu = embeds.to(DEVICE)
    rows = iter(embeds_gpu)
    expected_rows = [None if is_token else next(rows) for is_token in is_token_ids]
    for num_computed, num_scheduled in [(0, 2), (2, 3)]:
        batch = _batch(num_scheduled=[num_scheduled], idx_mapping=[0], req_ids=["a"])
        out, base = _apply(
            state, batch, num_computed=[num_computed], num_tokens=num_scheduled
        )
        for t in range(num_scheduled):
            expected = expected_rows[num_computed + t]
            torch.testing.assert_close(
                out[t], base[t] if expected is None else expected
            )


def test_index_reuse_clears_stale_entry():
    """A request added at a previously-used index without embeds must not
    inherit the prior occupant's pointer-table entry."""
    state = _make_state()
    state.add_request(0, _NewReqData("a", torch.randn(4, HIDDEN, dtype=torch.float32)))
    # A second live embeds request so the kernel actually launches (the
    # overlay is skipped entirely when no request holds embeds).
    other = torch.randn(2, HIDDEN, dtype=torch.float32)
    state.add_request(1, _NewReqData("other", other))
    state.apply_staged_writes()
    state.remove_request("a")
    state.add_request(0, _NewReqData("b", None))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[2, 2], idx_mapping=[0, 1], req_ids=["b", "other"])
    out, base = _apply(state, batch, num_computed=[0, 0], num_tokens=4)

    torch.testing.assert_close(out[0:2], base[0:2])
    torch.testing.assert_close(out[2:4], other.to(DEVICE))


@pytest.mark.parametrize("is_token_ids", [None, [False, True, False]])
def test_overlay_prompt_embeds_offset(is_token_ids):
    """After a prefix-cache hit only the positions from `prompt_embeds_offset`
    on are sent: their rows (compact for a mixed prompt) and their mask."""
    state = _make_state()
    num_rows = 3 if is_token_ids is None else is_token_ids.count(False)
    embeds = torch.randn(num_rows, HIDDEN, dtype=torch.float32)
    state.add_request(
        0,
        _NewReqData(
            "a",
            embeds,
            is_token_ids,
            prompt_embeds_offset=4,
            num_computed_tokens=4,
        ),
    )
    state.apply_staged_writes()

    # Chunks [4, 6) then [6, 7) plus a decode token past the prompt.
    out, base = _apply(state, _batch([2], [0], ["a"]), num_computed=[4], num_tokens=2)
    out2, base2 = _apply(state, _batch([2], [0], ["a"]), num_computed=[6], num_tokens=2)

    # Positions 4, 5, 6: a mixed prompt's position 5 is a token id.
    rows = embeds.to(DEVICE)
    if is_token_ids is None:
        torch.testing.assert_close(out, rows[0:2])
        torch.testing.assert_close(out2[0], rows[2])
    else:
        torch.testing.assert_close(out[0], rows[0])
        torch.testing.assert_close(out[1], base[1])
        torch.testing.assert_close(out2[0], rows[1])
    torch.testing.assert_close(out2[1], base2[1])


def test_overlay_fully_cached_prompt_embeds():
    """A request resumed past its prompt gets no rows and keeps the base
    embedding."""
    state = _make_state()
    state.add_request(
        0,
        _NewReqData(
            "a",
            torch.empty(0, HIDDEN),
            prompt_embeds_offset=6,
            num_computed_tokens=9,
        ),
    )
    # A second live embeds request so the kernel launches.
    other = torch.randn(2, HIDDEN, dtype=torch.float32)
    state.add_request(1, _NewReqData("other", other))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[1, 2], idx_mapping=[0, 1], req_ids=["a", "other"])
    out, base = _apply(state, batch, num_computed=[9, 0], num_tokens=3)

    torch.testing.assert_close(out[0:1], base[0:1])
    torch.testing.assert_close(out[1:3], other.to(DEVICE))


def test_host_rows_released_after_prompt():
    """Host rows are held while the prompt is being computed and released
    after the step that computes its last position."""
    state = _make_state()
    embeds = torch.randn(5, HIDDEN, dtype=torch.float32)
    state.add_request(0, _NewReqData("a", embeds))
    state.apply_staged_writes()

    out, _ = _apply(state, _batch([3], [0], ["a"]), num_computed=[0], num_tokens=3)
    torch.testing.assert_close(out, embeds[0:3].to(DEVICE))
    assert "a" in state.requests

    out, base = _apply(state, _batch([3], [0], ["a"]), num_computed=[3], num_tokens=3)
    torch.testing.assert_close(out[0:2], embeds[3:5].to(DEVICE))
    torch.testing.assert_close(out[2:3], base[2:3])
    assert "a" not in state.requests

    # Decode after the prompt keeps the base embedding.
    out, base = _apply(state, _batch([1], [0], ["a"]), num_computed=[6], num_tokens=1)
    torch.testing.assert_close(out, base)


class _Req:
    """A request in the randomized test: its prompt, rows and progress."""

    def __init__(self, rng: np.random.Generator, req_id: str, kind: str):
        self.req_id = req_id
        self.prompt_len = int(rng.integers(1, 200))
        # Positions below `offset` were prefix-cache hits and are not sent.
        self.offset = int(rng.integers(0, self.prompt_len)) if rng.random() < 0.5 else 0
        self.num_computed = self.offset
        self.is_token_ids: list[bool] | None = None
        self.rows: torch.Tensor | None = None
        num_positions = self.prompt_len - self.offset
        if kind == "plain":
            self.rows = torch.randn(num_positions, HIDDEN, dtype=torch.float32)
        elif kind == "mixed":
            self.is_token_ids = (rng.random(num_positions) < 0.4).tolist()
            num_rows = num_positions - sum(self.is_token_ids)
            self.rows = torch.randn(num_rows, HIDDEN, dtype=torch.float32)
        self.decode_steps_left = int(rng.integers(1, 4))

    def new_req_data(self) -> _NewReqData:
        return _NewReqData(
            self.req_id,
            self.rows,
            self.is_token_ids,
            prompt_embeds_offset=self.offset,
            num_computed_tokens=self.num_computed,
        )

    def expected_row(self, pos: int) -> torch.Tensor | None:
        """The prompt-embeds row for a position, or None for the base row."""
        if self.rows is None or pos < self.offset or pos >= self.prompt_len:
            return None
        rel = pos - self.offset
        if self.is_token_ids is None:
            return self.rows[rel]
        if self.is_token_ids[rel]:
            return None
        return self.rows[sum(1 for x in self.is_token_ids[:rel] if not x)]


@pytest.mark.parametrize("seed", range(4))
def test_overlay_matches_reference_randomized(seed):
    """Randomized steps against a reference built in Python: chunked prefill
    under the staging buffer's token budget, prefix-cache offsets, mixed
    prompts, requests without embeds, decode steps, preemption and slot reuse.
    Host rows are held only for requests still in prefill."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    state = _make_state()
    num_computed_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    free_slots = list(range(MAX_NUM_REQS))
    slot_of: dict[str, int] = {}
    running: list[_Req] = []
    next_id = 0
    for step in range(40):
        # Admit new requests into free slots.
        while free_slots and rng.random() < 0.5:
            kind = str(rng.choice(["plain", "plain", "mixed", "none"]))
            req = _Req(rng, f"r{next_id}", kind)
            next_id += 1
            slot = free_slots.pop(int(rng.integers(len(free_slots))))
            slot_of[req.req_id] = slot
            running.append(req)
            state.add_request(slot, req.new_req_data())
        state.apply_staged_writes()
        if not running:
            continue

        # Prefilling requests share the token budget in random chunks;
        # requests past their prompt take one decode token.
        budget = MAX_NUM_TOKENS
        batch: list[tuple[_Req, int]] = []
        for req in running:
            if budget == 0:
                break
            if req.num_computed < req.prompt_len:
                remaining = min(budget, req.prompt_len - req.num_computed)
                n = int(rng.integers(1, remaining + 1))
            else:
                n = 1
            batch.append((req, n))
            budget -= n

        idx_np = np.array([slot_of[r.req_id] for r, _ in batch], dtype=np.intp)
        num_sched = np.array([n for _, n in batch], dtype=np.int32)
        qsl = np.concatenate([[0], np.cumsum(num_sched)]).astype(np.int32)
        for r, _ in batch:
            num_computed_gpu[slot_of[r.req_id]] = r.num_computed
        input_batch = _Batch(
            req_ids=[r.req_id for r, _ in batch],
            num_reqs=len(batch),
            num_scheduled_tokens=num_sched,
            idx_mapping_np=idx_np,
            idx_mapping=torch.from_numpy(idx_np.astype(np.int64)).to(DEVICE),
            query_start_loc=torch.from_numpy(qsl).to(DEVICE),
            num_computed_prefill_tokens_np=np.array(
                [min(r.num_computed, r.prompt_len) for r, _ in batch],
                dtype=np.int32,
            ),
        )
        num_tokens = int(qsl[-1])
        base = torch.randn(num_tokens, HIDDEN, dtype=torch.float32, device=DEVICE)
        out = base.clone()
        state.apply(input_batch, num_computed_gpu, out)

        expected = base.clone()
        for (r, n), q0 in zip(batch, qsl[:-1]):
            for t in range(n):
                row = r.expected_row(r.num_computed + t)
                if row is not None:
                    expected[q0 + t] = row.to(DEVICE)
        torch.accelerator.synchronize()
        assert torch.equal(out, expected), f"step {step}"

        # Advance, then finish or randomly preempt some requests.
        for r, n in batch:
            r.num_computed += n
            if r.num_computed > r.prompt_len:
                r.decode_steps_left -= 1
        for r in list(running):
            finished = r.num_computed > r.prompt_len and r.decode_steps_left <= 0
            if finished or rng.random() < 0.03:
                running.remove(r)
                state.remove_request(r.req_id)
                free_slots.append(slot_of.pop(r.req_id))

        in_prefill = {
            r.req_id
            for r in running
            if r.rows is not None and r.num_computed < r.prompt_len
        }
        assert set(state.requests) == in_prefill, f"step {step}"
