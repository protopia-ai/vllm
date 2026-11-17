# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch

TOKEN_BLOCK = 16


@dataclass
class _RequestEmbeds:
    # The request's uncomputed prompt-embeds rows, on the host.
    rows: torch.Tensor
    # Mixed prompts only: each position's row, or -1 for a token-id position,
    # on the host (for picking a step's rows) and on the GPU (for the kernel).
    row_map_np: np.ndarray | None
    row_map: torch.Tensor | None
    # Position of rows[0] (or of row_map[0]) in the prompt.
    offset: int
    # End position of the prompt-embeds rows.
    end: int


def step_row_range(
    num_computed: int,
    num_scheduled: int,
    offset: int,
    end: int,
    row_map_np: np.ndarray | None,
) -> tuple[int, int] | None:
    """Rows a step needs, as [first, last) into the request's rows.

    The step covers positions [num_computed, num_computed + num_scheduled),
    clipped to the prompt-embeds end. Returns None when none of those
    positions take a prompt-embeds row.
    """
    start = num_computed
    stop = min(num_computed + num_scheduled, end)
    if start >= stop:
        return None
    if row_map_np is None:
        return start - offset, stop - offset
    rows = row_map_np[start - offset : stop - offset]
    rows = rows[rows >= 0]
    if rows.size == 0:
        return None
    # Rows of embeds positions increase with position, so the step's rows are
    # one contiguous range.
    return int(rows[0]), int(rows[-1]) + 1


class PromptEmbedsState:
    """GPU-side state for user-provided prompt embeddings.

    Each request's uncomputed embeddings stay on the host. Every step copies
    only the rows that step computes into one preallocated GPU staging buffer
    of `max_num_tokens` rows, so prompt embeddings take a fixed amount of GPU
    memory however many requests are running or how long their prompts are.
    A per-request pointer table (UVA) then lets a single triton kernel overlay
    all scheduled prompt-embeds rows from the staging buffer onto
    `inputs_embeds`, with no python loops on the GPU side. A request's host
    rows are released once its prompt has been computed.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.pin_memory = is_pin_memory_available()

        # req_id -> rows still to be computed. Removed once the prompt has
        # been computed, or when the request finishes or is preempted.
        self.requests: dict[str, _RequestEmbeds] = {}

        # This step's rows for every scheduled request, packed back to back.
        # Allocated here, before memory profiling, so the KV cache budget
        # accounts for it.
        self.staging = torch.empty(
            (max_num_tokens, hidden_size), dtype=dtype, device=device
        )
        self.row_bytes = self.staging.stride(0) * self.staging.element_size()

        # Indexed by req_state index. Stale entries after removal are
        # harmless: add_request rewrites all fields for every index it claims.
        self.embeds_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        self.row_map_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        # Rows start at position `offset`: those below it were prefix-cache
        # hits and are not sent. `embeds_lens` is the end position.
        self.embeds_offsets = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.embeds_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        prompt_embeds = new_req_data.prompt_embeds
        if prompt_embeds is None:
            self.requests.pop(new_req_data.req_id, None)
            self.embeds_lens.np[req_index] = 0
            return

        offset = new_req_data.prompt_embeds_offset
        assert offset <= new_req_data.num_computed_tokens
        # Own host copy: the incoming tensor may share memory that is reused
        # once this step's inputs have been read. Pinned, so the per-step
        # copies to the GPU run asynchronously.
        rows = torch.empty(
            prompt_embeds.shape,
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        rows.copy_(prompt_embeds)
        is_token_ids = new_req_data.prompt_is_token_ids
        row_map_np = None
        row_map = None
        num_positions = rows.shape[0]
        if is_token_ids is not None:
            # Mixed prompts carry only their embeds positions' rows: map each
            # position to its row, or to -1 for a token-id position.
            is_embeds = ~np.asarray(is_token_ids, dtype=np.bool_)
            row_map_np = np.where(
                is_embeds, np.cumsum(is_embeds, dtype=np.int32) - 1, -1
            ).astype(np.int32)
            row_map = async_tensor_h2d(
                row_map_np, device=self.device, dtype=torch.int32
            )
            num_positions = len(is_token_ids)
        self.requests[new_req_data.req_id] = _RequestEmbeds(
            rows, row_map_np, row_map, offset, offset + num_positions
        )
        self.row_map_ptrs.np[req_index] = 0 if row_map is None else row_map.data_ptr()
        self.embeds_offsets.np[req_index] = offset
        self.embeds_lens.np[req_index] = offset + num_positions

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def apply_staged_writes(self) -> None:
        self.row_map_ptrs.copy_to_uva()
        self.embeds_offsets.copy_to_uva()
        self.embeds_lens.copy_to_uva()

    def stage_step_rows(self, input_batch: InputBatch) -> bool:
        """Copy this step's rows to the staging buffer and point each request
        at them. Returns False when no scheduled request needs any rows."""
        if not self.requests:
            return False
        cursor = 0
        done: list[str] = []
        for i, req_id in enumerate(input_batch.req_ids):
            req = self.requests.get(req_id)
            if req is None:
                continue
            num_computed = int(input_batch.num_computed_prefill_tokens_np[i])
            num_scheduled = int(input_batch.num_scheduled_tokens[i])
            row_range = step_row_range(
                num_computed, num_scheduled, req.offset, req.end, req.row_map_np
            )
            if num_computed + num_scheduled >= req.end:
                # This step computes the last prompt position, so the host rows
                # are not needed after it. The copy below holds the pinned
                # block until the copy has run.
                done.append(req_id)
            if row_range is None:
                continue
            first, last = row_range
            n = last - first
            self.staging[cursor : cursor + n].copy_(
                req.rows[first:last], non_blocking=True
            )
            # The kernel reads row r at embeds_ptr + r * row_bytes, with r
            # counted from the request's first row, so shift the pointer back
            # by `first` rows to land row `first` at staging[cursor]. Only
            # rows in [first, last) are read this step.
            req_index = int(input_batch.idx_mapping_np[i])
            ptr = self.staging.data_ptr() + (cursor - first) * self.row_bytes
            self.embeds_ptrs.np[req_index] = ptr % (1 << 64)
            cursor += n
        for req_id in done:
            del self.requests[req_id]
        if cursor == 0:
            return False
        self.embeds_ptrs.copy_to_uva()
        return True

    def apply(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        """Overlay prompt embeddings onto `inputs_embeds` for the batch."""
        if not self.stage_step_rows(input_batch):
            return
        # The kernel reinterprets raw source pointers as inputs_embeds' dtype.
        assert inputs_embeds.dtype == self.dtype
        num_reqs = input_batch.num_reqs
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        grid = (num_reqs, triton.cdiv(max_query_len, TOKEN_BLOCK))
        _apply_prompt_embeds_kernel[grid](
            inputs_embeds,
            inputs_embeds.stride(0),
            self.embeds_ptrs.gpu,
            self.row_map_ptrs.gpu,
            self.embeds_offsets.gpu,
            self.embeds_lens.gpu,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            num_computed_tokens,
            self.hidden_size,
            TOKEN_BLOCK=TOKEN_BLOCK,
            BLOCK_SIZE=1024,
        )


@triton.jit
def _apply_prompt_embeds_kernel(
    inputs_embeds_ptr,
    inputs_embeds_stride,
    embeds_ptrs_ptr,  # uint64 [max_num_reqs], device pointers (0-len = unused)
    row_map_ptrs_ptr,  # uint64 [max_num_reqs], 0 = row is the position
    embeds_offsets_ptr,  # int32 [max_num_reqs], position of row 0
    embeds_lens_ptr,  # int32 [max_num_reqs], end position
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    hidden_size: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    embeds_len = tl.load(embeds_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= embeds_len:
        # No prompt embeds for this request, or they are fully consumed.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    num_rows = tl.minimum(query_end - query_start, embeds_len - num_computed)

    t_start = tl.program_id(1) * TOKEN_BLOCK
    if t_start >= num_rows:
        return

    src_ptr = tl.load(embeds_ptrs_ptr + req_state_idx).to(
        tl.pointer_type(inputs_embeds_ptr.dtype.element_ty)
    )
    offset = tl.load(embeds_offsets_ptr + req_state_idx)
    row_map_int = tl.load(row_map_ptrs_ptr + req_state_idx)
    row_map_ptr = row_map_int.to(tl.pointer_type(tl.int32))

    for t_offset in tl.static_range(TOKEN_BLOCK):
        t = t_start + t_offset
        if t < num_rows:
            src_row = (num_computed + t - offset).to(tl.int64)
            if row_map_int != 0:
                # -1 marks a token-id position, which keeps its base embedding.
                src_row = tl.load(row_map_ptr + src_row).to(tl.int64)
            if src_row >= 0:
                dst_row = (query_start + t).to(tl.int64)
                for h in tl.range(0, hidden_size, BLOCK_SIZE):
                    offs = h + tl.arange(0, BLOCK_SIZE)
                    h_mask = offs < hidden_size
                    row = tl.load(src_ptr + src_row * hidden_size + offs, mask=h_mask)
                    tl.store(
                        inputs_embeds_ptr + dst_row * inputs_embeds_stride + offs,
                        row,
                        mask=h_mask,
                    )
