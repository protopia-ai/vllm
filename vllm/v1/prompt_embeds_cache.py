# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side cache of prompt-embeds parts.

A multi-turn chat with `prompt_embeds` parts re-sends every earlier part on
every turn. Without this cache the frontend ships the whole joined tensor to
the engine over IPC each time (gigabytes near the context limit), and the
engine hashes every row again for prefix caching.

With `VLLM_PROMPT_EMBEDS_PART_CACHE_GB` > 0:

- The frontend keys each loaded part by a hash of its decoded tensor and sends
  the engine `PromptEmbedsPartRef`s instead of the joined tensor, with data
  only for parts it has not sent that engine before.
- The engine keeps parts in a byte-budgeted LRU store, rebuilds the joined
  tensor from it (zero rows at text positions of mixed prompts are rebuilt,
  not sent), and builds prefix-cache block keys from part keys and row ranges
  instead of hashing rows.
- If the engine evicted a part the frontend assumed it had, the request fails
  with `PromptEmbedsCacheMissError`, a `MultiModalCacheMissError`, and the
  frontend resends it with data. The client never sees the miss.
- The frontend never joins the parts: the joined `prompt_embeds` is a
  zero-memory stand-in of the joined shape, and each part's data is sent from
  the part itself.

With `VLLM_PROMPT_EMBEDS_TRUST_PART_IDS` as well, for deployments where a
gateway in front of vLLM computes each chat part's `id` from its content:

- A part with an `id` and `num_rows` is keyed by its `id`, not by hashing its
  decoded rows, and a part the frontend believes the engine holds is not
  decoded at all: the frontend carries a zero-memory stand-in of its shape.
- If the engine reports a miss for a part that was left undecoded, the
  frontend decodes it before the resend.

Keys and layouts travel on the tensor objects themselves, so any step that
builds a new tensor simply drops them and the request takes the data path.
"""

import asyncio
import bisect
import hashlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import torch
from blake3 import blake3

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.cache import MultiModalCacheMissError
from vllm.v1.utils import tensor_data

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest

logger = init_logger(__name__)

KEY_PREFIX = "prompt_embeds:"
_KEY_ATTR = "_vllm_prompt_embeds_part_key"
_LAYOUT_ATTR = "_vllm_prompt_embeds_layout"
# On a part stand-in: a coroutine function that decodes the real part.
_LOAD_ATTR = "_vllm_prompt_embeds_part_load"
# On a joined stand-in: the part of each layout segment, None for zeros.
_SEGMENTS_ATTR = "_vllm_prompt_embeds_segments"
_BYTES_PER_GB = 1 << 30


class PromptEmbedsPartRef(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
):
    """One part of a request's `prompt_embeds` by key with its rows only when the
    engine may not have them."""

    # None for a run of zero rows at text positions of a mixed prompt.
    key: str | None
    num_rows: int
    data: torch.Tensor | None = None


class PromptEmbedsCacheMissError(MultiModalCacheMissError):
    """The engine has no data for some part keys, the frontend must resend."""


class _Layout(NamedTuple):
    # Part keys in row order, None for a run of zero rows.
    keys: tuple[str | None, ...]
    # Row offsets of each segment in the joined tensor, plus its row count.
    offsets: tuple[int, ...]


def capacity_bytes() -> int:
    return int(envs.VLLM_PROMPT_EMBEDS_PART_CACHE_GB * _BYTES_PER_GB)


def enabled() -> bool:
    return capacity_bytes() > 0


# blake3 releases the GIL, so keys of a request's parts hash concurrently.
_key_pool: ThreadPoolExecutor | None = None


def _part_key(tensor: torch.Tensor) -> str:
    hasher = blake3()
    hasher.update(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
    hasher.update(tensor_data(tensor.contiguous()))
    return KEY_PREFIX + hasher.hexdigest()


async def tag_part(tensor: torch.Tensor) -> None:
    """Key a loaded part by its content off the event loop."""
    global _key_pool
    if not enabled():
        return
    if _key_pool is None:
        _key_pool = ThreadPoolExecutor(4, thread_name_prefix="prompt_embeds_key")
    key = await asyncio.get_running_loop().run_in_executor(_key_pool, _part_key, tensor)
    setattr(tensor, _KEY_ATTR, key)


def trusted_part_key(part_id: str | None) -> str | None:
    """The key of a part whose client-supplied `id` is trusted, else None.

    Trusted only with `VLLM_PROMPT_EMBEDS_TRUST_PART_IDS`, for deployments
    where a gateway in front of vLLM computes each part's `id` from its
    content. vLLM does not check that an `id` matches its `data`, so a client
    that can set `id` directly could bind it to other content.
    """
    if part_id is None or not enabled() or not envs.VLLM_PROMPT_EMBEDS_TRUST_PART_IDS:
        return None
    return f"{KEY_PREFIX}id:{part_id}"


def key_part(tensor: torch.Tensor, key: str) -> None:
    """Key a loaded part by its trusted `id` instead of hashing its rows."""
    setattr(tensor, _KEY_ATTR, key)


def engine_holds(key: str) -> bool:
    """Whether this process's frontend sender believes every engine it has
    sent to holds the part `key`, so the part need not be decoded."""
    return _sender is not None and _sender.holds(key)


def part_stand_in(
    num_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    key: str,
    load: Callable[[], Awaitable[torch.Tensor]],
) -> torch.Tensor:
    """A zero-memory `(num_rows, hidden_size)` tensor for a part the engine
    holds, keyed by `key`. Its rows are never read: the part goes to the
    engine as a ref. `load` decodes the real part if the engine reports a
    miss and the part has to be resent with data."""
    stand_in = torch.zeros(1, hidden_size, dtype=dtype).expand(num_rows, -1)
    setattr(stand_in, _KEY_ATTR, key)
    setattr(stand_in, _LOAD_ATTR, load)
    return stand_in


def _layout_for(
    total_rows: int,
    parts: Sequence[torch.Tensor],
    positions: Sequence[tuple[int, int]],
) -> tuple[_Layout, tuple[torch.Tensor | None, ...]] | None:
    """The layout of `total_rows` rows holding each part at its
    `(start, length)` position and zeros elsewhere, with the part of each
    segment (None for zeros), if every part is keyed."""
    keys: list[str | None] = []
    segments: list[torch.Tensor | None] = []
    offsets = [0]
    for part, (start, length) in zip(parts, positions, strict=True):
        key = getattr(part, _KEY_ATTR, None)
        if key is None:
            return None
        if start > offsets[-1]:
            keys.append(None)
            segments.append(None)
            offsets.append(start)
        keys.append(key)
        segments.append(part)
        offsets.append(start + length)
    if offsets[-1] < total_rows:
        keys.append(None)
        segments.append(None)
        offsets.append(total_rows)
    if not keys:
        return None
    return _Layout(tuple(keys), tuple(offsets)), tuple(segments)


def tag_joined(
    joined: torch.Tensor,
    parts: Sequence[torch.Tensor],
    positions: Sequence[tuple[int, int]],
) -> None:
    """Record that `joined` holds each part at its `(start, length)` position
    and zeros elsewhere, if every part is keyed."""
    found = _layout_for(len(joined), parts, positions)
    if found is not None:
        setattr(joined, _LAYOUT_ATTR, found[0])


def join_lazily(
    total_rows: int,
    parts: Sequence[torch.Tensor],
    positions: Sequence[tuple[int, int]],
) -> torch.Tensor | None:
    """A zero-memory stand-in for the joined `prompt_embeds`, or None when
    the rows must really be joined.

    With a frontend sender in this process and every part keyed, the joined
    rows are never read on the frontend: the sender sends the engine refs and
    the data of each part it has not sent, taken from the parts themselves.
    The stand-in has the joined shape, so length checks work unchanged.
    """
    if _sender is None or not parts:
        return None
    found = _layout_for(total_rows, parts, positions)
    if found is None:
        return None
    layout, segments = found
    stand_in = torch.zeros(1, parts[0].shape[1], dtype=parts[0].dtype)
    stand_in = stand_in.expand(total_rows, -1)
    setattr(stand_in, _LAYOUT_ATTR, layout)
    setattr(stand_in, _SEGMENTS_ATTR, segments)
    return stand_in


def _layout_of(tensor: torch.Tensor | None) -> _Layout | None:
    return None if tensor is None else getattr(tensor, _LAYOUT_ATTR, None)


class PromptEmbedsPartSender:
    """Frontend side: swaps a request's joined `prompt_embeds` for part refs,
    and resends requests the engine reports a miss for.

    Per engine, it remembers the keys it sent with data, under the engine's
    byte budget, and sends those keys without data. That is a guess, not a
    mirror: the engine evicts on its own, and a wrong guess costs one resend.
    """

    _MAX_PENDING = 1024

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._sent: dict[Any, OrderedDict[str, int]] = {}
        self._sent_bytes: dict[Any, int] = {}
        # Requests sent as refs, kept until the engine accepts or rejects them.
        self._pending: OrderedDict[str, EngineCoreRequest] = OrderedDict()

    @classmethod
    def from_env(cls) -> "PromptEmbedsPartSender | None":
        """The sender for this process's frontend, registered so the content
        parser and renderer can skip decoding and joining parts."""
        global _sender
        if not enabled():
            return None
        _sender = cls(capacity_bytes())
        return _sender

    def holds(self, key: str) -> bool:
        """Whether every engine this sender has sent to is believed to hold
        `key`. A request may go to any of them."""
        return bool(self._sent) and all(key in sent for sent in self._sent.values())

    def to_refs(self, request: "EngineCoreRequest", engine: Any) -> "EngineCoreRequest":
        layout = _layout_of(request.prompt_embeds)
        if layout is None:
            return request
        assert request.prompt_embeds is not None
        segments = getattr(request.prompt_embeds, _SEGMENTS_ATTR, None)
        sent = self._sent.setdefault(engine, OrderedDict())
        row_bytes = request.prompt_embeds[:1].nbytes
        refs = []
        for index, (key, start, end) in enumerate(
            zip(layout.keys, layout.offsets, layout.offsets[1:])
        ):
            data = None
            part = (
                request.prompt_embeds[start:end]
                if segments is None
                else segments[index]
            )
            if key is None:
                pass
            elif key in sent:
                sent.move_to_end(key)
            elif hasattr(part, _LOAD_ATTR):
                # Not decoded because the engine was believed to hold it. Sent
                # as a ref anyway: if the engine no longer has it, it reports
                # a miss and `load_parts` decodes it for the resend.
                pass
            else:
                data = part
                self._remember(engine, key, (end - start) * row_bytes)
            refs.append(PromptEmbedsPartRef(key, end - start, data))
        self._pending[request.request_id] = request
        while len(self._pending) > self._MAX_PENDING:
            self._pending.popitem(last=False)
        return msgspec.structs.replace(
            request, prompt_embeds=None, prompt_embeds_parts=refs
        )

    def _remember(self, engine: Any, key: str, nbytes: int) -> None:
        sent = self._sent[engine]
        sent[key] = nbytes
        total = self._sent_bytes.get(engine, 0) + nbytes
        while total > self._capacity and sent:
            total -= sent.popitem(last=False)[1]
        self._sent_bytes[engine] = total

    def take_misses(self, outputs: "EngineCoreOutputs") -> list["EngineCoreRequest"]:
        """Drop prompt-embeds misses from `outputs`, returning the requests
        to resend. Any other output means the engine took the request."""
        resend = []
        kept = []
        for output in outputs.outputs:
            original = self._pending.pop(output.request_id, None)
            missing = [
                key
                for key in output.mm_cache_miss_hashes or ()
                if key.startswith(KEY_PREFIX)
            ]
            if original is None or not missing:
                kept.append(output)
                continue
            for sent in self._sent.values():
                for key in missing:
                    sent.pop(key, None)
            if outputs.finished_requests:
                outputs.finished_requests.discard(output.request_id)
            resend.append(original)
        if resend:
            logger.warning(
                "Engine evicted prompt-embeds parts of %d request(s); "
                "resending them with data.",
                len(resend),
            )
            outputs.outputs = kept
        return resend

    async def load_parts(self, request: "EngineCoreRequest") -> None:
        """Decode the parts of `request` that were left undecoded and that no
        engine is believed to hold any more, before it is resent."""
        joined = request.prompt_embeds
        segments = getattr(joined, _SEGMENTS_ATTR, None)
        layout = _layout_of(joined)
        if segments is None or layout is None:
            return
        loaded = list(segments)
        for index, (key, part) in enumerate(zip(layout.keys, segments)):
            load = getattr(part, _LOAD_ATTR, None)
            if key is None or load is None:
                continue
            if any(key in sent for sent in self._sent.values()):
                continue
            tensor = await load()
            key_part(tensor, key)
            loaded[index] = tensor
        setattr(joined, _SEGMENTS_ATTR, tuple(loaded))


# This process's frontend sender, set by `PromptEmbedsPartSender.from_env`.
_sender: PromptEmbedsPartSender | None = None


class PromptEmbedsPartStore:
    """Engine side: byte-budgeted LRU of parts. Requests keep the tensors they
    resolved, so eviction never pulls rows from under a running request."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._parts: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> torch.Tensor | None:
        part = self._parts.get(key)
        if part is not None:
            self._parts.move_to_end(key)
        return part

    def put(self, key: str, part: torch.Tensor) -> None:
        if part.nbytes > self._capacity:
            return
        if (old := self._parts.pop(key, None)) is not None:
            self._bytes -= old.nbytes
        self._parts[key] = part
        self._bytes += part.nbytes
        while self._bytes > self._capacity:
            self._bytes -= self._parts.popitem(last=False)[1].nbytes

    def resolve(self, request: "EngineCoreRequest") -> None:
        """Replace `request.prompt_embeds_parts` with the joined tensor."""
        refs = request.prompt_embeds_parts
        if refs is None:
            return
        # Look up every referenced part before storing any sent one, so that
        # storing cannot evict a part this request still needs.
        parts = [
            ref.data if ref.data is not None or ref.key is None else self.get(ref.key)
            for ref in refs
        ]
        missing = [
            ref.key for ref, part in zip(refs, parts) if ref.key and part is None
        ]
        if missing:
            raise PromptEmbedsCacheMissError(missing)
        for ref in refs:
            if ref.key is not None and ref.data is not None:
                self.put(ref.key, ref.data)
        some_part = next(part for part in parts if part is not None)
        segments = [
            part
            if part is not None
            else some_part.new_zeros(ref.num_rows, some_part.shape[1])
            for ref, part in zip(refs, parts)
        ]
        joined = segments[0] if len(segments) == 1 else torch.cat(segments)
        offsets = [0]
        for ref in refs:
            offsets.append(offsets[-1] + ref.num_rows)
        setattr(
            joined,
            _LAYOUT_ATTR,
            _Layout(tuple(ref.key for ref in refs), tuple(offsets)),
        )
        request.prompt_embeds = joined
        request.prompt_embeds_parts = None


_store: PromptEmbedsPartStore | None = None


def resolve_parts(request: "EngineCoreRequest") -> None:
    """Engine hook: rebuild a request's `prompt_embeds` from its part refs."""
    global _store
    if request.prompt_embeds_parts is None:
        return
    if _store is None:
        _store = PromptEmbedsPartStore(capacity_bytes())
    _store.resolve(request)


def block_hash(
    prompt_embeds: torch.Tensor, first_row: int, end_row: int
) -> bytes | None:
    """A block's prompt-embeds digest from part keys and row ranges, or None
    when the tensor did not come from parts and its rows must be hashed."""
    layout = _layout_of(prompt_embeds)
    if layout is None:
        return None
    hasher = hashlib.sha256(b"prompt_embeds_parts")
    first_row, end_row = int(first_row), int(end_row)
    index = bisect.bisect_right(layout.offsets, first_row) - 1
    while index < len(layout.keys) and layout.offsets[index] < end_row:
        part_start = layout.offsets[index]
        start = max(first_row, part_start) - part_start
        end = min(end_row, layout.offsets[index + 1]) - part_start
        hasher.update((layout.keys[index] or "zeros").encode())
        hasher.update(start.to_bytes(8, "little") + end.to_bytes(8, "little"))
        index += 1
    return hasher.digest()
