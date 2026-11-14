# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine-side prompt-embeds part cache, through the real wire format."""

import asyncio
import time

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1 import prompt_embeds_cache as pec
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

HIDDEN = 16
ENGINE = b"engine-0"


@pytest.fixture(autouse=True)
def cache_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_PART_CACHE_GB", "1")


def _parts(*rows: int, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    parts = [torch.randn(n, HIDDEN, generator=gen) for n in rows]
    for part in parts:
        asyncio.run(pec.tag_part(part))
    return parts


def _mixed(parts: list[torch.Tensor], gap: int) -> torch.Tensor:
    """A mixed prompt's `prompt_embeds`: each part after `gap` text rows,
    which hold zeros, as the chat renderer builds it."""
    joined = torch.zeros(sum(len(p) + gap for p in parts), HIDDEN)
    positions, start = [], 0
    for part in parts:
        start += gap
        joined[start : start + len(part)] = part
        positions.append((start, len(part)))
        start += len(part)
    pec.tag_joined(joined, parts, positions)
    return joined


def _request(
    request_id: str, parts: list[torch.Tensor], gap: int = 0
) -> EngineCoreRequest:
    joined = _mixed(parts, gap)
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[0] * len(joined),
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=joined,
    )


def _over_the_wire(request: EngineCoreRequest) -> EngineCoreRequest:
    # A zero threshold sends every tensor out of band, as large ones are.
    frames = MsgpackEncoder(size_threshold=0).encode(_with_packed_prompt(request))
    return MsgpackDecoder(EngineCoreRequest).decode(frames)


def _sent_rows(request: EngineCoreRequest) -> list[int]:
    assert request.prompt_embeds is None
    assert request.prompt_embeds_parts is not None
    return [
        len(ref.data) if ref.data is not None else 0
        for ref in request.prompt_embeds_parts
    ]


def test_each_part_crosses_once_and_the_engine_rebuilds_the_prompt():
    sender = pec.PromptEmbedsPartSender(capacity=1 << 30)
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    p1, p2, p3 = _parts(3, 4, 5)

    first = sender.to_refs(_request("a", [p1, p2]), ENGINE)
    assert _sent_rows(first) == [3, 4]
    received = _over_the_wire(first)
    store.resolve(received)
    torch.testing.assert_close(received.prompt_embeds, torch.cat([p1, p2]))

    # The next turn repeats the earlier parts and adds one: only it crosses.
    second = sender.to_refs(_request("b", [p1, p2, p3]), ENGINE)
    assert _sent_rows(second) == [0, 0, 5]
    received = _over_the_wire(second)
    store.resolve(received)
    torch.testing.assert_close(received.prompt_embeds, torch.cat([p1, p2, p3]))
    assert received.prompt_embeds_parts is None


def test_an_evicted_part_is_resent_without_the_client_seeing_the_miss():
    p1, p2 = _parts(3, 4)
    sender = pec.PromptEmbedsPartSender(capacity=1 << 30)
    # The engine only holds one part, so the second evicts the first.
    store = pec.PromptEmbedsPartStore(capacity=p2.nbytes)
    store.resolve(_over_the_wire(sender.to_refs(_request("a", [p1]), ENGINE)))
    store.resolve(_over_the_wire(sender.to_refs(_request("b", [p2]), ENGINE)))

    original = _request("c", [p1, p2])
    with pytest.raises(pec.PromptEmbedsCacheMissError) as miss:
        store.resolve(_over_the_wire(sender.to_refs(original, ENGINE)))

    outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="c",
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                mm_cache_miss_hashes=miss.value.mm_hashes,
            )
        ],
        finished_requests={"c"},
    )
    resend = sender.take_misses(outputs)

    assert resend == [original]
    assert outputs.outputs == [] and outputs.finished_requests == set()
    retried = _over_the_wire(sender.to_refs(resend[0], ENGINE))
    assert _sent_rows(retried) == [3, 0]
    store.resolve(retried)
    torch.testing.assert_close(retried.prompt_embeds, torch.cat([p1, p2]))


def test_text_rows_of_a_mixed_prompt_are_rebuilt_not_sent():
    sender = pec.PromptEmbedsPartSender(capacity=1 << 30)
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    p1, p2 = _parts(3, 4)
    request = _request("a", [p1, p2], gap=2)

    sent = sender.to_refs(request, ENGINE)
    assert _sent_rows(sent) == [0, 3, 0, 4]
    received = _over_the_wire(sent)
    store.resolve(received)
    torch.testing.assert_close(received.prompt_embeds, _mixed([p1, p2], gap=2))


def test_a_rebuilt_tensor_takes_the_data_path():
    """Any step that makes a new tensor drops the parts, and sends data."""
    sender = pec.PromptEmbedsPartSender(capacity=1 << 30)
    request = _request("a", _parts(3, 4))
    request.prompt_embeds = request.prompt_embeds.clone()

    assert sender.to_refs(request, ENGINE) is request


def _block_key(embeds: torch.Tensor, first_row: int, end_row: int) -> bytes:
    key = pec.block_hash(embeds, first_row, end_row)
    assert key is not None
    return key


def test_block_keys_follow_the_parts_not_the_request():
    p1, p2, p3 = _parts(3, 4, 5)
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    sender = pec.PromptEmbedsPartSender(capacity=1 << 30)
    turn1 = _over_the_wire(sender.to_refs(_request("a", [p1, p2]), ENGINE))
    turn2 = _over_the_wire(sender.to_refs(_request("b", [p1, p2, p3]), ENGINE))
    store.resolve(turn1)
    store.resolve(turn2)

    # A block spanning the first two parts keys the same on both turns, so the
    # second turn finds the first turn's KV in the prefix cache.
    assert _block_key(turn1.prompt_embeds, 2, 6) == _block_key(
        turn2.prompt_embeds, 2, 6
    )
    # Other rows, or other parts in the same place, key differently.
    assert _block_key(turn2.prompt_embeds, 2, 6) != _block_key(
        turn2.prompt_embeds, 3, 7
    )
    q1, q2 = _parts(3, 4, seed=1)
    other = _over_the_wire(sender.to_refs(_request("c", [q1, q2]), ENGINE))
    store.resolve(other)
    assert _block_key(other.prompt_embeds, 2, 6) != _block_key(
        turn1.prompt_embeds, 2, 6
    )


def test_disabled_cache_sends_tensors_as_before(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_PART_CACHE_GB", "0")
    assert pec.PromptEmbedsPartSender.from_env() is None
    part = torch.randn(3, HIDDEN)
    asyncio.run(pec.tag_part(part))
    assert pec._layout_of(_mixed([part, part], gap=1)) is None
