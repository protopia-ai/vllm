# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-embeds parts keyed by trusted ids: parts the engine holds are not
decoded, and the frontend never joins the rows."""

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
def trusted_ids(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_PART_CACHE_GB", "1")
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_TRUST_PART_IDS", "1")
    monkeypatch.setattr(pec, "_sender", None)


@pytest.fixture
def sender() -> pec.PromptEmbedsPartSender:
    sender = pec.PromptEmbedsPartSender.from_env()
    assert sender is not None
    return sender


def _decoded(name: str, rows: int, seed: int) -> torch.Tensor:
    """A decoded part keyed by its id, as the content parser keys it."""
    part = torch.randn(rows, HIDDEN, generator=torch.Generator().manual_seed(seed))
    key = pec.trusted_part_key(name)
    assert key is not None
    pec.key_part(part, key)
    return part


def _held(name: str, rows: int, real: torch.Tensor, loads: list[str]):
    """A part left undecoded because the engine holds it; `loads` records
    every decode of it."""
    key = pec.trusted_part_key(name)
    assert key is not None

    async def load() -> torch.Tensor:
        loads.append(name)
        return real.clone()

    return pec.part_stand_in(rows, HIDDEN, real.dtype, key, load)


def _positions(parts: list[torch.Tensor], gap: int) -> list[tuple[int, int]]:
    positions, start = [], 0
    for part in parts:
        start += gap
        positions.append((start, len(part)))
        start += len(part)
    return positions


def _really_joined(parts: list[torch.Tensor], gap: int) -> torch.Tensor:
    joined = torch.zeros(sum(len(p) + gap for p in parts), HIDDEN)
    for part, (start, length) in zip(parts, _positions(parts, gap)):
        joined[start : start + length] = part
    return joined


def _request(
    request_id: str, parts: list[torch.Tensor], gap: int = 0
) -> EngineCoreRequest:
    total = sum(len(p) + gap for p in parts)
    joined = pec.join_lazily(total, parts, _positions(parts, gap))
    assert joined is not None
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[0] * total,
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


def _miss(request_id: str, error: pec.PromptEmbedsCacheMissError):
    return EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id=request_id,
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                mm_cache_miss_hashes=error.mm_hashes,
            )
        ],
        finished_requests={request_id},
    )


def test_ids_are_trusted_only_when_configured(monkeypatch):
    assert pec.trusted_part_key("blake3:ab") == "prompt_embeds:id:blake3:ab"
    assert pec.trusted_part_key(None) is None
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_TRUST_PART_IDS", "0")
    assert pec.trusted_part_key("blake3:ab") is None
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_TRUST_PART_IDS", "1")
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_PART_CACHE_GB", "0")
    assert pec.trusted_part_key("blake3:ab") is None


def test_the_frontend_never_joins_the_rows(sender):
    p1, p2 = _decoded("a", 3, seed=0), _decoded("b", 4, seed=1)
    request = _request("r1", [p1, p2], gap=2)

    # Joined shape for length checks, but one row of storage.
    assert request.prompt_embeds.shape == (11, HIDDEN)
    assert request.prompt_embeds.untyped_storage().nbytes() == HIDDEN * 4
    assert request.prompt_embeds.cpu() is request.prompt_embeds

    sent = sender.to_refs(request, ENGINE)
    assert _sent_rows(sent) == [0, 3, 0, 4]
    received = _over_the_wire(sent)
    pec.PromptEmbedsPartStore(capacity=1 << 30).resolve(received)
    torch.testing.assert_close(received.prompt_embeds, _really_joined([p1, p2], 2))


def test_no_lazy_join_without_a_sender_or_with_an_unkeyed_part(monkeypatch):
    p1 = _decoded("a", 3, seed=0)
    unkeyed = torch.randn(4, HIDDEN)
    assert pec.join_lazily(7, [p1], [(0, 3)]) is None
    pec.PromptEmbedsPartSender.from_env()
    assert pec.join_lazily(7, [p1, unkeyed], [(0, 3), (3, 4)]) is None


def test_held_parts_are_not_decoded_and_go_as_refs(sender):
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    p1, p2 = _decoded("a", 3, seed=0), _decoded("b", 4, seed=1)
    store.resolve(_over_the_wire(sender.to_refs(_request("r1", [p1, p2]), ENGINE)))
    key_a, key_c = pec.trusted_part_key("a"), pec.trusted_part_key("c")
    assert key_a is not None and key_c is not None
    assert pec.engine_holds(key_a) and not pec.engine_holds(key_c)

    # The next turn: the content parser leaves both earlier parts undecoded.
    loads: list[str] = []
    p3 = _decoded("c", 5, seed=2)
    turn2 = [_held("a", 3, p1, loads), _held("b", 4, p2, loads), p3]
    sent = sender.to_refs(_request("r2", turn2), ENGINE)

    assert _sent_rows(sent) == [0, 0, 5]
    assert loads == []
    received = _over_the_wire(sent)
    store.resolve(received)
    torch.testing.assert_close(received.prompt_embeds, torch.cat([p1, p2, p3]))


def test_an_evicted_undecoded_part_is_decoded_for_the_resend(sender):
    p1, p2 = _decoded("a", 3, seed=0), _decoded("b", 4, seed=1)
    # The engine holds one part at a time, so the second evicts the first.
    store = pec.PromptEmbedsPartStore(capacity=p2.nbytes)
    store.resolve(_over_the_wire(sender.to_refs(_request("r1", [p1]), ENGINE)))
    store.resolve(_over_the_wire(sender.to_refs(_request("r2", [p2]), ENGINE)))

    loads: list[str] = []
    original = _request("r3", [_held("a", 3, p1, loads), _held("b", 4, p2, loads)])
    with pytest.raises(pec.PromptEmbedsCacheMissError) as miss:
        store.resolve(_over_the_wire(sender.to_refs(original, ENGINE)))
    resend = sender.take_misses(_miss("r3", miss.value))
    assert resend == [original]

    # Only the part the engine lost is decoded, then it crosses with data.
    asyncio.run(sender.load_parts(resend[0]))
    assert loads == ["a"]
    retried = _over_the_wire(sender.to_refs(resend[0], ENGINE))
    assert _sent_rows(retried) == [3, 0]
    store.resolve(retried)
    torch.testing.assert_close(retried.prompt_embeds, torch.cat([p1, p2]))


def test_an_undecoded_part_the_sender_forgot_still_goes_as_a_ref():
    """Between parsing and sending, other requests can push a held part out
    of the sender's record. It still goes as a ref, and is not recorded as
    sent, since no data went."""
    p1 = _decoded("a", 3, seed=0)
    p2 = _decoded("b", 3, seed=1)
    # Room in the sender's record for one part.
    sender = pec.PromptEmbedsPartSender(capacity=p1.nbytes)
    pec._sender = sender
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    store.resolve(_over_the_wire(sender.to_refs(_request("r1", [p1]), ENGINE)))
    loads: list[str] = []
    held = _held("a", 3, p1, loads)
    store.resolve(_over_the_wire(sender.to_refs(_request("r2", [p2]), ENGINE)))

    sent = sender.to_refs(_request("r3", [held]), ENGINE)
    assert _sent_rows(sent) == [0]
    key_a = pec.trusted_part_key("a")
    assert key_a is not None and not pec.engine_holds(key_a)
    received = _over_the_wire(sent)
    store.resolve(received)
    torch.testing.assert_close(received.prompt_embeds, p1)
    assert loads == []


def test_a_part_keys_the_same_blocks_however_it_arrived(sender):
    store = pec.PromptEmbedsPartStore(capacity=1 << 30)
    p1, p2 = _decoded("a", 3, seed=0), _decoded("b", 4, seed=1)
    turn1 = _over_the_wire(sender.to_refs(_request("r1", [p1, p2]), ENGINE))
    store.resolve(turn1)
    loads: list[str] = []
    turn2 = [_held("a", 3, p1, loads), _held("b", 4, p2, loads)]
    turn2 = _over_the_wire(sender.to_refs(_request("r2", turn2), ENGINE))
    store.resolve(turn2)

    assert pec.block_hash(turn1.prompt_embeds, 2, 6) == pec.block_hash(
        turn2.prompt_embeds, 2, 6
    )
