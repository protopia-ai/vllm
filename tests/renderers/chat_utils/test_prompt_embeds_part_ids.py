# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat `prompt_embeds` parts with trusted `id` and `num_rows`: keyed by
`id`, and not decoded when the engine holds them."""

import io
from collections import OrderedDict
from typing import Final
from unittest import mock

import pybase64 as base64
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import parse_chat_messages_async
from vllm.v1 import prompt_embeds_cache as pec

_HIDDEN: Final[int] = 8
_DTYPE: Final[torch.dtype] = torch.float32
_ENGINE: Final[bytes] = b"engine-0"


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


def _model_config() -> mock.MagicMock:
    mc = mock.MagicMock()
    mc.enable_prompt_embeds = True
    mc.multimodal_config = None
    mc.allowed_local_media_path = None
    mc.allowed_media_domains = None
    mc.is_multimodal_model = False
    mc.get_hidden_size.return_value = _HIDDEN
    mc.hf_text_config = object()
    mc.dtype = _DTYPE
    return mc


def _encode(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode()


async def _parse(*parts: dict) -> list[torch.Tensor]:
    messages = [{"role": "user", "content": list(parts)}]
    _, mm_data, _ = await parse_chat_messages_async(
        messages, _model_config(), content_format="openai"
    )
    assert mm_data is not None
    return mm_data["prompt_embeds"]


def _part(data: str, part_id: str | None = None, num_rows=None) -> dict:
    part: dict = {"type": "prompt_embeds", "data": data}
    if part_id is not None:
        part["id"] = part_id
    if num_rows is not None:
        part["num_rows"] = num_rows
    return part


def _hold(sender: pec.PromptEmbedsPartSender, part_id: str) -> None:
    key = pec.trusted_part_key(part_id)
    assert key is not None
    sender._sent.setdefault(_ENGINE, OrderedDict())[key] = 1


@pytest.mark.asyncio
async def test_a_part_the_engine_holds_is_not_decoded(sender):
    _hold(sender, "blake3:aa")
    # Not a tensor at all: decoding it would fail.
    (embeds,) = await _parse(_part("bm90IGEgdGVuc29y", "blake3:aa", 5))

    assert embeds.shape == (5, _HIDDEN) and embeds.dtype == _DTYPE
    assert embeds.untyped_storage().nbytes() == _HIDDEN * 4
    assert getattr(embeds, pec._KEY_ATTR) == pec.trusted_part_key("blake3:aa")


@pytest.mark.asyncio
async def test_a_new_part_is_decoded_and_keyed_by_its_id(sender):
    tensor = torch.randn(3, _HIDDEN, dtype=_DTYPE)
    (embeds,) = await _parse(_part(_encode(tensor), "blake3:bb", 3))

    torch.testing.assert_close(embeds, tensor)
    assert getattr(embeds, pec._KEY_ATTR) == pec.trusted_part_key("blake3:bb")


@pytest.mark.asyncio
async def test_untrusted_ids_are_ignored(sender, monkeypatch):
    _hold(sender, "blake3:aa")
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_TRUST_PART_IDS", "0")
    tensor = torch.randn(3, _HIDDEN, dtype=_DTYPE)
    (embeds,) = await _parse(_part(_encode(tensor), "blake3:aa", 3))

    torch.testing.assert_close(embeds, tensor)
    assert getattr(embeds, pec._KEY_ATTR) == pec._part_key(tensor)


@pytest.mark.asyncio
async def test_num_rows_must_match_the_data(sender):
    tensor = torch.randn(3, _HIDDEN, dtype=_DTYPE)
    with pytest.raises(VLLMValidationError, match="`num_rows` 4"):
        await _parse(_part(_encode(tensor), "blake3:bb", 4))


@pytest.mark.asyncio
@pytest.mark.parametrize("num_rows", [None, 0, "3", True])
async def test_an_id_needs_a_positive_num_rows(sender, num_rows):
    tensor = torch.randn(3, _HIDDEN, dtype=_DTYPE)
    with pytest.raises(VLLMValidationError, match="needs `num_rows`"):
        await _parse(_part(_encode(tensor), "blake3:bb", num_rows))
