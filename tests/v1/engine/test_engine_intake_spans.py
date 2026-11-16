# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine intake stages recorded as spans for requests whose trace is sampled."""

import time
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1 import prompt_embeds_cache as pec
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import core as engine_core
from vllm.v1.engine.core import EngineCore

SAMPLED = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
UNSAMPLED = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"}
HIDDEN = 16

# CPU-only unit tests: no GPU or distributed state to clean up afterwards.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("headers", "sampled"),
    [
        (SAMPLED, True),
        (UNSAMPLED, False),
        (None, False),
        ({}, False),
        ({"traceparent": "garbage"}, False),
        ({"traceparent": "00-abc-def-zz"}, False),
    ],
)
def test_trace_sampled_reads_the_traceparent_flag(headers, sampled):
    assert engine_core._trace_sampled(headers) is sampled


@pytest.fixture
def recorded(monkeypatch) -> list[tuple]:
    """Spans passed to instrument_manual, as (name, start, end, attributes)."""
    spans: list[tuple] = []
    monkeypatch.setattr(engine_core, "extract_trace_context", lambda headers: "context")

    def record(name, start, end, attributes, context=None):
        assert context == "context"
        spans.append((name, start, end, attributes))

    monkeypatch.setattr(engine_core, "instrument_manual", record)
    return spans


def test_emit_records_each_stage_in_order(recorded):
    parts_attributes = {"vllm.prompt_embeds.parts": 3}
    intake = {
        "recv": 1,
        "decoded": 2,
        "resolve_start": 3,
        "resolve_end": 4,
        "build_start": 5,
        "build_end": 6,
        "put": 7,
        "dequeued": 8,
        "added": 9,
        "parts_attributes": parts_attributes,
    }
    engine_core._emit_engine_intake_spans(
        SimpleNamespace(trace_headers=SAMPLED), intake
    )
    assert recorded == [
        ("vllm.engine_core.receive_decode", 1, 2, None),
        ("vllm.engine_core.resolve_prompt_embeds_parts", 3, 4, parts_attributes),
        ("vllm.engine_core.build_request", 5, 6, None),
        ("vllm.engine_core.input_queue_wait", 7, 8, None),
        ("vllm.engine_core.scheduler_add", 8, 9, None),
    ]


def test_emit_skips_stages_missing_a_timestamp(recorded):
    engine_core._emit_engine_intake_spans(
        SimpleNamespace(trace_headers=SAMPLED),
        {"build_start": 5, "build_end": 6, "put": 7},
    )
    assert [span[0] for span in recorded] == ["vllm.engine_core.build_request"]


def test_emit_records_nothing_without_a_trace_context(monkeypatch, recorded):
    monkeypatch.setattr(engine_core, "extract_trace_context", lambda headers: None)
    engine_core._emit_engine_intake_spans(
        SimpleNamespace(trace_headers=SAMPLED), {"recv": 1, "decoded": 2}
    )
    assert recorded == []


def test_emit_does_not_raise_when_tracing_fails(monkeypatch):
    def fail(headers):
        raise RuntimeError("tracer down")

    monkeypatch.setattr(engine_core, "extract_trace_context", fail)
    engine_core._emit_engine_intake_spans(
        SimpleNamespace(trace_headers=SAMPLED), {"recv": 1, "decoded": 2}
    )


@pytest.fixture
def part_cache(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_EMBEDS_PART_CACHE_GB", "1")
    monkeypatch.setattr(pec, "_store", None)


def _engine() -> SimpleNamespace:
    """The attributes preprocess_add_request reads, for a text-only model
    without watermarking, prefix hashing or structured output."""
    return SimpleNamespace(
        mm_receiver_cache=None,
        vllm_config=SimpleNamespace(watermark_config=None),
        prefix_hash_config=None,
        request_block_hasher=None,
        structured_output_manager=None,
    )


def _request(trace_headers, parts) -> EngineCoreRequest:
    rows = sum(part.num_rows for part in parts)
    return EngineCoreRequest(
        request_id="request-0",
        prompt_token_ids=[0] * rows,
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        trace_headers=trace_headers,
        prompt_embeds_parts=parts,
    )


def _parts() -> list[pec.PromptEmbedsPartRef]:
    return [
        pec.PromptEmbedsPartRef("a", 4, torch.zeros(4, HIDDEN, dtype=torch.bfloat16)),
        pec.PromptEmbedsPartRef("b", 6, torch.zeros(6, HIDDEN, dtype=torch.bfloat16)),
    ]


def test_preprocess_times_a_sampled_request(part_cache):
    req, _ = EngineCore.preprocess_add_request(_engine(), _request(SAMPLED, _parts()))
    intake = req.engine_intake_ns
    assert intake is not None
    assert (
        intake["resolve_start"]
        <= intake["resolve_end"]
        <= intake["build_start"]
        <= intake["build_end"]
    )
    assert intake["parts_attributes"] == {
        "vllm.prompt_embeds.parts": 2,
        "vllm.prompt_embeds.parts_sent": 2,
        "vllm.prompt_embeds.rows": 10,
        "vllm.prompt_embeds.bytes": 10 * HIDDEN * 2,
    }


def test_preprocess_leaves_an_unsampled_request_untimed(part_cache):
    req, _ = EngineCore.preprocess_add_request(_engine(), _request(UNSAMPLED, _parts()))
    assert req.engine_intake_ns is None
