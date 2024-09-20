from typing import (TYPE_CHECKING, Generic, Iterable, List, Literal, Optional,
                    Tuple, Union)

import torch
from typing_extensions import NotRequired, TypedDict, TypeVar

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class EmbedsPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_embeds: torch.Tensor
    """Embeddings of the prompt to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


SingletonPrompt = Union[str, TextPrompt, TokensPrompt, EmbedsPrompt]
"""
Set of possible schemas for a single LLM input:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- Embeddings of a prompt (:class:`EmbedsPrompt`)

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
prompts explicitly, i.e. :class:`ExplicitEncoderDecoderPrompt`

A prompt of type :class:`SingletonPrompt` may be employed
as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
more than one prompt, i.e. :class:`ExplicitEncoderDecoderPrompt`
"""

_T1_co = TypeVar("_T1_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)
_T2_co = TypeVar("_T2_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)


# TODO: Make fields ReadOnly once mypy supports it
class ExplicitEncoderDecoderPrompt(TypedDict, Generic[_T1_co, _T2_co]):
    """
    Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively,
    may formatted according to any of the
    :class:`SingletonPrompt` schemas, and are not
    required to have the same schema.

    Only the encoder prompt may have multi-modal data.

    Note that an :class:`ExplicitEncoderDecoderPrompt` may not
    be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure themselves must be
    :class:`SingletonPrompt` instances.
    """

    encoder_prompt: _T1_co

    decoder_prompt: Optional[_T2_co]


PromptType = Union[SingletonPrompt, ExplicitEncoderDecoderPrompt]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  (:class:`ExplicitEncoderDecoderPrompt`)
"""


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


def token_inputs(
    prompt_token_ids: List[int],
    prompt: Optional[str] = None,
    multi_modal_data: Optional["MultiModalDataDict"] = None,
) -> TokenInputs:
    """Construct :class:`TokenInputs` from optional values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if multi_modal_data is not None:
        inputs["multi_modal_data"] = multi_modal_data

    return inputs


class EmbedInputs(TypedDict):
    """Represents embedding-based inputs."""

    type: Literal["embed"]
    """The type of inputs."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


def embed_inputs(
    prompt_embeds: torch.Tensor,
    multi_modal_data: Optional["MultiModalDataDict"] = None,
) -> EmbedInputs:
    """Construct :class:`EmbedInputs` from optional values."""
    inputs = EmbedInputs(type="embed", prompt_embeds=prompt_embeds)

    if multi_modal_data is not None:
        inputs["multi_modal_data"] = multi_modal_data

    return inputs


DecoderOnlyInputs = Union[TokenInputs, EmbedInputs]
"""
The inputs in :class:`~vllm.LLMEngine` before they are
passed to the model executor.

This specifies the data required for decoder-only models.
"""


class EmptyInputs(TypedDict):
    """Represents empty inputs."""

    type: Literal["empty"]
    """The type of inputs."""


def empty_inputs() -> EmptyInputs:
    """Construct :class:`EmptyInputs` from optional values."""
    return EmptyInputs(type="empty")


class EncoderDecoderInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.

    This specifies the required data for encoder-decoder models.
    """

    encoder: TokenInputs
    """The inputs for the encoder portion."""

    decoder: Union[EmptyInputs, TokenInputs]
    """The inputs for the decoder portion."""


_T1 = TypeVar("_T1", bound=SingletonPrompt, default=SingletonPrompt)
_T2 = TypeVar("_T2", bound=SingletonPrompt, default=SingletonPrompt)


def build_explicit_enc_dec_prompt(
    encoder_prompt: _T1,
    decoder_prompt: Optional[_T2],
) -> ExplicitEncoderDecoderPrompt[_T1, _T2]:
    return ExplicitEncoderDecoderPrompt(encoder_prompt=encoder_prompt,
                                        decoder_prompt=decoder_prompt)


def zip_enc_dec_prompts(
    enc_prompts: Iterable[_T1],
    dec_prompts: Iterable[Optional[_T2]],
) -> List[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    :class:`ExplicitEncoderDecoderPrompt` instances.
    """
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt)
        for (encoder_prompt, decoder_prompt) in zip(enc_prompts, dec_prompts)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: Iterable[ExplicitEncoderDecoderPrompt[_T1, _T2]],
) -> List[Tuple[_T1, Optional[_T2]]]:
    return [(enc_dec_prompt["encoder_prompt"],
             enc_dec_prompt["decoder_prompt"])
            for enc_dec_prompt in enc_dec_prompts]
