# [RFC]: Enable prompt_embeds content parts in Chat Completions API

### Motivation

vLLM supports `prompt_embeds` (pre-computed token embeddings) in the Completions API (`/v1/completions`) [via the `--enable-prompt-embeds` flag](https://docs.vllm.ai/en/stable/features/prompt_embeds/). This allows users to bypass the model's embedding layer by providing a serialized tensor of shape `(num_tokens, hidden_size)`.

However, the Chat Completions API (`/v1/chat/completions`) does not support `prompt_embeds`. Users who need to mix pre-computed embeddings with plain text content in a multi-turn conversation are forced to manually tokenize and apply chat templates outside of vLLM to use the completions endpoint.

**This RFC proposes adding `prompt_embeds` as a new content part type in the Chat Completions API, allowing users to interleave pre-computed embeddings with text within any message role.**

### Proposed Change

#### API Surface

A new content part type `"prompt_embeds"` is added to chat messages, following the same pattern as `"image_url"`, `"input_audio"`, and other multimodal content parts:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "prompt_embeds", "data": "<base64_encoded_tensor>"},
        {"type": "text", "text": "Summarize the document above."}
      ]
    }
  ]
}
```

The `data` field contains a base64-encoded serialized `torch.Tensor` of shape `(num_tokens, hidden_size)`, identical to the existing Completions API format.

**The feature remains gated behind `--enable-prompt-embeds`.**

Multiple `prompt_embeds` parts can appear in a single message or across messages (system, user, assistant), and can be freely interleaved with text parts.

#### Design Overview

The key challenge is that chat templates expect text, but `prompt_embeds` are pre-computed embeddings. The approach is:

1. **Placeholder substitution**: During message parsing, each `prompt_embeds` part is replaced with N (where N is the tensor's `num_tokens` dimension) copies of a dedicated placeholder token (`<prompt_embeds>`, registered as a special token in the tokenizer).
2. **Template rendering**: The chat template sees only text (including placeholder tokens) and renders normally.
3. **Position detection**: After tokenization, the placeholder token IDs are located in the token sequence to determine exact positions.
4. **Mask construction**: A full-length `is_token_ids` mask and embedding tensor are built, mapping each position to either the model's embedding layer (`True`) or the pre-computed embedding (`False`).
5. **GPU pipeline**: The existing `enable_prompt_embeds` forward pass handles the rest, **no changes to `gpu_model_runner.py`**.

#### Placeholder Token

A dedicated `<prompt_embeds>` special token is registered via `tokenizer.add_special_tokens()` at startup (when `--enable-prompt-embeds` is set). Special tokens are matched before BPE/WordPiece, so they always encode to exactly 1 token ID, never split into subwords. This approach:

- Guarantees 1:1 placeholder-to-position mapping.
- Avoids collision with chat template structural tokens (e.g `eos_token`)
- Works across all tokenizer families.
- Has precedent in vLLM (DeepSeek VL2 uses the same API at [`vllm/transformers_utils/processors/deepseek_vl2.py`)](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/processors/deepseek_vl2.py#L99-L103).

#### Data Flow

```
1. Request: messages[i].content[j] = {"type": "prompt_embeds", "data": "<base64>"}

2. parse_prompt_embeds()
  - safe_load_prompt_embeds() -> tensor (num_tokens, hidden_size)
  - tracker.add("prompt_embeds", tensor)
  - placeholder_str = "<prompt_embeds>" * num_tokens
  - inject placeholder_str as text into conversation
  
3. apply_chat_template(tokenize=True) -> token_ids (with placeholder token IDs)

4. _build_prompt_embeds_positions()
    - find consecutive chunks of placeholder token ID
    - return [(start_pos, length), ...] per tensor

5. Build full-length prompt_embeds tensor + is_token_ids mask.

6. EmbedsInput(prompt_embeds=full_tensor, prompt_token_ids=token_ids, is_token_ids=mask)

7. EngineCoreRequest -> Request -> InputBatch.add_request()
  - token_ids_cpu = prompt_token_ids
  - is_token_ids = per-position mask
  - req_prompt_embeds = full embedding tensor
  
8. gpu_model_runner forward pass (existing path, no changes)
    - is_token_ids=True  -> model.embed_input_ids()
    - is_token_ids=False -> pre-computed embeddings from buffer
```

#### Prefix Caching

The existing prefix caching mechanism works correctly for mixed mode without modification:

- `request.all_token_ids` will include both real and placeholder token IDs (primary hash key),
- `_gen_prompt_embeds_extra_hash_keys()` adds a SHA256 of the embedding tensor as an extra key.

Outcome: same tokens + same embeddings produce a cache hit.

#### Risks and Mitigations

1. **Placeholder token registration timing**: The special token must be registered *before* `get_cached_tokenizer()` wraps the tokenizer (which snapshots vocab/special token properties). Registration happens during renderer initialization when `--enable-prompt-embeds` is set.

2. **Multimodal data interaction**: `prompt_embeds` tensors are extracted from `mm_data` before the multimodal processor runs, since they require no encoder or model-specific processing. They go directly to the `is_token_ids` mask path.

### Any Other Things

- This builds on the existing `--enable-prompt-embeds` infrastructure. The GPU pipeline, the secure (`safe_load_prompt_embeds` with `weights_only=True`), and configuration flag are all reused.
- The implementation follows the same content part pattern used for `image_url`, `input_audio`, `audio_embeds`, and `image_embeds`. **No new abstractions are introduced**.
