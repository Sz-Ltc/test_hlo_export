from typing import Any
import numpy as np

import jax.nn
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from flax import nnx

from layers.embedding import RotaryEmbedding
from layers.linears import DenseGeneral
from layers.normalizations import RMSNorm
from layers.initializers import NdInitializer, nd_dense_init
from layers.nnx_wrappers import to_linen
from utils.common_types import Array, Config


# A large negative mask value is used for masking to ensure that the
# softmax function assigns an extremely low probability to the masked positions.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


def _generate_chunk_attention_mask(
  mask_shape: tuple[int, int], chunk_size: int, q_offset: int = 0
) -> jax.Array:
  """Generates an explicit boolean mask for chunked causal attention.

  This function computes the full boolean mask array where True indicates
  attention is allowed based on chunked causal rules (tokens attend only
  within the same chunk, and causally within that chunk).

  Args:
    mask_shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """

  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_offset
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  # chunk mask calculation
  same_chunk = (row_ids // chunk_size) == (col_ids // chunk_size)
  chunk_mask = same_chunk & (row_ids >= col_ids)
  return chunk_mask


def _make_block_mask_indices(bidirectional_mask):
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _make_bidirectional_block_mask(bidirectional_mask):
  """Creates bidirectional block mask from bidirectional_mask,
  where True corresponds to image tokens.

  bidirectional_mask shape: [B, L]
  bidirectional_block_mask shape: [B, L, L]
  Examples:
  bidirectional_mask = [[0, 1, 1, 1, 0, 0]]
  bidirectional_block_mask = [[
      [False, False, False, False, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
  ]]
  """
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  bidirectional_block_mask = (
    kv_block_indices[:, None, :] == q_block_indices[..., None]
  ) & (q_block_indices[..., None] > 0)
  return bidirectional_block_mask


def generate_attention_mask(
  query,
  key,
  decoder_segment_ids: Array | None,
  attention_type: str | None = "global",
  sliding_window_size: int | None = None,
  chunk_attn_window_size: int | None = None,
  previous_chunk: Any = None,
  bidirectional_mask: Any = None,
) -> Array | None:
  """Generates a combined attention mask for Transformer models.

  This function constructs an attention mask by potentially combining
  several types of masks based on the input parameters and model
  configuration. The generated mask dictates which query-key pairs are
  allowed to attend to each other.

  The masking logic can enforce:
  1.  **Sequence Separation:** Using `decoder_segment_ids`, attention is
    confined within distinct sequences in a batch. This is crucial when
    multiple unrelated sequences are packed together.
  2.  **Causality:** Preventing attention to future positions. This is
    standard for autoregressive decoding. For chunked prefill, as
    described in the SARATHI paper [2], causality is adjusted based
    on `previous_chunk` information.
  3.  **Specialized Attention Patterns:** Depending on `self.attention_type`,
    it can apply:
    * Local Sliding Window Attention: Restricts attention to a
        fixed-size window around each query position.
    * Chunk Attention: Divides sequences into chunks and applies
        masking at the chunk level.
  4.  **Bidirectional Attention for Sub-sequences:** If `bidirectional_mask`
    is provided (e.g., for image tokens in a multimodal model),
    those parts of the sequence can attend bidirectionally, and this
    mask is OR-ed with other generated masks.

  The overall approach and specific masking techniques are influenced by
  efficient attention mechanisms like those found in the Pallas MHA
  Flash Attention reference [1].

  Args:
    query: The query tensor, typically of shape
        `[batch_size, q_sequence_length, num_heads, head_dim]`.
        Used primarily for deriving sequence length.
    key: The key tensor, typically of shape
        `[batch_size, kv_sequence_length, num_heads, head_dim]`.
        Used primarily for deriving sequence length.
    decoder_segment_ids: Optional `Array` of shape `[batch_size, q_sequence_length]`.
        Identifies distinct sequences within the batch. Attention is
        restricted to elements within the same segment ID. In autoregressive
        mode, specific values (e.g., `common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR`)
        can mark the currently active sequence for decoding.
    previous_chunk: Optional. Information about previously processed
        key/value chunks, often a tensor representing the previous keys/values.
        Used to correctly offset causal masks in chunked attention or
        streaming scenarios. Its shape might be
        `[batch_size, prev_kv_sequence_length, ...]`.
    bidirectional_mask: Optional `Array` of shape `[batch_size, kv_sequence_length]`.
        If provided, this boolean mask indicates tokens (e.g., image tokens)
        that are allowed to attend bidirectionally. The resulting
        block-wise bidirectional mask is combined with other masks using a
        logical OR.

  Returns:
    An `Array` representing the attention mask, broadcastable to the shape
    `[batch_size, num_heads, q_sequence_length, kv_sequence_length]`.
    Positions with `0.0` allow attention, while positions with
    `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
    Returns `None` if no masking is determined to be necessary based on
    the inputs and configuration.

  References:
    [1] JAX Pallas MHA Flash Attention:
        https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
    [2] SARATHI: Efficient LLM Inference by Piggybacking Decodes with
        Chunked Prefills - ArXiv:2308.16369 (https://arxiv.org/abs/2308.16369)
  """
  mask = None
  if decoder_segment_ids is not None:
    mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
    mask = mask[:, None, None, :, :]

  _, q_seq_len, _, _ = query.shape
  _, kv_seq_len, _, _ = key.shape
  next_pos = 0
  if previous_chunk is not None:
    next_pos = previous_chunk.shape[1]
    if mask is not None:
      mask = mask[:, :, :, next_pos : next_pos + q_seq_len, :]

  mask_shape = (q_seq_len, kv_seq_len)
  # row_ids indicates the position of query
  # col_ids indicates the position of kv
  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  # Attention mask for chunked prefill is generated in the same way
  # as mentioned in SARATHI - https://arxiv.org/abs/2308.16369
  causal_mask = (col_ids <= row_ids + next_pos)[None, None, None, :, :]

  output_mask = None
  if (mask is not None) and (causal_mask is not None):
    output_mask = jnp.logical_and(mask, causal_mask)
  elif mask is not None:
    output_mask = mask
  elif causal_mask is not None:
    output_mask = causal_mask

  if attention_type == "local_sliding" and output_mask is not None:
    if sliding_window_size is None:
      raise ValueError(
        "Sliding_window_size must be set if Local Sliding attention type"
      )

    row_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, 1), 0) + next_pos
    col_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (1, kv_seq_len), 1)
    sliding_mask = (col_ids_sliding > (row_ids_sliding - sliding_window_size)) & (
      col_ids_sliding <= row_ids_sliding
    )
    output_mask = sliding_mask * output_mask
  elif attention_type == "chunk" and output_mask is not None:
    mask_shape = (q_seq_len, kv_seq_len)
    chunk_mask = _generate_chunk_attention_mask(
      mask_shape=(q_seq_len, kv_seq_len),
      chunk_size=chunk_attn_window_size,
      q_offset=next_pos,
    )
    output_mask = chunk_mask * output_mask

  if bidirectional_mask is not None:
    image_mask = _make_bidirectional_block_mask(bidirectional_mask)
    output_mask = output_mask | image_mask[:, None, None, ...]

  return (
    jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None
  )


def qk_product(query: Array, key: Array) -> Array:
  """Query-Key product.

  Args:
    query: Query projection, in shape of [b, t, n, d]
    key: Key projection in shape of [b, s, n_kv, d]

  Returns:
    results in shape [b, n_kv, n // n_kv, t, s].

  Annotations:
    b: batch size
    t: query length
    s: key / value length
    d: head / kv dimension
    n: number of query heads
    n_kv: number of kv heads, sometimes annotated as k
    n // n_kv: number of group for query, sometimes annotated with g
  """
  b, t, n, d = query.shape
  n_kv = key.shape[-2]
  query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
  result = jnp.einsum("btkgd,bskd->bkgts", query, key)
  return result


def wv_product(attn_weights: Array, value: Array) -> Array:
  """weighted value product.

  Args:
    attn_weights: Computed results of qk_einsum, in shape [b, n_kv, n // n_kv, t, s]
    value: Value projection, in shape of [b, s, n_kv, d]

  Returns:
    result in shape [b, t, n, d]

  Annotations:
    b: batch size
    t: query length
    s: key / value length
    d: head / kv dimension
    n: number of query heads
    n_kv: number of kv heads, sometimes annotated as k
    n // n_kv: number of group for query, sometimes annotated with g
  """
  out = jnp.einsum("bkgts,bskd->btkgd", attn_weights, value)
  b, t, n_kv, g, d = out.shape
  result = jnp.reshape(out, (b, t, n_kv * g, d))
  return result


def apply_attention_dot(
  query: Array,
  key: Array,
  value: Array,
  decoder_segment_ids: Array | None,
  attention_type: str | None = "global",
  sliding_window_size: int | None = None,
  chunk_attn_window_size: int | None = None,
  previous_chunk: Any = None,
  bidirectional_mask: Any = None,
  attn_logits_soft_cap: float | None = None,
  float32_logits: bool = False,
):
  """Apply Attention."""
  # special sharding for decode
  attn_weights = qk_product(query, key)

  if attn_logits_soft_cap:
    attn_weights = jnp.tanh(attn_weights / attn_logits_soft_cap)
    attn_weights = attn_weights * attn_logits_soft_cap

  # Casting softmax computation for float32 for model stability.
  if float32_logits:
    attn_weights = attn_weights.astype(jnp.float32)
  attn_mask = generate_attention_mask(
    query,
    key,
    decoder_segment_ids,
    attention_type,
    sliding_window_size,
    chunk_attn_window_size,
    previous_chunk,
    bidirectional_mask,
  )
  if attn_mask is not None:
    attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)
  return wv_product(attn_weights, value)


class MLA(nnx.Module):
  """Multi-Head Latent Attention (MLA) layer."""

  def __init__(
    self,
    config: Config,
    kernel_axes: tuple | None = None,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    name: str | None = None,
    *,
    rngs: nnx.Rngs,
  ):
    """Initializes the MLA module.

    Args:
      config: The model configuration.
      ... and other configuration parameters for MLA attention.
      rngs: The random number generators for initialization, passed by the
            nnx.to_linen wrapper.
    """
    self.config = config
    self.qk_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
    self.v_head_dim = self.config.qk_nope_head_dim

    # Assert required configuration parameters for MLA attention.
    assert self.config.kv_lora_rank > 0, "KV LoRA rank must be > 0"
    assert self.config.qk_nope_head_dim > 0, "QK NoPe head dim must be > 0"
    assert self.config.qk_rope_head_dim > 0, "QK RoPE head dim must be > 0"
    assert self.config.num_query_heads == self.config.num_kv_heads, (
      "MLA requires equal number of query and kv heads"
    )

    if self.config.q_lora_rank == 0:
      if kernel_axes is not None:
        assert len(kernel_axes) == 4
      # Standard Q projection (without LoRA).
      self.query = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(self.config.num_query_heads, self.qk_head_dim),
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=kernel_axes[0] if kernel_axes is not None else None,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
      )
    else:
      if kernel_axes is not None:
        assert len(kernel_axes) == 5
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.config.q_lora_rank,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=kernel_axes[0] if kernel_axes is not None else None,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
      )
      self.q_norm = RMSNorm(
        num_features=self.config.q_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
      )
      self.wq_b = DenseGeneral(
        in_features_shape=self.config.q_lora_rank,
        out_features_shape=(self.config.num_query_heads, self.qk_head_dim),
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=kernel_axes[1] if kernel_axes is not None else None,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
      in_features_shape=self.config.emb_dim,
      out_features_shape=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
      axis=-1,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[-3] if kernel_axes is not None else None,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )
    self.kv_norm = RMSNorm(
      num_features=self.config.kv_lora_rank,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      epsilon=self.config.normalization_layer_epsilon,
      rngs=rngs,
    )
    self.wkv_b = DenseGeneral(
      in_features_shape=self.config.kv_lora_rank,
      out_features_shape=(
        self.config.num_query_heads,
        (self.config.qk_nope_head_dim + self.v_head_dim),
      ),
      axis=-1,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[-2] if kernel_axes is not None else None,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )

    self.out_projection = DenseGeneral(
      in_features_shape=(self.config.num_query_heads, self.config.head_dim),
      out_features_shape=self.config.emb_dim,
      axis=(-2, -1),
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[-1]
      if kernel_axes is not None
      else None,  # trade speed with memory
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      use_bias=self.config.attention_out_projection_use_bias,
      rngs=rngs,
    )

    self.rotary_embedding = self.init_rotary_embedding(rngs)

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5

  def init_rotary_embedding(self, rngs):
    """Initializes the rotary embeddings, handling different model types.

    Returns:
      The rotary embedding module that will be used in the model.
    """
    # For MLA attention RoPE is applied to only `self.qk_rope_head_dim`
    # portion the heads.
    rope_embedding_dims = self.config.qk_rope_head_dim

    rotary_embedding = RotaryEmbedding(
      min_timescale=self.config.rope_min_timescale,
      max_timescale=self.config.rope_max_timescale,
      embedding_dims=rope_embedding_dims,
      fprop_dtype=self.config.dtype,
      rngs=rngs,
    )
    return rotary_embedding

  def mla_query_projection(self, inputs_q: Array, inputs_positions: Array) -> Array:
    """Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0."""
    if self.config.q_lora_rank == 0:
      q = self.query(inputs_q)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q)  # [B, L, q_lora_rank]
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      q = self.wq_b(low_rank_q)  # [B, L, n_heads * qk_head_dim]

    # Split into non-positional and rotary parts.
    q_nope, q_pe = jnp.split(q, [self.config.qk_nope_head_dim], axis=-1)
    q_pe = self.rotary_embedding(q_pe, position=inputs_positions)
    # Query projection is scaled by self.softmax_scale to be consistent
    # with MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale

    return query

  def mla_get_key_value(self, low_rank_main, key_rope):
    """get (key,value) pair from mla"""
    kv_out = self.wkv_b(low_rank_main)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.config.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(
      key_rope,
      (
        key_nope.shape[0],
        key_nope.shape[1],
        self.config.num_query_heads,
        key_rope.shape[3],
      ),
    )

    key = jnp.concatenate([key_nope, key_rope], axis=-1)
    return key, value

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array):
    """MLA key/value projection with integrated rotary embedding."""
    low_rank = self.wkv_a(inputs)
    low_rank_main, low_rank_rope = jnp.split(
      low_rank, [self.config.kv_lora_rank], axis=-1
    )
    low_rank_main = self.kv_norm(low_rank_main)

    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.rotary_embedding(key_rope, position=inputs_positions)

    key, value = self.mla_get_key_value(low_rank_main, key_rope)
    return key, value

  def __call__(
    self,
    inputs_q: Array,
    inputs_kv: Array,
    inputs_positions: Array | None = None,
    decoder_segment_ids: Array | None = None,
    *,
    deterministic: bool = False,
  ) -> Array:
    """Forward pass for MLA.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      deterministic: Disables dropout if set to True.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    query = self.mla_query_projection(inputs_q, inputs_positions)
    key, value = self.mla_kv_projection(inputs_kv, inputs_positions)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = apply_attention_dot(
      query, key, value, decoder_segment_ids=decoder_segment_ids
    )

    out = self.out_projection(out)
    return out


def mla_block(
  config: Config,
  kernel_axes: tuple | None = None,
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
  name: str | None = None,
):
  mla = to_linen(
    MLA,
    config=config,
    kernel_axes=kernel_axes,
    kernel_init=kernel_init,
    name=name,
  )
  return mla


class GQA(nnx.Module):
  """Group Queryed Attention (GQA) layer."""

  def __init__(
    self,
    config: Config,
    kernel_axes: tuple | None = None,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    name: str | None = None,
    *,
    rngs: nnx.Rngs,
  ):
    """Initializes the GQA module.

    Args:
      config: The model configuration.
      ... and other configuration parameters for GQA attention.
      rngs: The random number generators for initialization,
        passed by the nnx.to_linen wrapper.
    """
    self.config = config
    assert self.config.num_query_heads % self.config.num_kv_heads == 0
    self.n_repeat = self.config.num_query_heads // self.config.num_kv_heads
    assert self.config.emb_dim % self.config.num_query_heads == 0
    self.head_dim = self.config.emb_dim // self.config.num_query_heads

    if kernel_axes is not None:
      assert len(kernel_axes) == 4

    self.query_projection = DenseGeneral(
      in_features_shape=self.config.emb_dim,
      out_features_shape=(self.config.num_query_heads, self.head_dim),
      axis=-1,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[0] if kernel_axes is not None else None,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )

    self.key_projection = DenseGeneral(
      in_features_shape=self.config.emb_dim,
      out_features_shape=(self.config.num_kv_heads, self.head_dim),
      axis=-1,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[1] if kernel_axes is not None else None,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )

    self.value_projection = DenseGeneral(
      in_features_shape=self.config.emb_dim,
      out_features_shape=(self.config.num_kv_heads, self.head_dim),
      axis=-1,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[2] if kernel_axes is not None else None,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )

    self.out_projection = DenseGeneral(
      in_features_shape=(self.config.num_query_heads, self.head_dim),
      out_features_shape=self.config.emb_dim,
      axis=(-2, -1),
      kernel_init=kernel_init,
      kernel_axes=kernel_axes[3]
      if kernel_axes is not None
      else None,  # trade speed with memory
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      matmul_precision=self.config.matmul_precision,
      use_bias=self.config.attention_out_projection_use_bias,
      rngs=rngs,
    )

    self.rotary_embedding = self.init_rotary_embedding(rngs)

    # Set softmax scaling.
    self.softmax_scale = self.config.head_dim**-0.5

  def init_rotary_embedding(self, rngs):
    """Initializes the rotary embeddings, handling different model types.

    Returns:
      The rotary embedding module that will be used in the model.
    """
    rope_embedding_dims = self.head_dim

    rotary_embedding = RotaryEmbedding(
      min_timescale=self.config.rope_min_timescale,
      max_timescale=self.config.rope_max_timescale,
      embedding_dims=rope_embedding_dims,
      fprop_dtype=self.config.dtype,
      rngs=rngs,
    )
    return rotary_embedding

  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array,
    inputs_v: Array,
    inputs_positions: Array | None = None,
    decoder_segment_ids: Array | None = None,
    *,
    deterministic: bool = False,
  ) -> Array:
    """Forward pass for GQA.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_k: Key input   [batch, k_length, embed_dim].
      inputs_v: Value input   [batch, v_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      deterministic: Disables dropout if set to True.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      GQA-attended outputs.
    """
    q, k, v = (
      self.query_projection(inputs_q),
      self.key_projection(inputs_k),
      self.value_projection(inputs_v),
    )
    q_rope = self.rotary_embedding(q, position=inputs_positions) * self.softmax_scale
    k_rope = self.rotary_embedding(k, position=inputs_positions)

    out = apply_attention_dot(
      q_rope, k_rope, v, decoder_segment_ids=decoder_segment_ids
    )
    out = self.out_projection(out)
    return out


def gqa_block(
  config: Config,
  kernel_axes: tuple | None = None,
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
  name: str | None = None,
):
  gqa = to_linen(
    GQA,
    config=config,
    kernel_axes=kernel_axes,
    kernel_init=kernel_init,
    name=name,
  )
  return gqa
