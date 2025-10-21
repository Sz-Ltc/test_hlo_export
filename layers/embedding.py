"""Embedding Layers."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from flax import nnx

from utils.common_types import DType, Array, Config
from layers.initializers import (
  Initializer,
  default_embed_init,
  variable_to_logically_partitioned,
)
from layers import nnx_wrappers


def input_embed(
  *,
  num_embeddings: int,
  num_features: int,
  config: Config,
  cast_input_dtype: None | DType = None,
  dtype: DType = jnp.float32,
  embedding_init: Initializer = default_embed_init,
  name: str | None = None,
):
  """Initializes the Embed NNX module and returns it as a Linen module.

  This function serves as a bridge to use the NNX-based `Embed` module within
  a Linen model. It wraps the `Embed` module using `nnx.bridge.to_linen`,
  making it compatible with the Linen API.

  Args:
    num_embeddings: The number of embeddings.
    num_features: The number of feature dimensions for each embedding.
    config: The model configuration.
    cast_input_dtype: The dtype to cast the input to, if any.
    dtype: The dtype of the embedding vectors.
    embedding_init: The initializer for the embedding matrix.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `Embed` module.
  """
  return nnx_wrappers.to_linen(
    InputEmbed,
    num_embeddings=num_embeddings,
    num_features=num_features,
    config=config,
    cast_input_dtype=cast_input_dtype,
    dtype=dtype,
    embedding_init=embedding_init,
    metadata_fn=variable_to_logically_partitioned,
    name=name,
  )


class InputEmbed(nnx.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors."""

  def __init__(
    self,
    num_embeddings: int,
    num_features: int,
    config: Config,
    cast_input_dtype: None | DType = None,
    dtype: DType = jnp.float32,
    embedding_init: Initializer = default_embed_init,
    *,
    # Not used in Embed but passed in by nnx.bridge.to_linen.
    # TODO: Remove when bridge no longer needed
    rngs: nnx.Rngs,
  ):
    """Initializes the Embed module.

    Args:
      num_embeddings: The number of embeddings.
      num_features: The number of feature dimensions for each embedding.
      config: The model configuration.
      cast_input_dtype: The dtype to cast the input to, if any.
      dtype: The dtype of the embedding vectors.
      embedding_init: The initializer for the embedding matrix.
      rngs: The random number generators for initialization.
    """
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.config = config
    self.cast_input_dtype = cast_input_dtype
    self.dtype = dtype

    self.embedding = nnx.Param(
      embedding_init(
        rngs.params(),
        (self.num_embeddings, self.num_features),
        self.config.weight_dtype,
      )
    )

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `num_features` dimension appended.
    """
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding.value, self.dtype))
    else:
      output = jnp.asarray(self.embedding.value, self.dtype)[inputs]

    return output


def output_embed(
  *,
  num_embeddings: int,
  num_features: int,
  config: Config,
  dtype: DType = jnp.float32,
  embedding_init: Initializer = default_embed_init,
  name: str | None = None,
):
  """Initializes the Embed NNX module and returns it as a Linen module.

  This function serves as a bridge to use the NNX-based `Embed` module within
  a Linen model. It wraps the `Embed` module using `nnx.bridge.to_linen`,
  making it compatible with the Linen API.

  Args:
    num_embeddings: The number of embeddings.
    num_features: The number of feature dimensions for each embedding.
    config: The model configuration.
    dtype: The dtype of the embedding vectors.
    embedding_init: The initializer for the embedding matrix.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `Embed` module.
  """
  return nnx_wrappers.to_linen(
    OutputEmbed,
    num_embeddings=num_embeddings,
    num_features=num_features,
    config=config,
    dtype=dtype,
    embedding_init=embedding_init,
    metadata_fn=variable_to_logically_partitioned,
    name=name,
  )


class OutputEmbed(nnx.Module):
  def __init__(
    self,
    num_embeddings: int,
    num_features: int,
    config: Config,
    dtype: DType = jnp.float32,
    embedding_init: Initializer = default_embed_init,
    *,
    # Not used in Embed but passed in by nnx.bridge.to_linen.
    # TODO: Remove when bridge no longer needed
    rngs: nnx.Rngs,
  ):
    """Initializes the Embed module.

    Args:
      num_embeddings: The number of embeddings.
      num_features: The number of feature dimensions for each embedding.
      config: The model configuration.
      dtype: The dtype of the embedding vectors.
      embedding_init: The initializer for the embedding matrix.
      rngs: The random number generators for initialization.
    """
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.config = config
    self.dtype = dtype

    self.embedding = nnx.Param(
      embedding_init(
        rngs.params(),
        (self.num_embeddings, self.num_features),
        self.config.weight_dtype,
      )
    )

  def __call__(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `num_features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    return jnp.dot(
      query,
      jnp.asarray(self.embedding.value, jnp.bfloat16).T,
      preferred_element_type=self.dtype,
    )


class RotaryEmbedding(nnx.Module):
  """Rotary Position Embedding."""

  def __init__(
    self,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    # Not used in RotaryEmbedding but passed in by nnx.bridge.to_linen.
    # TODO: Remove when bridge no longer needed
    rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the RotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.embedding_dims = embedding_dims
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype

    if self.embedding_dims % 2:
      raise ValueError(
        "Embedding dim for rotary position embedding must be a multiple of 2."
      )

  @property
  def timescale(self):
    """Returns the timescale for the rotary embedding."""
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    return self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

  def __call__(
    self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    inputs: jax.Array,
    position: jax.Array | None = None,
  ) -> jax.Array:
    """Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    assert position is not None
    if len(inputs.shape) != 4:
      raise ValueError(
        "Input is assumed to be a rank 4 tensor of shape[batch, sequence, heads, dims]."
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
        "The embedding dims of the rotary position embedding"
        "must match the hidden dimension of the inputs."
      )

    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale
    sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
    cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    x_out = jnp.concatenate((first_part, second_part), axis=-1)
    return x_out
