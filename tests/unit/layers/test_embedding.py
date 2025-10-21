import numpy as np
import pytest
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock

from layers.embedding import (
  RotaryEmbedding,
  InputEmbed,
  OutputEmbed,
  input_embed,
  output_embed,
)
from utils.common_types import Config


def test_timescale_progression_matches_geometric_sequence():
  embedding = RotaryEmbedding(
    min_timescale=1,
    max_timescale=10_000,
    embedding_dims=4,
  )

  expected = np.array([1.0, 100.0], dtype=np.float32)
  np.testing.assert_allclose(np.array(embedding.timescale), expected, rtol=1e-6)


def test_rotary_embedding_applies_expected_rotation():
  embedding = RotaryEmbedding(
    min_timescale=1,
    max_timescale=1,
    embedding_dims=4,
    cast_as_fprop_dtype=False,
    fprop_dtype=jnp.float32,
  )

  inputs = jnp.array(
    [[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]],
    dtype=jnp.float32,
  )
  position = jnp.array([[0, 1]], dtype=jnp.int32)

  outputs = embedding(inputs, position=position)

  position_np = np.expand_dims(np.expand_dims(position, axis=-1), axis=-1)
  sinusoid_input = position_np
  sin = np.sin(sinusoid_input).astype(np.float32)
  cos = np.cos(sinusoid_input).astype(np.float32)
  first_half = np.array(inputs[..., :2])
  second_half = np.array(inputs[..., 2:])
  expected_first = first_half * cos - second_half * sin
  expected_second = second_half * cos + first_half * sin
  expected = np.concatenate([expected_first, expected_second], axis=-1)

  np.testing.assert_allclose(np.array(outputs), expected, rtol=1e-6, atol=1e-6)


def test_rotary_embedding_rejects_odd_hidden_dimension():
  with pytest.raises(ValueError, match="must be a multiple of 2"):
    RotaryEmbedding(min_timescale=1, max_timescale=1, embedding_dims=3)


def test_rotary_embedding_rejects_non_rank4_inputs():
  embedding = RotaryEmbedding(min_timescale=1, max_timescale=1, embedding_dims=4)

  bad_inputs = jnp.ones((2, 2, 4), dtype=jnp.float32)
  with pytest.raises(ValueError, match="rank 4 tensor"):
    embedding(bad_inputs, position=jnp.ones((2, 2), dtype=jnp.int32))


def test_rotary_embedding_requires_matching_hidden_dimension():
  embedding = RotaryEmbedding(min_timescale=1, max_timescale=1, embedding_dims=4)

  mismatched_inputs = jnp.ones((1, 1, 1, 6), dtype=jnp.float32)
  with pytest.raises(ValueError, match="must match the hidden dimension"):
    embedding(mismatched_inputs, position=jnp.zeros((1, 1), dtype=jnp.int32))


# Tests for InputEmbed
@pytest.fixture
def mock_config():
  config = MagicMock()
  config.vocab_size = 1000
  config.emb_dim = 32
  config.weight_dtype = jnp.float32
  config.use_iota_embed = False
  return config


def test_input_embed_init(mock_config):
  """Test that InputEmbed initializes correctly."""
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = InputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Check that the embedding matrix has the right shape
  assert embed.embedding.value.shape == (mock_config.vocab_size, mock_config.emb_dim)

  # Check that the attributes are set correctly
  assert embed.num_embeddings == mock_config.vocab_size
  assert embed.num_features == mock_config.emb_dim
  assert embed.config == mock_config


def test_input_embed_call(mock_config):
  """Test that InputEmbed.__call__ works correctly."""
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = InputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Create input tokens
  batch_size = 2
  seq_len = 3
  tokens = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
  assert tokens.shape == (batch_size, seq_len)

  # Embed the tokens
  output = embed(tokens)

  # Check that the output has the right shape
  assert output.shape == (batch_size, seq_len, mock_config.emb_dim)

  # Check that the output values match the embedding table
  for i in range(batch_size):
    for j in range(seq_len):
      token_id = int(tokens[i, j])
      np.testing.assert_array_equal(output[i, j], embed.embedding.value[token_id])


def test_input_embed_with_iota(mock_config):
  """Test that InputEmbed works with use_iota_embed=True."""
  mock_config.use_iota_embed = True
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = InputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Create input tokens
  batch_size = 2
  seq_len = 3
  tokens = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

  # Embed the tokens
  output = embed(tokens)

  # Check that the output has the right shape
  assert output.shape == (batch_size, seq_len, mock_config.emb_dim)


def test_input_embed_rejects_non_integer_inputs(mock_config):
  """Test that InputEmbed rejects non-integer inputs."""
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = InputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Create float inputs
  tokens = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

  # Embedding should raise an error
  with pytest.raises(ValueError, match="Input type must be an integer"):
    embed(tokens)


def test_input_embed(mock_config):
  """Test that input_embed returns a Linen module."""
  linen_module = input_embed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
  )

  # Check that the module has an init method
  assert hasattr(linen_module, "init")

  # Check that the module has an apply method
  assert hasattr(linen_module, "apply")


# Tests for OutputEmbed
def test_output_embed_init(mock_config):
  """Test that OutputEmbed initializes correctly."""
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = OutputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Check that the embedding matrix has the right shape
  assert embed.embedding.value.shape == (mock_config.vocab_size, mock_config.emb_dim)

  # Check that the attributes are set correctly
  assert embed.num_embeddings == mock_config.vocab_size
  assert embed.num_features == mock_config.emb_dim
  assert embed.config == mock_config


def test_output_embed_call(mock_config):
  """Test that OutputEmbed.__call__ works correctly."""
  rng = jax.random.PRNGKey(0)
  rngs = jax.random.split(rng, 2)

  embed = OutputEmbed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
    rngs=MagicMock(params=lambda: rngs[0]),
  )

  # Create input hidden states
  batch_size = 2
  seq_len = 3
  hidden_states = jnp.ones(
    (batch_size, seq_len, mock_config.emb_dim), dtype=jnp.float32
  )

  # Project the hidden states
  output = embed(hidden_states)

  # Check that the output has the right shape (batch_size, seq_len, vocab_size)
  assert output.shape == (batch_size, seq_len, mock_config.vocab_size)


def test_output_embed(mock_config):
  """Test that output_embed returns a Linen module."""
  linen_module = output_embed(
    num_embeddings=mock_config.vocab_size,
    num_features=mock_config.emb_dim,
    config=mock_config,
  )

  # Check that the module has an init method
  assert hasattr(linen_module, "init")

  # Check that the module has an apply method
  assert hasattr(linen_module, "apply")
