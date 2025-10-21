import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.nnx import Module
from layers.nnx_wrappers import (
  is_vanilla_variable,
  to_linen_var,
  ToLinen,
)


@pytest.fixture
def mock_mesh():
  os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    os.environ.get("XLA_FLAGS", ""),
  ])
  return jax.sharding.Mesh(np.array(jax.devices()).reshape(-1), ("X"))


# Test is_vanilla_variable
@pytest.mark.parametrize(
  "metadata, expected",
  [
    ({}, True),
    ({"_hooks": ()}, True),
    ({"_hooks": (lambda x: x,)}, False),
    ({"other": "value"}, False),
  ],
)
def test_is_vanilla_variable(metadata, expected):
  """Test is_vanilla_variable function with different metadata."""

  # Create a mock variable state
  class MockVariableState:
    def __init__(self, metadata):
      self.metadata = metadata

    def get_metadata(self):
      return self.metadata

  vs = MockVariableState(metadata)
  result = is_vanilla_variable(vs)
  assert result == expected


# Test to_linen_var (Linen variable conversion)
def test_to_linen_var():
  """Test to_linen_var function."""

  # Create a mock variable state with linen metadata
  class MockVariableState:
    def __init__(self, metadata):
      self.metadata = metadata
      self.value = jnp.zeros((3, 3))
      self.type = object

    def get_metadata(self):
      return self.metadata

  # Test with vanilla variable (no linen_meta_type)
  vs_vanilla = MockVariableState({})
  result_vanilla = to_linen_var(vs_vanilla)
  # Use jnp.array_equal for array comparison
  assert jnp.array_equal(result_vanilla, vs_vanilla.value)

  # Test with non-vanilla variable (should return NNXMeta)
  vs_non_vanilla = MockVariableState({"other": "value"})
  result_non_vanilla = to_linen_var(vs_non_vanilla)
  # Should return NNXMeta object
  assert hasattr(result_non_vanilla, "var_type")


# Test ToLinen class (converting an NNX module to Linen)
def test_tolinen_conversion():
  """Test ToLinen conversion."""

  # Create a simple NNX module
  class SimpleNNXModule(Module):
    def __init__(self):
      self.weight = jnp.ones((32, 64))

    def __call__(self, x):
      return x @ self.weight

  # Create ToLinen wrapper
  tolinnen_module = ToLinen(SimpleNNXModule, skip_rng=True, name="test_module")

  # Test initialization with proper RNG key
  x = jnp.ones((1, 32))
  rng = jax.random.PRNGKey(0)

  # Initialize the module
  variables = tolinnen_module.init(rng, x)

  # Test application
  output = tolinnen_module.apply(variables, x)
  assert output.shape == (1, 64)


# Test ToLinen class (converting an NNX module to Linen), and sharding it
def test_tolinen_conversion_with_sharding(mock_mesh):
  """Test ToLinen conversion, and sharding it."""

  # Create a simple NNX module
  class SimpleNNXModule(Module):
    def __init__(self):
      self.weight = flax.nnx.Param(
        jnp.ones((32, 64 * len(jax.devices()))),  # RNG key and shape for W2 creation
        sharding=(None, "X"),
      )

    def __call__(self, x):
      return x @ self.weight

  # Create ToLinen wrapper
  tolinnen_module = ToLinen(SimpleNNXModule, skip_rng=True, name="test_module")

  # Test initialization with proper RNG key
  x = jnp.ones((1, 32))
  rng = jax.random.PRNGKey(0)

  # Initialize the module
  with mock_mesh:
    variables = tolinnen_module.init(rng, x, mesh=mock_mesh)
    assert variables["params"]["weight"].shape == (32, 64 * len(jax.devices()))
    for idx in range(len(jax.devices())):
      assert variables["params"]["weight"].addressable_data(idx).shape == (32, 64)

    # Test application
    output = tolinnen_module.apply(variables, x)
    assert output.shape == (1, 64 * len(jax.devices()))
    for idx in range(len(jax.devices())):
      assert output.addressable_data(idx).shape == (1, 64)
