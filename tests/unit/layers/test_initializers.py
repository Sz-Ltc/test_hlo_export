import types
import pytest
import os
import numpy as np

import jax
import jax.numpy as jnp
from flax.linen import with_partitioning

from layers import initializers


@pytest.fixture
def mock_mesh():
  os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    os.environ.get("XLA_FLAGS", ""),
  ])
  return jax.sharding.Mesh(np.array(jax.devices()).reshape(-1), ("X"))


# --------------------------
# default_embed_init
# --------------------------
def test_default_embed_init_is_deterministic_and_nonzero():
  key0 = jax.random.PRNGKey(0)
  key1 = jax.random.PRNGKey(1)
  shape = (16, 8)
  dtype = jnp.float32

  arr_a = initializers.default_embed_init(key0, shape, dtype)
  arr_b = initializers.default_embed_init(key0, shape, dtype)
  arr_c = initializers.default_embed_init(key1, shape, dtype)

  assert jnp.array_equal(arr_a, arr_b)

  assert not jnp.array_equal(arr_a, arr_c)

  assert arr_a.shape == shape
  assert arr_a.dtype == dtype
  assert not jnp.all(arr_a == 0.0)


# --------------------------
# default_bias_init
# --------------------------
def test_default_bias_init_returns_zeros():
  key = jax.random.PRNGKey(0)
  shape = (5, 3)
  dtype = jnp.float32

  out = initializers.default_bias_init(key, shape, dtype)

  assert out.shape == shape
  assert out.dtype == dtype
  assert jnp.all(out == 0.0)


# --------------------------
# nd_dense_init
# --------------------------
@pytest.mark.parametrize(
  "scale,mode,distribution,shape,in_axis,out_axis",
  [
    (1.0, "fan_in", "normal", (8, 4), 0, 1),
    (2.0, "fan_out", "truncated_normal", (7, 5, 3), (0, 1), 2),
    (0.5, "fan_avg", "uniform", (9, 2), 0, 0),
  ],
)
def test_nd_dense_init_matches_jax_variance_scaling(
  scale, mode, distribution, shape, in_axis, out_axis
):
  key = jax.random.PRNGKey(42)
  dtype = jnp.float32

  # Expected: JAX's initializer with the same settings.
  expected_fn = jax.nn.initializers.variance_scaling(
    scale, mode, distribution, in_axis=in_axis, out_axis=out_axis
  )
  expected = expected_fn(key, shape, dtype)

  # Actual: our factory with axes applied at call time.
  init_fn = initializers.nd_dense_init(scale, mode, distribution)
  actual = init_fn(key, shape, dtype, in_axis, out_axis)

  # Same PRNGKey + same algorithm => results should match exactly.
  assert actual.shape == expected.shape
  assert actual.dtype == expected.dtype
  assert jnp.allclose(actual, expected)


# --------------------------
# variable_to_logically_partitioned
# --------------------------
class _DummyVar:
  """Duck-typed stand-in for nnx.VariableState used by the function under test."""

  def __init__(self, value, type_name="SomeType", sharding=("x",), metadata=None):
    self.value = value
    self.type = types.SimpleNamespace(__name__=type_name)
    self.sharding = sharding
    self._metadata = metadata or {}

  def get_metadata(self):
    return self._metadata


def test_variable_to_logically_partitioned_qtensor_branch(monkeypatch):
  class _FakeQTensor:
    pass

  monkeypatch.setattr(initializers.aqt_tensor, "QTensor", _FakeQTensor, raising=False)
  qt = _FakeQTensor()
  var = _DummyVar(value=qt)

  out = initializers.variable_to_logically_partitioned(var)
  assert out is qt, "Should return QTensor values unmodified"


def test_variable_to_logically_partitioned_overwrite_with_gradient_branch():
  sentinel = object()

  class _overwrite_with_gradient:
    def __init__(self, value, sharding=("x",), metadata=None):
      self.value = value
      self.sharding = sharding
      self._metadata = metadata or {}

    def get_metadata(self):
      return self._metadata

  var = _overwrite_with_gradient(value=sentinel)

  out = initializers.variable_to_logically_partitioned(var)
  assert out is sentinel, (
    "Should return value unmodified when type is _overwrite_with_gradient"
  )


def test_variable_to_logically_partitioned_wraps_with_logical_partitioning(monkeypatch):
  # Capture the args that would be passed to nn.LogicallyPartitioned without
  # requiring actual Flax behavior in the test.
  class _CaptureLP:
    def __init__(self, value, sharding, *, mesh=None, rules=None):
      self.value = value
      self.sharding = sharding
      self.mesh = mesh
      self.rules = rules

  monkeypatch.setattr(initializers.nn, "LogicallyPartitioned", _CaptureLP, raising=True)

  value = object()
  sharding = ("dp", "mp")
  metadata = {"mesh": "mesh-obj", "rules": "rules-obj"}
  var = _DummyVar(value=value, sharding=sharding, metadata=metadata)

  out = initializers.variable_to_logically_partitioned(var)

  assert isinstance(out, _CaptureLP)
  assert out.value is value
  assert out.sharding == sharding
  assert out.mesh == "mesh-obj"
  assert out.rules == "rules-obj"


def test_shard_variables(mock_mesh):
  variable1 = with_partitioning(jnp.ones, (None, "X"))((32, 64 * len(jax.devices())))
  variable2 = with_partitioning(jnp.ones, ("X",))((32 * len(jax.devices()), 64))
  variables = {"variable1": variable1, "variable2": variable2}

  sharding_variables = initializers.shard_variables(variables, mesh=mock_mesh)

  assert sharding_variables["variable1"].shape == (32, 64 * len(jax.devices()))
  assert sharding_variables["variable2"].shape == (32 * len(jax.devices()), 64)

  for idx in range(len(jax.devices())):
    assert sharding_variables["variable1"].addressable_data(idx).shape == (32, 64)
    assert sharding_variables["variable2"].addressable_data(idx).shape == (32, 64)
