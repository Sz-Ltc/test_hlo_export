import numpy as np
import pytest
from types import SimpleNamespace
from functools import reduce

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jrandom = pytest.importorskip("jax.random")

from layers.attention import (
  DEFAULT_MASK_VALUE,
  _generate_chunk_attention_mask,
  _make_bidirectional_block_mask,
  _make_block_mask_indices,
  apply_attention_dot,
  apply_mask_to_logits,
  generate_attention_mask,
  gqa_block,
  mla_block,
  qk_product,
  wv_product,
)
from utils import clear_jax_backends


def test_apply_mask_to_logits_replaces_disallowed_entries():
  logits = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
  mask = jnp.array([[0.0, DEFAULT_MASK_VALUE]], dtype=jnp.float32)

  masked = apply_mask_to_logits(logits, mask)

  np.testing.assert_allclose(masked[:, 0], logits[:, 0])
  assert masked[0, 1] == pytest.approx(DEFAULT_MASK_VALUE)


def test_generate_chunk_attention_mask_matches_expected_pattern():
  chunk_mask = _generate_chunk_attention_mask((4, 4), chunk_size=2)

  # Each token can attend within its chunk, but only to current/past positions.
  expected = np.array([
    [True, False, False, False],
    [True, True, False, False],
    [False, False, True, False],
    [False, False, True, True],
  ])
  np.testing.assert_array_equal(np.array(chunk_mask), expected)


def test_generate_attention_mask_combines_segment_and_causal():
  query = jnp.zeros((1, 4, 1, 2), dtype=jnp.float32)
  key = jnp.zeros((1, 4, 1, 2), dtype=jnp.float32)
  decoder_segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

  mask = generate_attention_mask(query, key, decoder_segment_ids)

  assert mask is not None
  expected_boolean = np.array([
    [True, False, False, False],
    [True, True, False, False],
    [False, False, True, False],
    [False, False, True, True],
  ])
  expected = np.where(expected_boolean, 0.0, DEFAULT_MASK_VALUE)
  np.testing.assert_allclose(np.array(mask[0, 0, 0]), expected)


def test_generate_attention_mask_local_sliding_requires_window():
  query = jnp.zeros((1, 2, 1, 2), dtype=jnp.float32)
  key = jnp.zeros((1, 2, 1, 2), dtype=jnp.float32)
  decoder_segment_ids = jnp.array([[0, 0]], dtype=jnp.int32)

  with pytest.raises(ValueError, match="Sliding_window_size must be set"):
    generate_attention_mask(
      query,
      key,
      decoder_segment_ids,
      attention_type="local_sliding",
    )


def test_qk_product_groups_heads_correctly():
  query = jnp.arange(24, dtype=jnp.float32).reshape(1, 2, 4, 3)
  key = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 2, 3)

  product = qk_product(query, key)

  assert product.shape == (1, 2, 2, 2, 4)


def test_wv_product_reconstructs_values():
  attn_weights = jnp.ones((1, 2, 2, 2, 4), dtype=jnp.float32)
  value = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 2, 2)

  output = wv_product(attn_weights, value)

  assert output.shape == (1, 2, 4, 2)
  sum_per_head = jnp.sum(value, axis=1)[:, None, :, None, :]
  expected = jnp.broadcast_to(sum_per_head, (1, 2, 2, 2, 2)).reshape(1, 2, 4, 2)
  np.testing.assert_allclose(np.array(output), np.array(expected))


def test_generate_chunk_attention_mask_rejects_non_positive_chunk():
  with pytest.raises(ValueError, match="chunk_size must be positive"):
    _generate_chunk_attention_mask((2, 2), chunk_size=0)


def test_make_block_mask_indices_identifies_segments():
  bidirectional_mask = jnp.array([[0, 1, 1, 0, 1]], dtype=jnp.int32)
  indices = _make_block_mask_indices(bidirectional_mask)
  expected = np.array([[0, 1, 1, 0, 2]])
  np.testing.assert_array_equal(np.array(indices), expected)


def test_make_bidirectional_block_mask_creates_square_blocks():
  bidirectional_mask = jnp.array([[0, 1, 1, 0, 1]], dtype=jnp.int32)
  block_mask = _make_bidirectional_block_mask(bidirectional_mask)
  expected = np.array([
    [
      [False, False, False, False, False],
      [False, True, True, False, False],
      [False, True, True, False, False],
      [False, False, False, False, False],
      [False, False, False, False, True],
    ]
  ])
  np.testing.assert_array_equal(np.array(block_mask), expected)


def test_generate_attention_mask_with_chunk_and_bidirectional():
  query = jnp.zeros((1, 4, 1, 2), dtype=jnp.float32)
  key = jnp.zeros((1, 4, 1, 2), dtype=jnp.float32)
  decoder_segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)
  bidirectional_mask = jnp.array([[0, 1, 1, 0]], dtype=jnp.int32)

  mask = generate_attention_mask(
    query,
    key,
    decoder_segment_ids,
    attention_type="chunk",
    chunk_attn_window_size=2,
    previous_chunk=None,
    bidirectional_mask=bidirectional_mask,
  )

  assert mask is not None
  mask_np = np.array(mask[0, 0, 0])

  assert mask_np.shape == (4, 4)
  assert mask_np[1, 2] == 0.0
  assert mask_np[0, 3] == DEFAULT_MASK_VALUE


def test_generate_attention_mask_local_sliding_window():
  query = jnp.zeros((1, 3, 1, 2), dtype=jnp.float32)
  key = jnp.zeros((1, 3, 1, 2), dtype=jnp.float32)
  decoder_segment_ids = jnp.array([[0, 0, 0]], dtype=jnp.int32)

  mask = generate_attention_mask(
    query,
    key,
    decoder_segment_ids,
    attention_type="local_sliding",
    sliding_window_size=1,
  )

  mask_np = np.array(mask[0, 0, 0])
  assert mask_np[1, 1] == 0.0
  assert mask_np[2, 0] == DEFAULT_MASK_VALUE


def test_apply_attention_dot_with_soft_cap_and_float32_logits():
  query = jnp.ones((1, 2, 2, 1), dtype=jnp.float32)
  key = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)
  value = jnp.array([[[[1.0]], [[2.0]]]], dtype=jnp.float32)

  output = apply_attention_dot(
    query,
    key,
    value,
    decoder_segment_ids=None,
    attn_logits_soft_cap=2.0,
    float32_logits=True,
  )

  expected = np.array([[[[1.0], [1.0]], [[1.5], [1.5]]]], dtype=np.float32)
  np.testing.assert_allclose(np.array(output), expected, rtol=1e-6, atol=1e-6)


def _sharding_config():
  cfg = {
    "interconnect": {"mesh_shape": [2, 4], "mesh_name": ["X", "Y"]},
    "test_data_sharding": [None, None, None],
    "test_mla_sharding": [["X", "Y"], ["X", None], [None, "Y"], None],
    "test_gqa_sharding": [["X", "Y"], ["X", None], [None, "Y"], None],
  }
  return cfg


def _base_attention_config(**overrides):
  cfg = {
    "kv_lora_rank": 2,
    "q_lora_rank": 0,
    "qk_nope_head_dim": 2,
    "qk_rope_head_dim": 2,
    "num_query_heads": 2,
    "num_kv_heads": 2,
    "emb_dim": 4,
    "head_dim": 2,
    "dtype": jnp.float32,
    "weight_dtype": jnp.float32,
    "normalization_layer_epsilon": 1e-6,
    "matmul_precision": "default",
    "rope_min_timescale": 1.0,
    "rope_max_timescale": 10_000.0,
    "attention_out_projection_use_bias": False,
  }
  cfg.update(overrides)
  return SimpleNamespace(**cfg)


def _run_module(module, *args, **kwargs):
  rng = jrandom.key(0)
  outputs, variables = module.init_with_output(rng, *args, **kwargs)
  kwargs.pop("mesh", None)
  reapplied = module.apply(variables, *args, **kwargs)
  return outputs, reapplied


def test_mla_block_forward_without_lora():
  config = _base_attention_config(q_lora_rank=0)
  mla = mla_block(config=config)

  inputs = jnp.ones((1, 3, config.emb_dim), dtype=config.dtype)
  positions = jnp.arange(3)[None, :]
  decoder_segment_ids = jnp.array([[0, 0, 0]], dtype=jnp.int32)

  outputs, reapplied = _run_module(
    mla,
    inputs,
    inputs,
    positions,
    decoder_segment_ids,
    deterministic=True,
  )

  assert outputs.shape == (1, 3, config.emb_dim)
  np.testing.assert_allclose(np.array(outputs), np.array(reapplied))


def test_mla_block_forward_with_lora_path():
  config = _base_attention_config(q_lora_rank=2)
  mla = mla_block(config=config)

  inputs = jnp.ones((1, 2, config.emb_dim), dtype=config.dtype)
  positions = jnp.arange(2)[None, :]

  outputs, reapplied = _run_module(
    mla,
    inputs,
    inputs,
    positions,
    None,
    deterministic=True,
  )

  assert outputs.shape == (1, 2, config.emb_dim)
  np.testing.assert_allclose(np.array(outputs), np.array(reapplied))


def test_mla_block_forward_pass_with_sharding(monkeypatch):
  overrides_cfg = _sharding_config()
  config = _base_attention_config(**overrides_cfg)
  mesh_size = reduce(lambda x, y: x * y, config.interconnect["mesh_shape"])
  clear_jax_backends()
  ori_jax_num_cpu_devices = jax.config.jax_num_cpu_devices
  monkeypatch.setenv("XLA_FLAGS", f"--xla_force_host_platform_device_count={mesh_size}")
  jax.config.update("jax_num_cpu_devices", mesh_size)
  mesh = jax.sharding.Mesh(
    devices=np.array(jax.devices()).reshape(config.interconnect["mesh_shape"]),
    axis_names=config.interconnect["mesh_name"],
  )
  mla = mla_block(
    config=config,
    kernel_axes=getattr(config, "test_mla_sharding", ()),
  )

  data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(*config.test_data_sharding)
  )
  position_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(*config.test_data_sharding[:-1])
  )
  inputs_q = jax.device_put(
    jnp.ones((1, 3, config.emb_dim), dtype=config.dtype), data_sharding
  )
  inputs_kv = jax.device_put(
    jnp.ones((1, 3, config.emb_dim), dtype=config.dtype), data_sharding
  )
  positions = jax.device_put(jnp.arange(3)[None, :], position_sharding)
  decoder_segment_ids = jnp.array([[0, 0, 0]], dtype=jnp.int32)

  outputs, reapplied = _run_module(
    mla,
    inputs_q,
    inputs_kv,
    positions,
    decoder_segment_ids,
    deterministic=True,
    mesh=mesh,
  )

  assert outputs.shape == (1, 3, config.emb_dim)
  np.testing.assert_allclose(np.array(outputs), np.array(reapplied))

  clear_jax_backends()
  jax.config.update("jax_num_cpu_devices", ori_jax_num_cpu_devices)


def test_gqa_block_forward_pass():
  config = _base_attention_config()
  gqa = gqa_block(config=config)

  inputs_q = jnp.ones((1, 3, config.emb_dim), dtype=config.dtype)
  inputs_k = jnp.ones((1, 3, config.emb_dim), dtype=config.dtype)
  inputs_v = jnp.ones((1, 3, config.emb_dim), dtype=config.dtype)
  positions = jnp.arange(3)[None, :]
  decoder_segment_ids = jnp.array([[0, 0, 0]], dtype=jnp.int32)

  outputs, reapplied = _run_module(
    gqa,
    inputs_q,
    inputs_k,
    inputs_v,
    positions,
    decoder_segment_ids,
    deterministic=True,
  )

  assert outputs.shape == (1, 3, config.emb_dim)
  np.testing.assert_allclose(np.array(outputs), np.array(reapplied))


def test_gqa_block_forward_pass_with_sharding(monkeypatch):
  overrides_cfg = _sharding_config()
  config = _base_attention_config(**overrides_cfg)
  mesh_size = reduce(lambda x, y: x * y, config.interconnect["mesh_shape"])
  clear_jax_backends()
  ori_jax_num_cpu_devices = jax.config.jax_num_cpu_devices
  monkeypatch.setenv("XLA_FLAGS", f"--xla_force_host_platform_device_count={mesh_size}")
  jax.config.update("jax_num_cpu_devices", mesh_size)
  mesh = jax.sharding.Mesh(
    devices=np.array(jax.devices()).reshape(config.interconnect["mesh_shape"]),
    axis_names=config.interconnect["mesh_name"],
  )
  gqa = gqa_block(
    config=config,
    kernel_axes=getattr(config, "test_gqa_sharding", ()),
  )

  data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(*config.test_data_sharding)
  )
  position_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(*config.test_data_sharding[:-1])
  )
  inputs_q = jax.device_put(
    jnp.ones((1, 3, config.emb_dim), dtype=config.dtype), data_sharding
  )
  inputs_k = jax.device_put(
    jnp.ones((1, 3, config.emb_dim), dtype=config.dtype), data_sharding
  )
  inputs_v = jax.device_put(
    jnp.ones((1, 3, config.emb_dim), dtype=config.dtype), data_sharding
  )
  positions = jax.device_put(jnp.arange(3)[None, :], position_sharding)
  decoder_segment_ids = jnp.array([[0, 0, 0]], dtype=jnp.int32)

  outputs, reapplied = _run_module(
    gqa,
    inputs_q,
    inputs_k,
    inputs_v,
    positions,
    decoder_segment_ids,
    deterministic=True,
    mesh=mesh,
  )

  assert outputs.shape == (1, 3, config.emb_dim)
  np.testing.assert_allclose(
    np.array(outputs), np.array(reapplied), rtol=1e-6, atol=1e-6
  )

  clear_jax_backends()
  jax.config.update("jax_num_cpu_devices", ori_jax_num_cpu_devices)
