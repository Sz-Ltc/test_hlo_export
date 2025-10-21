import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = jax.numpy

import hlo_exporter
from hlo_exporter import HLOExporter


@pytest.fixture
def mock_config():
  config = MagicMock()
  config.emb_dim = 128
  config.mlp_dim = 512
  config.dtype = jnp.float32
  config.weight_dtype = jnp.float32
  config.normalization_layer_epsilon = 1e-6
  config.mlp_activations = ["silu"]
  config.matmul_precision = "default"  # Add missing matmul_precision attribute
  config.test_data_sharding = ()
  config.output_sharding = None
  config.test_mlp_block_sharding = [["X"], ["X"]]
  devices = jax.devices()
  mesh_shape = (len(devices),)
  axis_names = ("X",)
  config.interconnect = {"mesh_shape": mesh_shape, "mesh_name": axis_names}
  return config


@pytest.fixture
def exporter(mock_config):
  return HLOExporter(mock_config)


def test_hlo_exporter_init(mock_config):
  exporter = HLOExporter(mock_config)

  assert exporter.config == mock_config
  assert isinstance(exporter.input_rng, jax.Array)


def test_export_rmsnorm_layer(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    exporter.export_rmsnorm_layer(temp_dir, batch_size=1, seq_len=64)

    # Check that IO data directory was created
    io_dir = Path(temp_dir) / "rmsnorm_io"
    assert io_dir.exists()

    # Check input files
    assert (io_dir / "input_0.npy").exists()
    assert (io_dir / "input_0.txt").exists()

    # Check output files
    assert (io_dir / "output_0.npy").exists()
    assert (io_dir / "output_0.txt").exists()
    assert (io_dir / "output_meta.txt").exists()

    # Verify input data shape
    input_data = np.load(io_dir / "input_0.npy")
    assert input_data.shape == (1, 64, exporter.config.emb_dim)

    # Verify output data shape
    output_data = np.load(io_dir / "output_0.npy")
    assert output_data.shape == (1, 64, exporter.config.emb_dim)


def test_export_dense_general_layer(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    exporter.export_dense_general_layer(temp_dir, batch_size=2, seq_len=32)

    io_dir = Path(temp_dir) / "dense_general_io"
    assert io_dir.exists()

    # Check files exist
    assert (io_dir / "input_0.npy").exists()
    assert (io_dir / "output_0.npy").exists()

    # Verify shapes
    input_data = np.load(io_dir / "input_0.npy")
    output_data = np.load(io_dir / "output_0.npy")

    assert input_data.shape == (2, 32, exporter.config.emb_dim)
    assert output_data.shape == (2, 32, exporter.config.emb_dim)


def test_export_mlp_block_layer(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    exporter.export_mlp_block_layer(temp_dir, batch_size=1, seq_len=16)

    io_dir = Path(temp_dir) / "mlp_block_io"
    assert io_dir.exists()

    # Check files exist
    assert (io_dir / "input_0.npy").exists()
    assert (io_dir / "output_0.npy").exists()

    # Verify shapes
    input_data = np.load(io_dir / "input_0.npy")
    output_data = np.load(io_dir / "output_0.npy")

    assert input_data.shape == (1, 16, exporter.config.emb_dim)
    assert output_data.shape == (1, 16, exporter.config.emb_dim)


def test_save_io_data_single_input(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    # Create test data
    inputs = jnp.ones((2, 4))
    outputs = jnp.ones((2, 4)) * 2

    exporter._save_io_data(inputs, outputs, "test_layer", temp_dir)

    io_dir = Path(temp_dir) / "test_layer_io"
    assert io_dir.exists()

    # Check input files
    assert (io_dir / "input_0.npy").exists()
    assert (io_dir / "input_0.txt").exists()

    # Check output files
    assert (io_dir / "output_0.npy").exists()
    assert (io_dir / "output_0.txt").exists()
    assert (io_dir / "output_meta.txt").exists()

    # Verify data
    saved_input = np.load(io_dir / "input_0.npy")
    saved_output = np.load(io_dir / "output_0.npy")

    np.testing.assert_array_equal(saved_input, inputs)
    np.testing.assert_array_equal(saved_output, outputs)


def test_save_io_data_multiple_inputs(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    # Create test data with multiple inputs
    inputs = [jnp.ones((2, 4)), jnp.ones((2, 8))]
    outputs = jnp.ones((2, 4)) * 3

    exporter._save_io_data(inputs, outputs, "multi_input", temp_dir)

    io_dir = Path(temp_dir) / "multi_input_io"
    assert io_dir.exists()

    # Check multiple input files
    assert (io_dir / "input_0.npy").exists()
    assert (io_dir / "input_1.npy").exists()
    assert (io_dir / "input_0.txt").exists()
    assert (io_dir / "input_1.txt").exists()

    # Verify data
    saved_input_0 = np.load(io_dir / "input_0.npy")
    saved_input_1 = np.load(io_dir / "input_1.npy")

    np.testing.assert_array_equal(saved_input_0, inputs[0])
    np.testing.assert_array_equal(saved_input_1, inputs[1])


def test_save_io_data_complex_output_structure(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    # Create test data with complex output structure
    inputs = jnp.ones((2, 4))
    outputs = {"logits": jnp.ones((2, 4)), "hidden": jnp.ones((2, 8))}

    exporter._save_io_data(inputs, outputs, "complex_output", temp_dir)

    io_dir = Path(temp_dir) / "complex_output_io"
    assert io_dir.exists()

    # Should have metadata file describing the structure
    assert (io_dir / "output_meta.txt").exists()

    # Should have separate files for each leaf
    assert (io_dir / "output_0.npy").exists()
    assert (io_dir / "output_1.npy").exists()

    # Check metadata content
    with open(io_dir / "output_meta.txt") as f:
      meta_content = f.read()
      assert "num_leaves: 2" in meta_content
      assert "output_type: dict" in meta_content


def test_save_io_data_handles_exceptions(exporter, capsys):
  # Test error handling when save fails
  with patch("numpy.save", side_effect=Exception("Mock save error")):
    with tempfile.TemporaryDirectory() as temp_dir:
      inputs = jnp.ones((2, 4))
      outputs = jnp.ones((2, 4))

      exporter._save_io_data(inputs, outputs, "error_test", temp_dir)

      # Should print error message
      captured = capsys.readouterr()
      assert "Failed to save input/output data" in captured.out


def test_save_io_data_creates_directories(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    # Use nested directory that doesn't exist
    nested_dir = os.path.join(temp_dir, "nested", "path")

    inputs = jnp.ones((1, 2))
    outputs = jnp.ones((1, 2))

    exporter._save_io_data(inputs, outputs, "test", nested_dir)

    # Should create the nested directory structure
    io_dir = Path(nested_dir) / "test_io"
    assert io_dir.exists()
    assert (io_dir / "input_0.npy").exists()


def test_txt_files_contain_metadata(exporter):
  with tempfile.TemporaryDirectory() as temp_dir:
    inputs = jnp.ones((3, 5), dtype=jnp.float32)
    outputs = jnp.ones((3, 5), dtype=jnp.float32) * 2

    exporter._save_io_data(inputs, outputs, "metadata_test", temp_dir)

    io_dir = Path(temp_dir) / "metadata_test_io"

    # Check input txt file contains shape and dtype
    with open(io_dir / "input_0.txt") as f:
      content = f.read()
      assert "shape: (3, 5)" in content
      assert "dtype: float32" in content

    # Check output txt file contains shape and dtype
    with open(io_dir / "output_0.txt") as f:
      content = f.read()
      assert "shape: (3, 5)" in content
      assert "dtype: float32" in content


def test_to_named_sharding_passthrough(exporter):
  partition_spec = jax.sharding.PartitionSpec()
  named_sharding = jax.sharding.NamedSharding(exporter.mesh, partition_spec)

  result = exporter._to_named_sharding(named_sharding)

  assert result is named_sharding


def test_to_named_sharding_allows_none(exporter):
  assert exporter._to_named_sharding(None, allow_none=True) is None


def test_to_named_sharding_from_sequence(exporter):
  result = exporter._to_named_sharding(())

  assert isinstance(result, jax.sharding.NamedSharding)


def test_to_named_sharding_from_partition_spec(exporter):
  spec = jax.sharding.PartitionSpec()
  result = exporter._to_named_sharding(spec)

  assert isinstance(result, jax.sharding.NamedSharding)


def test_to_named_sharding_raises_on_invalid_input(exporter):
  with pytest.raises(TypeError):
    exporter._to_named_sharding("invalid")


def test_export_rmsnorm_layer_with_grads(monkeypatch, exporter, tmp_path):
  mock_compute = MagicMock(return_value=(0.0, 0.0, 0.0))
  monkeypatch.setattr(
    hlo_exporter.HLOExporter, "_compute_loss_and_gradients", mock_compute
  )

  exporter.export_rmsnorm_layer(tmp_path, batch_size=1, seq_len=8, with_grads=True)

  assert mock_compute.call_count == 1


def test_export_dense_general_layer_with_grads(monkeypatch, exporter, tmp_path):
  mock_compute = MagicMock(return_value=(0.0, 0.0, 0.0))
  monkeypatch.setattr(
    hlo_exporter.HLOExporter, "_compute_loss_and_gradients", mock_compute
  )

  exporter.export_dense_general_layer(
    tmp_path, batch_size=1, seq_len=8, with_grads=True
  )

  assert mock_compute.call_count == 1


def test_export_mlp_block_layer_with_grads(monkeypatch, exporter, tmp_path):
  mock_compute = MagicMock(return_value=(0.0, 0.0, 0.0))
  monkeypatch.setattr(
    hlo_exporter.HLOExporter, "_compute_loss_and_gradients", mock_compute
  )

  exporter.export_mlp_block_layer(tmp_path, batch_size=1, seq_len=8, with_grads=True)

  assert mock_compute.call_count == 1


class _DummyMLABlock:
  def init(self, *_args, **_kwargs):
    return {"params": jnp.zeros(())}

  def apply(self, _vars, q, kv, position, deterministic=True):
    del deterministic
    return q + kv + position[:, :, None].astype(q.dtype)


class _DummyGQABlock:
  def init(self, *_args, **_kwargs):
    return {"params": jnp.zeros(())}

  def apply(self, _vars, q, k, v, position, deterministic=True):
    del deterministic
    return q + k + v + position[:, :, None].astype(q.dtype)


def test_export_mla_layer(monkeypatch, exporter, tmp_path):
  monkeypatch.setattr(hlo_exporter, "mla_block", lambda **_kwargs: _DummyMLABlock())

  exporter.export_mla_layer(tmp_path, batch_size=1, seq_len=4)

  io_dir = Path(tmp_path) / "mla_io"
  assert io_dir.exists()
  assert (io_dir / "output_0.npy").exists()


def test_export_gqa_layer(monkeypatch, exporter, tmp_path):
  monkeypatch.setattr(hlo_exporter, "gqa_block", lambda **_kwargs: _DummyGQABlock())

  exporter.export_gqa_layer(tmp_path, batch_size=1, seq_len=4)

  io_dir = Path(tmp_path) / "gqa_block_io"
  assert io_dir.exists()
  assert (io_dir / "output_0.npy").exists()


def test_compute_loss_and_gradients(exporter):
  variables = jnp.ones((2, 2))
  inputs = jnp.ones((2, 2))
  outputs = jnp.ones((2, 2))

  def apply_fn(vars_, data):
    return vars_ + data

  loss, gradients, target = HLOExporter._compute_loss_and_gradients(
    exporter, variables, inputs, outputs, apply_fn
  )

  assert gradients.shape == variables.shape
  assert loss.shape == ()
  assert target.shape == outputs.shape
