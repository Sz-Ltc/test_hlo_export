from pathlib import Path
from collections.abc import Sequence
import numpy as np

import jax
import jax.numpy as jnp

from utils.common_types import Config
from layers.normalizations import rms_norm
from layers.linears import dense_general, mlp_block
from layers.attention import mla_block, gqa_block
from layers.embedding import input_embed, output_embed


class HLOExporter:
  """Export HLO (High Level Optimizer) representations of neural network layers.

  This class provides functionality to export various neural network layers
  and their input/output data for analysis and debugging purposes.
  """

  def __init__(self, config: Config):
    """Initialize the HLO exporter with configuration.

    Args:
        config: Configuration object containing model parameters
    """
    self.config = config
    self.mesh = jax.sharding.Mesh(
      devices=np.array(jax.devices()).reshape(config.interconnect["mesh_shape"]),
      axis_names=config.interconnect["mesh_name"],
    )
    self.input_rng = jax.random.PRNGKey(0)
    self.comp_base = {
      "xla_dump_hlo_as_text": True,
      "xla_dump_hlo_as_dot": True,
    }

  def _to_named_sharding(self, sharding_spec, *, allow_none: bool = False):
    """Convert a user-provided sharding spec into ``NamedSharding``.

    Args:
      sharding_spec: Partition description for array placement.
      allow_none: When ``True`` and ``sharding_spec`` is ``None`` the method
        returns ``None`` instead of a replicated sharding.

    Returns:
      A ``NamedSharding`` bound to the exporter mesh, or ``None`` when
      ``allow_none`` is ``True`` and the specification is ``None``.
    """

    if isinstance(sharding_spec, jax.sharding.Sharding):
      return sharding_spec

    if sharding_spec is None:
      if allow_none:
        return None
      sharding_spec = ()

    if isinstance(sharding_spec, jax.sharding.PartitionSpec):
      partition_spec = sharding_spec
    elif isinstance(sharding_spec, Sequence) and not isinstance(
      sharding_spec, (str, bytes)
    ):
      partition_spec = jax.sharding.PartitionSpec(*sharding_spec)
    else:
      raise TypeError(f"Unsupported sharding specification: {type(sharding_spec)!r}")

    return jax.sharding.NamedSharding(self.mesh, partition_spec)

  def export_rmsnorm_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export RMS normalization layer with sample input/output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    rmsnorm = rms_norm(
      num_features=self.config.emb_dim,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      epsilon=self.config.normalization_layer_epsilon,
      name="rms_norm",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )
    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)

    variables = rmsnorm.init(self.input_rng, hidden_states)

    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    rmsnorm_jit = jax.jit(rmsnorm.apply, out_shardings=output_sharding)
    rmsnorm_jit.lower(variables, hidden_states).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "rmsnorm"),
    })

    outputs = rmsnorm_jit(variables, hidden_states)

    if with_grads:
      self._compute_loss_and_gradients(variables, hidden_states, outputs, rmsnorm.apply)

    # 保存输入输出数据
    self._save_io_data(hidden_states, outputs, "rmsnorm", output_dir)

  def export_dense_general_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export dense general layer with sample input/output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    dense_layer = dense_general(
      inputs_shape=(batch_size, seq_len, self.config.emb_dim),
      out_features_shape=self.config.emb_dim,
      axis=-1,
      kernel_axes=getattr(self.config, "test_dense_general_sharding", ()),
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      use_bias=True,
      name="dense_general_layer",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )
    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )

    hidden_states = jax.device_put(hidden_states, data_sharding)
    variables = dense_layer.init(self.input_rng, hidden_states, mesh=self.mesh)

    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    dense_jit = jax.jit(dense_layer.apply, out_shardings=output_sharding)
    dense_jit.lower(variables, hidden_states).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "dense_general"),
    })

    outputs = dense_jit(variables, hidden_states)

    if with_grads:
      self._compute_loss_and_gradients(
        variables, hidden_states, outputs, dense_layer.apply
      )

    # 保存输入输出数据
    self._save_io_data(hidden_states, outputs, "dense_general", output_dir)

  def export_mlp_block_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export MLP block layer with sample input/output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients

    Returns:
        Path to the exported data directory
    """

    mlp_layer = mlp_block(
      config=self.config,
      in_features=self.config.emb_dim,
      intermediate_dim=self.config.mlp_dim,
      activations=self.config.mlp_activations,
      kernel_axes=getattr(self.config, "test_mlp_block_sharding", None),
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      use_bias=True,
      intermediate_dropout_rate=0.1,
      name="mlp_block_layer",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )

    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)
    params_rng, dropout_rng = jax.random.split(self.input_rng)
    variables = mlp_layer.init(
      {"params": params_rng, "dropout": dropout_rng},
      hidden_states,
      deterministic=True,
      mesh=self.mesh,
    )

    def mlp_apply(vars_, hidden):
      return mlp_layer.apply(vars_, hidden, deterministic=True)

    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    mlp_jit = jax.jit(mlp_apply, out_shardings=output_sharding)
    mlp_jit.lower(variables, hidden_states).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "mlp_block"),
    })

    outputs = mlp_jit(variables, hidden_states)

    if with_grads:
      self._compute_loss_and_gradients(variables, hidden_states, outputs, mlp_apply)

    # 保存输入输出数据
    self._save_io_data(hidden_states, outputs, "mlp_block", output_dir)

    return output_dir

  def export_mla_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export MLA layer with sample input/output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    mla = mla_block(
      config=self.config,
      kernel_axes=getattr(self.config, "test_mla_sharding", ()),
      name="mla_layer",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )
    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)
    pos_spec = getattr(self.config, "test_data_sharding", ())
    if isinstance(pos_spec, Sequence) and not isinstance(pos_spec, (str, bytes)):
      trimmed_pos_spec = pos_spec[:-1]
    else:
      trimmed_pos_spec = ()
    position_sharding = self._to_named_sharding(trimmed_pos_spec)
    position = jnp.repeat(jnp.arange(seq_len).reshape(1, -1), batch_size, axis=0)
    position = jax.device_put(position, position_sharding)
    params_rng, dropout_rng = jax.random.split(self.input_rng)
    variables = mla.init(
      {"params": params_rng, "dropout": dropout_rng},
      hidden_states,
      hidden_states,
      position,
      deterministic=True,
      mesh=self.mesh,
    )

    def mla_apply(vars_, q, kv, position):
      return mla.apply(vars_, q, kv, position, deterministic=True)

    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    mla_jit = jax.jit(mla_apply, out_shardings=output_sharding)
    mla_jit.lower(variables, hidden_states, hidden_states, position).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "mla"),
    })

    outputs = mla_jit(variables, hidden_states, hidden_states, position)

    # 保存输入输出数据
    self._save_io_data(hidden_states, outputs, "mla", output_dir)

    return output_dir

  def export_gqa_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export GQA layer with sample input/output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    gqa = gqa_block(
      config=self.config,
      kernel_axes=getattr(self.config, "test_gqa_sharding", ()),
      name="gqa_layer",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )
    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)
    pos_spec = getattr(self.config, "test_data_sharding", ())
    if isinstance(pos_spec, Sequence) and not isinstance(pos_spec, (str, bytes)):
      trimmed_pos_spec = pos_spec[:-1]
    else:
      trimmed_pos_spec = ()
    position_sharding = self._to_named_sharding(trimmed_pos_spec)
    position = jnp.repeat(jnp.arange(seq_len).reshape(1, -1), batch_size, axis=0)
    position = jax.device_put(position, position_sharding)
    params_rng, dropout_rng = jax.random.split(self.input_rng)
    variables = gqa.init(
      {"params": params_rng, "dropout": dropout_rng},
      hidden_states,
      hidden_states,
      hidden_states,
      position,
      deterministic=True,
      mesh=self.mesh,
    )

    def gqa_apply(vars_, q, k, v, position):
      return gqa.apply(vars_, q, k, v, position, deterministic=True)

    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    gqa_jit = jax.jit(gqa_apply, out_shardings=output_sharding)
    gqa_jit.lower(
      variables, hidden_states, hidden_states, hidden_states, position
    ).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "gqa"),
    })
    outputs = gqa_jit(variables, hidden_states, hidden_states, hidden_states, position)

    # 保存输入输出数据
    self._save_io_data(hidden_states, outputs, "gqa_block", output_dir)

  def export_input_embedding_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export Embedding layer with sample input data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    embedding = input_embed(
      num_embeddings=self.config.vocab_size,
      num_features=self.config.emb_dim,
      dtype=self.config.dtype,
      config=self.config,
      name="input_embedding",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "embedding_data_sharding", ())
    )
    # 生成随机整数索引，范围在[0, vocab_size)之间
    hidden_states = jax.random.randint(
      self.input_rng, (batch_size, seq_len), 0, self.config.vocab_size, dtype=jnp.int32
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)
    variables = embedding.init(self.input_rng, hidden_states)
    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    embedding_jit = jax.jit(embedding.apply, out_shardings=output_sharding)
    embedding_jit.lower(variables, hidden_states).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "input_embedding"),
    })

    outputs = embedding_jit(variables, hidden_states)

    if with_grads:
      self._compute_loss_and_gradients(
        variables, hidden_states, outputs, embedding.apply
      )

    self._save_io_data(hidden_states, outputs, "input_embedding", output_dir)

  def export_output_embedding_layer(
    self,
    output_dir: str,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    with_grads: bool = False,
  ):
    """Export Embedding layer with sample output data.

    Args:
        output_dir: Directory to save the exported data
        batch_size: Batch size for the sample input
        seq_len: Sequence length for the sample input
        with_grads: Whether to also compute loss and gradients
    """

    embedding = output_embed(
      num_embeddings=self.config.vocab_size,
      num_features=self.config.emb_dim,
      dtype=self.config.dtype,
      config=self.config,
      name="output_embedding",
    )

    data_sharding = self._to_named_sharding(
      getattr(self.config, "test_data_sharding", ())
    )
    hidden_states = jax.random.normal(
      self.input_rng,
      (batch_size, seq_len, self.config.emb_dim),
      dtype=self.config.dtype,
    )
    hidden_states = jax.device_put(hidden_states, data_sharding)
    variables = embedding.init(self.input_rng, hidden_states)
    output_sharding = self._to_named_sharding(
      getattr(self.config, "output_sharding", None), allow_none=True
    )
    embedding_jit = jax.jit(embedding.apply, out_shardings=output_sharding)
    embedding_jit.lower(variables, hidden_states).compile({
      **self.comp_base,
      "xla_dump_to": str(Path(output_dir) / "output_embedding"),
    })

    outputs = embedding_jit(variables, hidden_states)

    if with_grads:
      self._compute_loss_and_gradients(
        variables, hidden_states, outputs, embedding.apply
      )

    self._save_io_data(hidden_states, outputs, "output_embedding", output_dir)

  def _save_io_data(self, inputs, outputs, layer_name: str, output_dir: str) -> None:
    """Save input and output data for a layer to disk.

    Args:
        inputs: Input data to the layer
        outputs: Output data from the layer
        layer_name: Name of the layer for file naming
        output_dir: Base directory to save the data
    """
    try:
      # Create IO data directory
      io_dir = Path(output_dir) / f"{layer_name}_io"
      io_dir.mkdir(parents=True, exist_ok=True)

      # Save input data
      if isinstance(inputs, (list, tuple)):
        for idx, input_data in enumerate(inputs):
          input_array = np.asarray(jax.device_get(input_data))
          np.save(io_dir / f"input_{idx}.npy", input_array)

          # Also save as txt file (for easy viewing)
          txt_file = io_dir / f"input_{idx}.txt"
          with open(txt_file, "w") as f:
            f.write(f"Input {idx} shape: {input_array.shape}\n")
            f.write(f"Input {idx} dtype: {input_array.dtype}\n")
            f.write(f"Input {idx} data:\n")
            f.write(str(input_array))
      else:
        input_array = np.asarray(jax.device_get(inputs))
        np.save(io_dir / "input_0.npy", input_array)

        # Save as txt file
        txt_file = io_dir / "input_0.txt"
        with open(txt_file, "w") as f:
          f.write(f"Input shape: {input_array.shape}\n")
          f.write(f"Input dtype: {input_array.dtype}\n")
          f.write("Input data:\n")
          f.write(str(input_array))

      # Save output data
      def _save_output(obj, prefix="output"):
        leaves, treedef = jax.tree.flatten(obj)

        # Save metadata
        meta = {
          "num_leaves": len(leaves),
          "treedef": str(treedef),
          "output_type": type(obj).__name__,
        }

        with open(io_dir / f"{prefix}_meta.txt", "w", encoding="utf-8") as f:
          f.write("Output Metadata:\n")
          f.write("=" * 50 + "\n")
          for key, value in meta.items():
            f.write(f"{key}: {value}\n")

        # Save each leaf node
        for i, leaf in enumerate(leaves):
          leaf_array = np.asarray(jax.device_get(leaf))
          np.save(io_dir / f"{prefix}_{i}.npy", leaf_array)

          # Save as txt file
          txt_file = io_dir / f"{prefix}_{i}.txt"
          with open(txt_file, "w") as f:
            f.write(f"Output {i} shape: {leaf_array.shape}\n")
            f.write(f"Output {i} dtype: {leaf_array.dtype}\n")
            f.write(f"Output {i} data:\n")
            f.write(str(leaf_array))

      _save_output(outputs)

      print(f"Saved input/output data to: {io_dir}")

    except Exception as e:
      print(f"Failed to save input/output data: {e}")

  def _compute_loss_and_gradients(self, variables, inputs, outputs, apply_fn):
    """Compute loss and gradients for a given layer using JAX.

    Args:
        variables: Layer variables/parameters
        inputs: Input data to the layer
        outputs: Output data from the layer
        apply_fn: Model apply function (optional, for proper gradient computation)

    Returns:
        Tuple of (loss, gradients, target)
    """
    # 构造随机的真实值(target)
    target = jax.random.normal(self.input_rng, outputs.shape, dtype=outputs.dtype)

    def mse_loss_fn(variables, inputs, target):
      """Mean squared error loss with proper gradient computation"""
      outputs = apply_fn(variables, inputs)
      return jax.numpy.mean(jax.lax.square(outputs - target))

    def loss_and_grad_fn(variables, inputs, target):
      """同时计算 loss 和梯度"""
      loss = mse_loss_fn(variables, inputs, target)
      grad_fn = jax.grad(mse_loss_fn, argnums=0)
      gradients = grad_fn(variables, inputs, target)
      return loss, gradients

    loss_and_grad_fn_jit = jax.jit(loss_and_grad_fn)
    loss, gradients = loss_and_grad_fn_jit(variables, inputs, target)

    return loss, gradients, target
