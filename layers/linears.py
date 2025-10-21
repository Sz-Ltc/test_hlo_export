#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Linear Layers."""

import functools
import operator
from typing import Any
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import jax.numpy as jnp

from jax import lax
from jax.ad_checkpoint import checkpoint_name

from flax import nnx
import flax.linen as nn

from utils.common_types import DType, Array, Config
from layers import nnx_wrappers
from layers.initializers import (
  NdInitializer,
  nd_dense_init,
  default_bias_init,
  variable_to_logically_partitioned,
)
from utils.array_utils import normalize_axes, canonicalize_tuple


def _convert_to_activation_function(
  fn_or_string: str | Callable[..., Any],
) -> Callable[..., Any]:
  """Convert string or function to activation function.

  Args:
      fn_or_string: Activation function name string or function object

  Returns:
      Activation function

  Raises:
      ValueError: When conversion is not possible
  """
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
      f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _compute_dot_general_nnx(
  inputs,
  kernel,
  axis,
  contract_ind,
  matmul_precision,
):
  """Compute dot_general operation for NNX version.

  Args:
      inputs: Input tensor
      kernel: Kernel tensor
      axis: Contraction axis
      contract_ind: Contraction index
      matmul_precision: Matrix multiplication precision

  Returns:
      Result of dot_general operation
  """
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  return dot_general(
    inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision
  )


class DenseGeneral(nnx.Module):
  """A linear transformation with flexible axes."""

  def __init__(
    self,
    in_features_shape: Iterable[int] | int,
    out_features_shape: Iterable[int] | int,
    axis: Iterable[int] | int = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[str | None, ...] = (),
    use_bias: bool = False,
    matmul_precision: str = "default",
    *,  # Following arguments are keyword-only
    rngs: nnx.Rngs = None,
  ):
    """Initializes the DenseGeneral module.

    Args:
      in_features_shape: tuple with numbers of input features for axes specified in
        'axis'.
      out_features_shape: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      use_bias: whether to add bias in linear transformation.
      matmul_precision: Precision for matrix multiplication.
      rngs: RNG state for initialization in nnx.
    """
    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(
      len(self.axis), len(self.axis) + len(self.out_features_shape)
    )

    self.kernel = nnx.Param(
      self.kernel_init(
        rngs.params(),
        kernel_shape,
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      ),
      sharding=self.kernel_axes,
    )

    if self.use_bias:
      bias_axes = (
        self.kernel_axes[-len(self.out_features_shape) :]
        if self.kernel_axes is not None
        else None
      )
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
        default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
        sharding=bias_axes,
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array, _initializing: bool = False) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
          f"Input dimension {inputs.shape[ax]} at axis {ax} "
          f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(self.axis)))
    output = _compute_dot_general_nnx(
      inputs,
      kernel,
      norm_axis,
      contract_ind,
      self.matmul_precision,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output


def dense_general(
  *,
  inputs_shape: tuple[int, ...] | None = None,
  in_features_shape: tuple[int, ...] | int | None = None,
  out_features_shape: Iterable[int] | int,
  axis: Iterable[int] | int = -1,
  weight_dtype: DType = jnp.float32,
  dtype: DType = jnp.float32,
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
  kernel_axes: tuple[str | None, ...] = (),
  use_bias: bool = False,
  matmul_precision: str = "default",
  name: str | None = None,
):
  """Creates a DenseGeneral Linen module using nnx.bridge.to_linen.

  Args:
    inputs_shape: tuple with the shape of the inputs
    in_features_shape: tuple with numbers of input features for axes specified in
      'axis'.
    out_features_shape: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: logical axes for partitioning the kernel.
    use_bias: whether to add bias in linear transformation.
    matmul_precision: Precision for matrix multiplication.
    name: name passed to the ToLinen Module
  """
  if not (inputs_shape is not None) ^ (in_features_shape is not None):
    raise ValueError("Exactly one of inputs_shape or in_features must be specified.")

  if inputs_shape is not None:
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(
      inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape))
    )
  else:
    assert in_features_shape is not None
  module = nnx_wrappers.to_linen(
    DenseGeneral,
    in_features_shape=in_features_shape,
    out_features_shape=out_features_shape,
    axis=axis,
    weight_dtype=weight_dtype,
    dtype=dtype,
    kernel_init=kernel_init,
    kernel_axes=kernel_axes,
    use_bias=use_bias,
    matmul_precision=matmul_precision,
    name=name,
    metadata_fn=variable_to_logically_partitioned,
    abstract_init=False,
  )
  return module


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
    self,
    config: Config,
    in_features: int,
    intermediate_dim: int = 2048,
    activations: Sequence[str | Callable[..., Any]] = ("relu",),
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: list[list[str | None]] | None = None,
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    use_bias: bool = False,
    *,
    rngs: nnx.Rngs,
  ) -> None:
    """A MlpBlock module.

    Args:
      config: Config object containing model parameters.
      in_features: Number of input features.
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      kernel_axes: a list of logical axes for partitioning the kernel
        concluded in the block.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: computation data type for the dense layer.
      weight_dtype: weight data type for the dense layer.
      use_bias: whether to add bias in all feedforward layers.
    """
    self.config = config
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.activations = activations
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.use_bias = use_bias
    assert len(kernel_axes) == len(self.activations) + 1

    for idx in range(len(self.activations)):
      dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
      module = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes[idx],
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
      )
      setattr(self, dense_name, module)
    self.dropout = nnx.Dropout(
      rate=self.intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs
    )
    self.wo = DenseGeneral(
      in_features_shape=self.intermediate_dim,
      out_features_shape=in_features,
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      kernel_init=self.kernel_init,
      kernel_axes=kernel_axes[-1],
      use_bias=self.use_bias,
      matmul_precision=self.config.matmul_precision,
      rngs=rngs,
    )

  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
      module = getattr(self, dense_name)
      x = module(inputs)
      x = checkpoint_name(x, "mlp" + dense_name)
      if cfg.activations_in_float32:
        x = x.astype(jnp.float32)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    output = self.wo(x)

    output = checkpoint_name(output, "mlpwo")
    return output


def mlp_block(
  *,
  config: Config,
  in_features: int,
  intermediate_dim: int = 2048,
  activations: Sequence[str | Callable[..., Any]] = ("relu",),
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
  kernel_axes: list[list[str | None]] | None = None,
  intermediate_dropout_rate: float = 0.1,
  dtype: Any = jnp.float32,
  weight_dtype: Any = jnp.float32,
  use_bias: bool = False,
  name: str | None = None,
):
  """Creates a MlpBlock Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
    MlpBlock,
    config=config,
    in_features=in_features,
    intermediate_dim=intermediate_dim,
    activations=activations,
    kernel_init=kernel_init,
    kernel_axes=kernel_axes,
    intermediate_dropout_rate=intermediate_dropout_rate,
    dtype=dtype,
    weight_dtype=weight_dtype,
    use_bias=use_bias,
    name=name,
    metadata_fn=variable_to_logically_partitioned,
    abstract_init=False,
  )
  return module
