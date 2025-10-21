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

"""Normalization Layers."""

from typing import Any

from flax import linen as nn
from flax import nnx
from jax import lax
import jax.numpy as jnp
from layers import nnx_wrappers
from layers.initializers import Initializer, variable_to_logically_partitioned


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
    self,
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: tuple[str | None, ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    *,
    rngs: nnx.Rngs,
  ):
    """Initialize RMSNorm module.

    Args:
        num_features: Number of feature dimensions
        epsilon: Numerical stability constant to prevent division by zero
        dtype: Computation data type
        weight_dtype: Weight data type
        kernel_axes: Kernel sharding axes
        scale_init: Scale parameter initializer
        rngs: Random number generator state for nnx
    """
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.scale = nnx.Param(
      scale_init(rngs.params(), (num_features,), weight_dtype),
      sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply RMS normalization to input tensor.

    Args:
        x: Input tensor

    Returns:
        Normalized tensor
    """
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.scale.value

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


def rms_norm(
  num_features: int,
  epsilon: float = 1e-6,
  dtype: Any = jnp.float32,
  weight_dtype: Any = jnp.float32,
  kernel_axes: tuple[str | None, ...] = (),
  scale_init: Initializer = nn.initializers.ones,
  name: str | None = None,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
    RMSNorm,
    num_features=num_features,
    epsilon=epsilon,
    dtype=dtype,
    weight_dtype=weight_dtype,
    kernel_axes=kernel_axes,
    scale_init=scale_init,
    name=name,
    metadata_fn=variable_to_logically_partitioned,
  )
  return module
