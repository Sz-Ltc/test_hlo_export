import jax
import jax.numpy as jnp

from layers.normalizations import rms_norm


def test_rms_norm_shape_and_finite():
  rng = jax.random.PRNGKey(0)
  nf = 8
  mod = rms_norm(num_features=nf)
  x = jax.random.normal(rng, (2, 3, nf))
  vars_ = mod.init(rng, x)
  y = mod.apply(vars_, x)
  assert y.shape == x.shape
  assert jnp.isfinite(y).all()
