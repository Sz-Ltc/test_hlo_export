import jax


def clear_jax_backends():
  jax._src.api.clean_up()
  jax._src.xla_bridge._clear_backends()
  jax._src.util.clear_all_caches()
