"""
python tests/ffn_hlo_export_naive.py
"""

from typing import Any
import os
import argparse

os.environ["XLA_FLAGS"] = (
  "--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./hlo_outputs/ffn_naive"
)

import jax
import jax.numpy as jnp


Array = jnp.ndarray
Params = dict[str, Array]


def silu(x: Array) -> Array:
  return x * jax.nn.sigmoid(x)


def init_ffn_params(key: Any, d_model: int, d_hidden: int) -> Params:
  k1, k2, k3 = jax.random.split(key, 3)
  # SwiGLU: wi_0, wi_1 双路投影
  wi_0 = jax.random.normal(k1, (d_model, d_hidden), dtype=jnp.bfloat16) / jnp.sqrt(
    d_model
  ).astype(jnp.bfloat16)
  bi_0 = jnp.zeros((d_hidden,), dtype=jnp.bfloat16)
  wi_1 = jax.random.normal(k2, (d_model, d_hidden), dtype=jnp.bfloat16) / jnp.sqrt(
    d_model
  ).astype(jnp.bfloat16)
  bi_1 = jnp.zeros((d_hidden,), dtype=jnp.bfloat16)
  # 输出投影
  wo = jax.random.normal(k3, (d_hidden, d_model), dtype=jnp.bfloat16) / jnp.sqrt(
    d_hidden
  ).astype(jnp.bfloat16)
  bo = jnp.zeros((d_model,), dtype=jnp.bfloat16)
  return {"wi_0": wi_0, "bi_0": bi_0, "wi_1": wi_1, "bi_1": bi_1, "wo": wo, "bo": bo}


def ffn_apply(params: Params, x: Array) -> Array:
  # x: [batch, seq, emb]
  # SwiGLU: silu(wi_0(x)) * wi_1(x)
  h0 = jnp.einsum("bse,eh->bsh", x, params["wi_0"]) + params["bi_0"]
  h1 = jnp.einsum("bse,eh->bsh", x, params["wi_1"]) + params["bi_1"]
  h = silu(h0) * h1  # 门控
  # 输出投影
  y = jnp.einsum("bsh,he->bse", h, params["wo"]) + params["bo"]
  return y


def main():
  parser = argparse.ArgumentParser(description="FFN HLO 导出")
  parser.add_argument("--batch", type=int, default=int(os.environ.get("FFN_BATCH", 2)))
  parser.add_argument(
    "--seq_len", type=int, default=int(os.environ.get("FFN_SEQ_LEN", 128))
  )
  parser.add_argument(
    "--d_model", type=int, default=int(os.environ.get("FFN_D_MODEL", 256))
  )
  parser.add_argument(
    "--d_hidden", type=int, default=int(os.environ.get("FFN_D_HIDDEN", 1024))
  )
  parser.add_argument("--seed", type=int, default=int(os.environ.get("FFN_SEED", 0)))
  args = parser.parse_args()

  batch_size = args.batch
  seq_len = args.seq_len
  d_model = args.d_model
  d_hidden = args.d_hidden
  key = jax.random.key(args.seed)

  params = init_ffn_params(key, d_model, d_hidden)
  x = jax.random.normal(key, (batch_size, seq_len, d_model), dtype=jnp.bfloat16)

  f_jitted = jax.jit(ffn_apply)
  f_jitted(params, x)

  print("HLO files exported to: ./hlo_outputs/ffn_naive")


if __name__ == "__main__":
  main()
