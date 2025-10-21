"""
python tests/linear_hlo_export_naive.py
"""

import os
import argparse

os.environ["XLA_FLAGS"] = (
  "--xla_dump_hlo_as_text --xla_dump_hlo_as_dot "
  "--xla_dump_to=./hlo_outputs/linear_naive"
)

import jax
import jax.numpy as jnp


Array = jnp.ndarray
Params = dict[str, Array]


def init_linear_params(key, in_features: int, out_features: int) -> Params:
  k = jax.random.split(key, 1)[0]
  w = jax.random.normal(k, (in_features, out_features), dtype=jnp.bfloat16) / jnp.sqrt(
    in_features
  ).astype(jnp.bfloat16)
  b = jnp.zeros((out_features,), dtype=jnp.bfloat16)
  return {"w": w, "b": b}


def linear_apply(params: Params, x: Array) -> Array:
  # x: [batch, seq, in_features]
  y = jnp.einsum("bsi,io->bso", x, params["w"]) + params["b"]
  return y


def main():
  parser = argparse.ArgumentParser(description="Linear HLO 导出")
  parser.add_argument("--batch", type=int, default=int(os.environ.get("LIN_BATCH", 2)))
  parser.add_argument(
    "--seq_len", type=int, default=int(os.environ.get("LIN_SEQ_LEN", 128))
  )
  parser.add_argument(
    "--in_features", type=int, default=int(os.environ.get("LIN_IN", 256))
  )
  parser.add_argument(
    "--out_features", type=int, default=int(os.environ.get("LIN_OUT", 256))
  )
  parser.add_argument("--seed", type=int, default=int(os.environ.get("LIN_SEED", 0)))
  args = parser.parse_args()

  batch_size = args.batch
  seq_len = args.seq_len
  in_features = args.in_features
  out_features = args.out_features
  key = jax.random.key(args.seed)

  params = init_linear_params(key, in_features, out_features)
  x = jax.random.normal(key, (batch_size, seq_len, in_features), dtype=jnp.bfloat16)

  lin_jit = jax.jit(linear_apply)
  lin_jit(params, x)

  print("HLO files exported to: ./hlo_outputs/linear_naive")


if __name__ == "__main__":
  main()
