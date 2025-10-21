import argparse
import os
import sys

import pyconfig
from globals import PKG_DIR
from hlo_exporter import HLOExporter


def main():
  """Main entry point for HLO export functionality.

  This function parses command line arguments and exports HLO representations
  of neural network layers based on the specified capture type and target.
  """
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
    "--capture",
    type=str,
    choices=["model", "layer", "sublayer", "chain"],
    required=True,
    help="Capture type: model | layer | sublayer | chain",
  )
  parser.add_argument(
    "--target",
    type=str,
    default="",
    help="Target description when capturing layer/sublayer/chain",
  )
  parser.add_argument("--output_dir", type=str, default="./hlo_outputs")
  parser.add_argument("--model_name", type=str)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--seq_len", type=int, default=128)
  parser.add_argument(
    "--weight_dtype", type=str, default="bfloat16", help="Weight data type"
  )
  parser.add_argument("--base_emb_dim", type=int, help="Base embedding dimension")
  parser.add_argument(
    "--base_num_query_heads", type=int, help="Base number of query heads"
  )
  parser.add_argument("--base_num_kv_heads", type=int, help="Base number of KV heads")
  parser.add_argument("--base_mlp_dim", type=int, help="Base MLP dimension")
  parser.add_argument(
    "--base_num_decoder_layers", type=int, help="Base number of decoder layers"
  )
  parser.add_argument("--head_dim", type=int, help="Head dimension")
  parser.add_argument("--vocab_size", type=int, help="Vocabulary size")
  parser.add_argument(
    "--dtype", type=str, default="bfloat16", help="Data type for activations"
  )
  parser.add_argument(
    "--normalization_layer_epsilon", type=float, help="Normalization layer epsilon"
  )
  parser.add_argument("--decoder_block", type=str, help="Decoder block type")
  parser.add_argument("--override_model_config", type=str, help="Override model config")
  parser.add_argument("--hardware", type=str, help="Hardware type")
  parser.add_argument(
    "--with_grads",
    action="store_true",
    help="If set, run forward+backward and compute loss/gradients",
  )
  parser.add_argument("--use_iota_embed", action="store_true")
  args = parser.parse_args()

  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)

  if args.capture in ("layer", "sublayer", "chain") and not args.target:
    parser.error("When --capture is layer/sublayer/chain, --target must be provided")

  config_files = [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")]

  config_overrides = {
    "run_name": "export_hlo",
    "enable_checkpointing": False,
    "attention": "dot_product",
    "override_model_config": True,
    **{
      k: v
      for k, v in vars(args).items()
      if v is not None
      and k
      in [
        "weight_dtype",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
        "base_mlp_dim",
        "base_num_decoder_layers",
        "head_dim",
        "vocab_size",
        "dtype",
        "normalization_layer_epsilon",
        "decoder_block",
        "override_model_config",
        "model_name",
        "hardware",
        "use_iota_embed",
      ]
    },
  }
  print(f"Config overrides: {config_overrides}")

  config = pyconfig.initialize(config_files, **config_overrides)
  exporter = HLOExporter(config)

  if args.capture == "layer":
    target = args.target.lower()
    if target == "rmsnorm":
      exporter.export_rmsnorm_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "dense_general":
      exporter.export_dense_general_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "mlp_block":
      exporter.export_mlp_block_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "mla":
      exporter.export_mla_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "gqa":
      exporter.export_gqa_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "input_embedding":
      exporter.export_input_embedding_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    elif target == "output_embedding":
      exporter.export_output_embedding_layer(
        str(output_dir),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        with_grads=args.with_grads,
      )
    else:
      raise ValueError(f"unsupported layer: {args.target}")

  print(f"HLO files exported to: {output_dir}")


if __name__ == "__main__":
  main()
