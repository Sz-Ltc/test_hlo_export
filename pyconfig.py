"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import OrderedDict
from typing import Any, Dict, List
import os
import re
import datetime
from functools import reduce
import jax

import omegaconf
from utils import max_logging, clear_jax_backends

_MAX_PREFIX = "M_"
_BASE_CONFIG_ATTR = "base_config"
_DEFAULT_MESH_NAME = ["X", "Y", "Z"]


def yaml_key_to_env_key(s: str) -> str:
  return _MAX_PREFIX + s.upper()


def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")


_yaml_types_to_parser = {str: str, int: int, float: float, bool: string_to_bool}


def validate_model_name(model_name: str) -> None:
  if not isinstance(model_name, str):
    raise ValueError(f"model_name must be a string, got {type(model_name)}")


def validate_and_update_keys(raw_keys: dict, model_vars: Any, config_name: str) -> dict:
  for key, value in model_vars.items():
    if key in raw_keys:
      max_logging.log(f"Overriding {key} from model config {config_name}")
    raw_keys[key] = value
  return raw_keys


def validate_no_keys_overwritten_twice(
  keys_from_env_and_command_line: list, keys_from_model: list
) -> None:
  intersection = set(keys_from_env_and_command_line) & set(keys_from_model)
  if intersection:
    raise ValueError(
      f"Keys {intersection} were overwritten by both env/command line and model config"
    )


class _HyperParameters:
  def _validate_env_variables(self, raw_data_from_yaml: dict[str, Any]):
    for environment_var in os.environ:
      if environment_var[: len(_MAX_PREFIX)] == _MAX_PREFIX:
        proposed_key = environment_var[len(_MAX_PREFIX) :].lower()
        if proposed_key not in raw_data_from_yaml:
          raise ValueError(
            f"We received env `{environment_var}` but it doesn't match a key, "
            "so it is assumed a mistake."
          )
        if not environment_var[len(_MAX_PREFIX) :].isupper():
          raise ValueError(
            f"We received env `{environment_var}` but it isn't all uppercase."
          )

  def _update_from_env_and_command_line(
    self, raw_keys, raw_data_from_yaml, argv, **kwargs
  ) -> list[str]:
    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
    kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
    cmdline_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)
    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(cmdline_cfg, resolve=True)
    assert raw_data_from_cmd_line is not None

    updated_keys = []
    for k in raw_data_from_cmd_line:
      if k not in raw_data_from_yaml:
        raise ValueError(
          f"Key {str(k)} was passed at the command line but isn't in config."
        )

    for k in raw_data_from_yaml:
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(
          f"You are passing overrides by both CLI and ENV for `{k}`. "
          "This isn't allowed."
        )

      if k not in raw_data_from_cmd_line and yaml_key_to_env_key(k) not in os.environ:
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      updated_keys.append(k)
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
        type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
          f"For key '{k}', type {type(raw_data_from_yaml[k])} not in "
          f"{_yaml_types_to_parser.keys()}, can't pass at the CLI or ENV"
        )

      if new_proposal is None:
        raw_keys[k] = None
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal
      else:
        try:
          _type_parser = _yaml_types_to_parser[type(raw_data_from_yaml[k])]
          assert callable(_type_parser)
          raw_keys[k] = _type_parser(new_proposal)
        except ValueError as e:
          raise ValueError(
            f"Couldn't parse value from CLI or ENV '{new_proposal}' for key '{k}'"
          ) from e

    return updated_keys

  def _load_config(self, config_name: str) -> Any:
    base_cfg = omegaconf.OmegaConf.load(config_name)
    raw_data_from_yaml = omegaconf.OmegaConf.to_container(base_cfg, resolve=True)

    return raw_data_from_yaml

  @staticmethod
  def update_model_vars(
    base_config_path, raw_keys, config_name: str, keys_from_env_and_command_line
  ):
    validate_model_name(raw_keys["model_name"])
    max_logging.log(f"Running Model: {raw_keys['model_name']}")

    updated_keys = []
    if raw_keys["model_name"] != "default":
      model_name = raw_keys["model_name"]
      file_path = os.path.join(
        os.path.dirname(base_config_path), "models", f"{model_name}.yml"
      )
      if not os.path.isfile(file_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "configs", "models", f"{model_name}.yml")
      _model_vars = omegaconf.OmegaConf.load(file_path)
      model_vars = omegaconf.OmegaConf.to_container(_model_vars, resolve=True)
      assert isinstance(model_vars, dict)
      if raw_keys["override_model_config"]:
        model_vars = {
          key: value
          for key, value in model_vars.items()
          if key not in keys_from_env_and_command_line
        }
      updated_keys = list(model_vars.keys())
      validate_and_update_keys(raw_keys, model_vars, config_name)
    return updated_keys

  @staticmethod
  def update_hardware_vars(
    base_config_path, raw_keys, config_name: str, keys_from_env_and_command_line
  ):
    """Override config with hardware-specific yaml when `hardware` is specified.
    Looks for configs/hardware/{hardware}.yml next to the base config,
    then falls back to package configs.
    Respects override_model_config to avoid clobbering CLI/ENV-provided keys.
    """
    updated_keys: List[Any] = []
    hardware = raw_keys.get("hardware", "")
    if not hardware:
      return updated_keys

    file_path = os.path.join(
      os.path.dirname(base_config_path), "hardware", f"{hardware}.yml"
    )
    if not os.path.isfile(file_path):
      dir_path = os.path.dirname(os.path.realpath(__file__))
      file_path = os.path.join(dir_path, "configs", "hardware", f"{hardware}.yml")
    if not os.path.isfile(file_path):
      raise ValueError(
        f"Hardware config not found for '{hardware}'. Looked under 'hardware/' "
        "and 'configs/hardware/'."
      )

    _hw_vars = omegaconf.OmegaConf.load(file_path)
    hw_vars = omegaconf.OmegaConf.to_container(_hw_vars, resolve=True)
    assert isinstance(hw_vars, dict)
    if raw_keys.get("override_model_config"):
      hw_vars = {
        key: value
        for key, value in hw_vars.items()
        if key not in keys_from_env_and_command_line
      }
    updated_keys = list(hw_vars.keys())
    validate_and_update_keys(raw_keys, hw_vars, config_name)
    return updated_keys

  @staticmethod
  def user_init(raw_keys):
    if raw_keys["run_name"] == "":
      raw_keys["run_name"] = os.environ.get("JOBSET_NAME")
      if raw_keys["run_name"] == "":
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M")
        raw_keys["run_name"] = f"{raw_keys['model_name']}_{timestamp}"

    run_name = raw_keys["run_name"]
    base_output_directory = raw_keys["base_output_directory"]
    if run_name:
      raw_keys["tensorboard_dir"] = os.path.join(
        base_output_directory, run_name, "tensorboard", ""
      )
      raw_keys["checkpoint_dir"] = os.path.join(
        base_output_directory, run_name, "checkpoints", ""
      )
      raw_keys["metrics_dir"] = os.path.join(
        base_output_directory, run_name, "metrics", ""
      )

    # Set derived model dimensions
    raw_keys["emb_dim"] = raw_keys["base_emb_dim"]
    raw_keys["num_query_heads"] = raw_keys["base_num_query_heads"]
    raw_keys["num_kv_heads"] = raw_keys["base_num_kv_heads"]
    raw_keys["mlp_dim"] = raw_keys["base_mlp_dim"]
    raw_keys["num_decoder_layers"] = raw_keys["base_num_decoder_layers"]

  def check_config(self):
    def _check_hardware():
      if self.keys.get("interconnect", None) is None:
        self.keys["interconnect"] = {}
      mesh_shape = self.keys["interconnect"].get("mesh_shape", None)
      mesh_name = self.keys["interconnect"].get("mesh_name", None)
      if mesh_shape is None and mesh_name is None:
        self.keys["interconnect"]["mesh_shape"] = [1, 1, 1]
        self.keys["interconnect"]["mesh_name"] = _DEFAULT_MESH_NAME
      elif mesh_shape is None and mesh_name is not None:
        self.keys["interconnect"]["mesh_shape"] = [1 for _ in range(len(mesh_name))]
      elif mesh_shape is not None and mesh_name is None:
        self.keys["interconnect"]["mesh_name"] = [
          _DEFAULT_MESH_NAME[i % 3] for i in range(len(mesh_shape))
        ]
      else:
        dims = max(len(mesh_shape), len(mesh_name))
        self.keys["interconnect"]["mesh_shape"] = [
          mesh_shape[i] if i < len(mesh_shape) else 1 for i in range(dims)
        ]
        self.keys["interconnect"]["mesh_name"] = [
          mesh_name[i] if i < len(mesh_name) else _DEFAULT_MESH_NAME[i % 3]
          for i in range(dims)
        ]
      mesh_shape = self.keys["interconnect"].get("mesh_shape", None)
      mesh_name = self.keys["interconnect"].get("mesh_name", None)
      assert len(set(mesh_name)) == len(mesh_shape), f"Duplicate mesh name: {mesh_name}"
      assert 0 < len(mesh_name) <= 3 and 0 < len(mesh_shape) <= 3
      for name, size in zip(mesh_name, mesh_shape, strict=False):
        assert isinstance(name, str) and isinstance(size, int)

      mesh_size = reduce(lambda x, y: x * y, mesh_shape)
      hardware = self.keys.get("hardware", "cpu")
      clear_jax_backends()
      if hardware == "cpu":
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" in xla_flags:
          xla_flags = re.sub(
            r"(?<=xla_force_host_platform_device_count=)\d+", str(mesh_size), xla_flags
          )
        else:
          xla_flags = " ".join([
            f"--xla_force_host_platform_device_count={mesh_size}",
            xla_flags,
          ])
        jax.config.update("jax_num_cpu_devices", mesh_size)
        os.environ["XLA_FLAGS"] = xla_flags
      assert mesh_size == len(jax.devices()), (
        f"Mesh not map, {mesh_size} != {len(jax.devices())}"
      )
      return mesh_name

    def _check_no_duplicate(sharding):
      new_sharding = [s for s in sharding if s is not None]
      assert len(new_sharding) == len(set(new_sharding)), (
        f"Duplicate sharding: {sharding}"
      )

    def _check_sharding(sharding, num_sharding, sharding_size, mesh_name):
      if sharding is None:
        return [() for _ in range(num_sharding)]
      new_sharding = []
      for s in sharding:
        if s is None:
          new_sharding.append(())
        else:
          assert len(s) == sharding_size
          _check_no_duplicate(s)
          for s_ in s:
            assert s_ is None or s_ in mesh_name
          if all(s_ is None for s_ in s):
            new_sharding.append(())
          else:
            new_sharding.append(s)
      new_sharding += [() for _ in range(num_sharding - len(new_sharding))]
      return new_sharding

    def _check_attention_sharding(num_layers, mesh_name):
      attention_sharding = self.keys.get("attention_sharding", [])
      if len(attention_sharding) > num_layers:
        max_logging.warn(
          "Some configurations will be discarded for attention_sharding."
        )
      attention_sharding += [None for _ in range(num_layers - len(attention_sharding))]
      # MHA check
      self.keys["attention_sharding"] = [
        _check_sharding(attention_sharding[i], 4, 2, mesh_name)
        for i in range(num_layers)
      ]

    def _check_mlp_sharding(num_layers, mesh_name):
      mlp_sharding = self.keys.get("mlp_sharding", [])
      if len(mlp_sharding) > num_layers:
        max_logging.warn("Some configurations will be discarded for mlp_sharding.")
      mlp_sharding += [None for _ in range(num_layers - len(mlp_sharding))]
      # FFN check
      self.keys["mlp_sharding"] = [
        _check_sharding(mlp_sharding[i], 3, 2, mesh_name) for i in range(num_layers)
      ]

    def _check_data_sharding(mesh_name):
      data_sharding = self.keys.get("data_sharding", [])
      _check_no_duplicate(data_sharding)
      for sharding_axes in data_sharding:
        assert sharding_axes is None or sharding_axes in mesh_name
      self.keys["data_sharding"] = data_sharding

    # TODO: design and check.
    def _check_output_sharding(): ...

    mesh_name = _check_hardware()
    num_layers = self.keys.get("num_layers", 1)
    _check_attention_sharding(num_layers, mesh_name)
    _check_mlp_sharding(num_layers, mesh_name)
    _check_data_sharding(mesh_name)
    _check_output_sharding()

  def __init__(self, argv: list[str], **kwargs):
    config_name: str = argv[1]
    raw_data_from_yaml = self._load_config(config_name)

    self._validate_env_variables(raw_data_from_yaml)

    raw_keys: Dict[Any, Any] = OrderedDict()
    keys_from_env_and_command_line = self._update_from_env_and_command_line(
      raw_keys, raw_data_from_yaml, argv, **kwargs
    )
    max_logging.log(
      f"Updating keys from env and command line: {keys_from_env_and_command_line}"
    )
    keys_from_model = _HyperParameters.update_model_vars(
      argv[1], raw_keys, config_name, keys_from_env_and_command_line
    )
    max_logging.log(f"Updating keys from model: {keys_from_model}")
    if not raw_keys["override_model_config"]:
      validate_no_keys_overwritten_twice(
        keys_from_env_and_command_line, keys_from_model
      )

    keys_from_hardware = _HyperParameters.update_hardware_vars(
      argv[1], raw_keys, config_name, keys_from_env_and_command_line
    )
    if keys_from_hardware:
      max_logging.log(
        f"Updating keys from hardware '{raw_keys.get('hardware', '')}': "
        f"{keys_from_hardware}"
      )
      if not raw_keys["override_model_config"]:
        validate_no_keys_overwritten_twice(
          keys_from_env_and_command_line, keys_from_hardware
        )

    _HyperParameters.user_init(raw_keys)

    self.keys = raw_keys
    keys = [k for k in raw_keys]
    keys.sort()
    self.check_config()

    if raw_keys.get("log_config", True):
      for k in keys:
        if k != "hf_access_token":
          max_logging.log(f"Config param {k}: {raw_keys[k]}")


class HyperParameters:
  def __init__(self, config):
    object.__setattr__(self, "_config", config)

  def __getattr__(self, attr):
    try:
      return object.__getattribute__(self, "_config").keys[attr]
    except AttributeError as exc:
      raise AttributeError(
        f"'{self.__class__.__name__}' object has no attribute '{attr}'"
      ) from exc

  def __setattr__(self, attr, value):
    raise ValueError("Reinitialization of config is not allowed")

  def get_keys(self):
    return self._config.keys


def initialize(argv, **kwargs):
  initial_config = _HyperParameters(argv, **kwargs)
  config = HyperParameters(initial_config)
  return config
