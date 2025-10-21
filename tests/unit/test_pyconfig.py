import os
from pathlib import Path
import pytest

import pyconfig


def _base_cfg_path() -> str:
  return str(Path(__file__).resolve().parents[2] / "configs" / "base.yml")


def test_hardware_override_tpu_applies():
  cfg = pyconfig.initialize(["test.py", _base_cfg_path()], hardware="tpu")
  keys = cfg.get_keys()
  assert keys.get("vendor") == "google"
  assert keys.get("model") == "v6e"
  assert isinstance(keys.get("chips"), int)


# Test utility functions
def test_yaml_key_to_env_key():
  assert pyconfig.yaml_key_to_env_key("model_name") == "M_MODEL_NAME"
  assert (
    pyconfig.yaml_key_to_env_key("base_output_directory") == "M_BASE_OUTPUT_DIRECTORY"
  )


def test_string_to_bool():
  assert pyconfig.string_to_bool("true") is True
  assert pyconfig.string_to_bool("TRUE") is True
  assert pyconfig.string_to_bool("false") is False
  assert pyconfig.string_to_bool("FALSE") is False

  with pytest.raises(ValueError, match="Can't convert invalid to bool"):
    pyconfig.string_to_bool("invalid")


def test_validate_model_name():
  pyconfig.validate_model_name("llama3.2-1b")  # Should not raise

  with pytest.raises(ValueError, match="model_name must be a string"):
    pyconfig.validate_model_name(123)


def test_validate_and_update_keys():
  raw_keys = {"key1": "value1", "key2": "value2"}
  model_vars = {"key2": "new_value2", "key3": "value3"}

  result = pyconfig.validate_and_update_keys(raw_keys, model_vars, "test_config")

  assert result["key1"] == "value1"
  assert result["key2"] == "new_value2"  # overridden
  assert result["key3"] == "value3"  # added


def test_validate_no_keys_overwritten_twice():
  # Should not raise when no overlap
  pyconfig.validate_no_keys_overwritten_twice(["key1", "key2"], ["key3", "key4"])

  # Should raise when overlap exists
  with pytest.raises(ValueError, match=r"Keys {'key2'} were overwritten"):
    pyconfig.validate_no_keys_overwritten_twice(["key1", "key2"], ["key2", "key3"])


def test_env_validation_invalid_env_key(monkeypatch):
  monkeypatch.setenv("M_INVALID_KEY", "value")

  with pytest.raises(
    ValueError, match="We received env `M_INVALID_KEY` but it doesn't match a key"
  ):
    pyconfig.initialize(["test.py", _base_cfg_path()])


def test_env_validation_not_uppercase(monkeypatch):
  if os.name != "nt":
    monkeypatch.setenv("M_model_name", "value")  # lowercase

    with pytest.raises(
      ValueError, match="We received env `M_model_name` but it isn't all uppercase"
    ):
      pyconfig.initialize(["test.py", _base_cfg_path()])


# Test CLI and ENV conflicts
def test_cli_env_conflict_raises_error(monkeypatch):
  monkeypatch.setenv("M_MODEL_NAME", "env-model")

  with pytest.raises(
    ValueError, match="You are passing overrides by both CLI and ENV for `model_name`"
  ):
    pyconfig.initialize(["test.py", _base_cfg_path()], model_name="cli-model")


def test_unknown_cli_key_raises_error():
  with pytest.raises(
    ValueError,
    match="Key unknown_key was passed at the command line but isn't in config",
  ):
    pyconfig.initialize(["test.py", _base_cfg_path()], unknown_key="value")


# Test hardware config not found
def test_hardware_config_not_found():
  with pytest.raises(ValueError, match="Hardware config not found for 'nonexistent'"):
    pyconfig.initialize(["test.py", _base_cfg_path()], hardware="nonexistent")


# Test model config loading
def test_model_config_loading():
  cfg = pyconfig.initialize(["test.py", _base_cfg_path()], model_name="llama3.2-1b")
  keys = cfg.get_keys()
  # Should load model-specific config
  assert keys["model_name"] == "llama3.2-1b"


def test_model_config_override_mode():
  # Test override_model_config=True prevents model config from overriding CLI
  cfg = pyconfig.initialize(
    ["test.py", _base_cfg_path()],
    model_name="llama3.2-1b",
    override_model_config=True,
    base_emb_dim=999,
  )
  keys = cfg.get_keys()
  assert keys["base_emb_dim"] == 999  # CLI value preserved


# Test user_init behavior
def test_user_init_with_jobset_name(monkeypatch):
  monkeypatch.setenv("JOBSET_NAME", "test-job")

  cfg = pyconfig.initialize(["test.py", _base_cfg_path()], run_name="")
  assert cfg.get_keys()["run_name"] == "test-job"


def test_user_init_derived_dimensions():
  cfg = pyconfig.initialize(
    ["test.py", _base_cfg_path()],
    base_emb_dim=128,
    base_num_query_heads=8,
    base_num_kv_heads=4,
    base_mlp_dim=256,
    base_num_decoder_layers=12,
  )
  keys = cfg.get_keys()

  assert keys["emb_dim"] == 128
  assert keys["num_query_heads"] == 8
  assert keys["num_kv_heads"] == 4
  assert keys["mlp_dim"] == 256
  assert keys["num_decoder_layers"] == 12


# Test HyperParameters class behavior
def test_hyperparameters_getattr():
  cfg = pyconfig.initialize(["test.py", _base_cfg_path()])
  assert hasattr(cfg, "model_name")
  assert cfg.model_name == "default"


def test_hyperparameters_setattr_forbidden():
  cfg = pyconfig.initialize(["test.py", _base_cfg_path()])

  with pytest.raises(ValueError, match="Reinitialization of config is not allowed"):
    cfg.model_name = "new_value"


# Test type conversion from CLI/ENV
def test_type_conversion_from_env(monkeypatch):
  monkeypatch.setenv("M_BASE_EMB_DIM", "256")  # string -> int
  monkeypatch.setenv("M_ACTIVATIONS_IN_FLOAT32", "true")  # string -> bool

  cfg = pyconfig.initialize(["test.py", _base_cfg_path()])
  keys = cfg.get_keys()

  assert keys["base_emb_dim"] == 256
  assert keys["activations_in_float32"] is True


def test_invalid_type_conversion(monkeypatch):
  monkeypatch.setenv("M_BASE_EMB_DIM", "not_a_number")

  with pytest.raises(
    ValueError,
    match="Couldn't parse value from CLI or ENV 'not_a_number' for key 'base_emb_dim'",
  ):
    pyconfig.initialize(["test.py", _base_cfg_path()])
