from utils import max_logging


def test_log_prints_without_error(capsys):
  max_logging.log("hello")
  out = capsys.readouterr().out
  assert "hello" in out
