from pathlib import Path

from globals import PKG_DIR


def test_pkg_dir_points_to_project_root():
  pkg = Path(PKG_DIR)
  assert (pkg / "configs").exists()
  assert (pkg / "layers").exists()
