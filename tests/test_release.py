import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_version", ROOT / "tools/check_version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, version="0.1.0"):
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "wav-compare"\nversion = "{version}"\n'
    )
    (root / "uv.lock").write_text(f'[[package]]\nname = "wav-compare"\nversion = "{version}"\n')
    (root / ".release-please-manifest.json").write_text(json.dumps({".": version}))
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## {version}\n\n### Features\n- Audio workbench\n"
    )


def test_source_versions_are_consistent():
    assert module.check(ROOT)


def test_release_tag_and_changelog_are_checked(tmp_path):
    fixture(tmp_path)
    assert module.check(tmp_path, "v0.1.0") == "0.1.0"
    with pytest.raises(ValueError, match="Tag"):
        module.check(tmp_path, "v0.2.0")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n")
    with pytest.raises(ValueError, match="changelog"):
        module.check(tmp_path, "v0.1.0")


def test_stale_lock_fails(tmp_path):
    fixture(tmp_path)
    (tmp_path / "uv.lock").write_text('[[package]]\nname="wav-compare"\nversion="0.0.0"\n')
    with pytest.raises(ValueError, match="mismatch"):
        module.check(tmp_path)
