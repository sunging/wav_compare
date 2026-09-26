"""Check the source, release manifest and locked local package agree."""

import argparse
import json
import tomllib
from pathlib import Path


def check(root=Path("."), tag=None):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    manifest = json.loads((root / ".release-please-manifest.json").read_text())["."]
    packages = tomllib.loads((root / "uv.lock").read_text())["package"]
    locked = next(p["version"] for p in packages if p["name"] == project["name"])
    if not version == manifest == locked:
        raise ValueError(f"Version mismatch: project={version}, manifest={manifest}, lock={locked}")
    if tag:
        if tag != f"v{version}":
            raise ValueError(f"Tag {tag} does not match {version}")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        if f"## {version}" not in changelog and f"## [{version}]" not in changelog:
            raise ValueError(f"No changelog entry for {version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    print(check(tag=parser.parse_args().tag))
