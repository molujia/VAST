"""Installation metadata must retain resources required by public APIs."""
from pathlib import Path
import tomllib


def test_annotation_schema_is_declared_as_package_data():
    root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = metadata.get("tool", {}).get("setuptools", {}).get("package-data", {})
    assert "schema.json" in package_data.get("vast_v3.annotations", [])
