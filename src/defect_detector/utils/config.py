"""Small helper to load the project's YAML config from anywhere in the codebase."""
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to project root) into an absolute Path."""
    return PROJECT_ROOT / relative_path