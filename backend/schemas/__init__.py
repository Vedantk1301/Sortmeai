"""
Helpers for accessing JSON schemas used by the Sortme system.
"""

import json
from importlib import resources
from typing import Any, Dict


def load_schema(name: str) -> Dict[str, Any]:
    with resources.files(__package__).joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["load_schema"]
