"""Security and data minimization assertions ensuring unminimized payloads do not leak."""

from typing import Any

from pydantic import BaseModel


def assert_data_minimization(target: Any, forbidden_substring: str) -> None:
    """Recursively verify that forbidden substrings or raw payload fields are not present."""
    if isinstance(target, str):
        if forbidden_substring in target:
            raise AssertionError(
                f"Data minimization violation: found '{forbidden_substring}' in string"
            )
    elif isinstance(target, BaseModel):
        data = target.model_dump(mode="json")
        assert_data_minimization(data, forbidden_substring)
    elif isinstance(target, dict):
        for k, v in target.items():
            if k == "raw_payload":
                raise AssertionError(
                    "Data minimization violation: 'raw_payload' key present in state"
                )
            assert_data_minimization(k, forbidden_substring)
            assert_data_minimization(v, forbidden_substring)
    elif isinstance(target, (list, tuple, set)):
        for item in target:
            assert_data_minimization(item, forbidden_substring)
