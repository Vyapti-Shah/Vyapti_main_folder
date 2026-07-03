"""Filesystem helper utilities."""

import os


def create_directory(path: str) -> None:
    """Creates a directory (including parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def build_path(*parts: str) -> str:
    """Joins path components into a single path string."""
    return os.path.join(*parts)
