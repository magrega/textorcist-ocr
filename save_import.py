"""Local, byte-preserving save transplantation. No game files are executed.

Follows the Steam guide workflow: Base64-decode each .dat, keep the local
installation's binary prefix (and trailing bytes), replace only the JSON
progress object from the incoming save, then Base64-encode again.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
import shutil
import tempfile

MAX_SIZE = 8 * 1024 * 1024


def default_save_directory() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Textorcist"


def list_dat_files(folder: Path) -> list[Path]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Choose an existing save folder.")
    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".dat"
    )
    if not files:
        raise ValueError(f"No .dat saves found in {folder}")
    return files


def split_save(path: Path) -> tuple[bytes, bytes, bytes]:
    if path.suffix.lower() != ".dat" or not path.is_file() or path.is_symlink():
        raise ValueError("Choose an existing, regular .dat save file.")
    if path.stat().st_size > MAX_SIZE:
        raise ValueError(f"{path.name} exceeds the 8 MB safety limit.")
    try:
        data = base64.b64decode(b"".join(path.read_bytes().split()), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{path.name} is not valid Base64.") from exc
    # Latin-1 is a reversible byte mapping, not a conversion of the binary header.
    text = data.decode("latin-1")
    decoder = json.JSONDecoder()
    candidates = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, size = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        end = start + size
        if isinstance(obj, dict):
            candidates.append((start, end))
    valid: list[tuple[int, int]] = []
    for start, end in candidates:
        if end == len(text):
            valid.append((start, end))
        elif text[end:].strip("\x00\r\n\t ") == "":
            valid.append((start, end))
    if len(valid) != 1:
        raise ValueError(
            f"{path.name}: unrecognized save format — expected one JSON object "
            "followed by optional null/whitespace bytes."
        )
    start, end = valid[0]
    return data[:start], data[start:end], data[end:]


def prepare_import(source: Path, target: Path) -> bytes:
    if source.resolve() == target.resolve():
        raise ValueError("Incoming and current saves must be different files.")
    prefix, _, suffix = split_save(target)
    _, payload, _ = split_save(source)
    return base64.b64encode(prefix + payload + suffix)


def prepare_folder_import(source_dir: Path, target_dir: Path) -> dict[str, bytes]:
    """Validate every matching .dat pair and return encoded replacements by name."""
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("Incoming and current save folders must be different.")
    sources = {path.name.lower(): path for path in list_dat_files(source_dir)}
    targets = {path.name.lower(): path for path in list_dat_files(target_dir)}
    missing = sorted(set(sources) - set(targets))
    if missing:
        raise ValueError(
            "Launch the game once so it creates local saves for: " + ", ".join(missing)
        )
    replacements: dict[str, bytes] = {}
    for name, source in sources.items():
        target = targets[name]
        try:
            replacements[target.name] = prepare_import(source, target)
        except ValueError as exc:
            raise ValueError(f"Cannot import {target.name}: {exc}") from exc
    return replacements


def _backup_dat_files(target_dir: Path) -> Path:
    backup_root = target_dir / "helper-backups"
    backup_root.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="save-", dir=backup_root))
    for file in list_dat_files(target_dir):
        shutil.copy2(file, backup / file.name)
        if (backup / file.name).read_bytes() != file.read_bytes():
            raise OSError("Backup verification failed; destination was not changed.")
    return backup


def _write_replacement(target: Path, replacement: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".helper-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_save(source: Path, target: Path) -> Path:
    replacement = prepare_import(source, target)
    backup = _backup_dat_files(target.parent)
    _write_replacement(target, replacement)
    return backup


def import_folder(source_dir: Path, target_dir: Path) -> tuple[Path, list[str]]:
    """Back up all local .dat files, then transplant JSON from every incoming save."""
    replacements = prepare_folder_import(source_dir, target_dir)
    backup = _backup_dat_files(target_dir)
    written: list[str] = []
    try:
        for name, payload in replacements.items():
            _write_replacement(target_dir / name, payload)
            written.append(name)
    except Exception:
        # Restore every file we already replaced from the backup.
        for name in written:
            shutil.copy2(backup / name, target_dir / name)
        raise
    return backup, written
