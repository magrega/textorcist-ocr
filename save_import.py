"""Local, byte-preserving save transplantation. No game files are executed."""
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


def split_save(path: Path) -> tuple[bytes, bytes, bytes]:
    if path.suffix.lower() != ".dat" or not path.is_file() or path.is_symlink():
        raise ValueError("Choose an existing, regular .dat save file.")
    if path.stat().st_size > MAX_SIZE:
        raise ValueError("Save exceeds the 8 MB safety limit.")
    try:
        data = base64.b64decode(b"".join(path.read_bytes().split()), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Save is not valid Base64.") from exc
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
        raise ValueError("Unrecognized save format: expected one JSON object followed by optional null/whitespace bytes.")
    start, end = candidates[0]
    return data[:start], data[start:end], data[end:]


def prepare_import(source: Path, target: Path) -> bytes:
    if source.resolve() == target.resolve():
        raise ValueError("Incoming and current saves must be different files.")
    prefix, _, suffix = split_save(target)
    _, payload, _ = split_save(source)
    return base64.b64encode(prefix + payload + suffix)


def import_save(source: Path, target: Path) -> Path:
    replacement = prepare_import(source, target)
    backup_root = target.parent / "helper-backups"
    backup_root.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="save-", dir=backup_root))
    # Complete all backups before touching the destination.
    for file in target.parent.iterdir():
        if file.suffix.lower() == ".dat" and file.is_file():
            if file.is_symlink():
                raise ValueError("Refusing to back up a linked save file.")
            shutil.copy2(file, backup / file.name)
            if (backup / file.name).read_bytes() != file.read_bytes():
                raise OSError("Backup verification failed; destination was not changed.")
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
    return backup
