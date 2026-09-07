"""I/O, hashing, and conservative path helpers."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

NUMBER_RE = re.compile(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_cif_number(value: Any, default: float | None = None) -> float | None:
    """Parse CIF numbers, including standard uncertainty notation ``12.3(4)``."""
    if value is None:
        return default
    text = str(value).strip().strip("'").strip('"')
    if text in {"", ".", "?", "none", "None"}:
        return default
    match = NUMBER_RE.match(text)
    if not match:
        return default
    try:
        return float(match.group(0).replace("D", "E").replace("d", "e"))
    except ValueError:
        return default


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"无法 JSON 序列化: {type(value)!r}")


def dumps(value: Any, *, indent: bool = False) -> bytes:
    option = orjson.OPT_SORT_KEYS
    if indent:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(value, option=option, default=json_default)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"目标已存在，拒绝覆盖: {target}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"目标已存在，拒绝覆盖: {target}")
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return target


def atomic_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, dumps(value, indent=True) + b"\n", overwrite=overwrite)


def resolve_input_path(
    value: str | Path, *, must_exist: bool = True, directory: bool | None = None
) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("缺少路径")
    path = Path(str(value)).expanduser()
    # Resolve symlinks for reads so an audit names the actual object.  Writes are
    # always performed by atomic_write_* in the requested parent directory.
    resolved = path.resolve(strict=False)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"文件不存在: {resolved}")
    if directory is True and resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(str(resolved))
    if directory is False and resolved.exists() and not resolved.is_file():
        raise IsADirectoryError(str(resolved))
    return resolved


def ensure_not_source(source: Path, target: Path) -> None:
    if source.resolve(strict=False) == target.resolve(strict=False):
        raise ValueError("输出路径不能覆盖源文件；请使用新的文件名")


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "structure"


def extension_for_format(fmt: str) -> str:
    return {
        "cif": ".cif",
        "mcif": ".cif",
        "poscar": ".vasp",
        "vasp": ".vasp",
        "contcar": ".vasp",
        "extxyz": ".extxyz",
        "xyz": ".xyz",
        "json": ".json",
    }.get(fmt.lower(), ".dat")


def format_float(value: float, digits: int = 8) -> str:
    if abs(float(value)) < 5e-13:
        value = 0.0
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".") or "0"
