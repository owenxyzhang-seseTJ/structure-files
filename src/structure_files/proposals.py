"""Two-phase proposal/apply workflow with source SHA-256 protection."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock

from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    ensure_not_source,
    now_iso,
    resolve_input_path,
    safe_stem,
    sha256_file,
)


def _default_target(source: Path, fmt: str, suffix: str | None = None) -> Path:
    extension = suffix or {
        "cif": ".cif",
        "poscar": ".vasp",
        "extxyz": ".extxyz",
        "xyz": ".xyz",
    }.get(fmt.lower(), ".out")
    return source.parent / f"{safe_stem(source)}.processed{extension}"


def create_proposal(
    *,
    operation: str,
    sources: Iterable[str | Path],
    artifacts: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    requested_output: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_paths = [resolve_input_path(path, must_exist=True, directory=False) for path in sources]
    if not source_paths:
        raise ValueError("proposal 至少需要一个源文件")
    root = Path(output_dir).expanduser().resolve() if output_dir else source_paths[0].parent
    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / ".dsh-structure-proposals"
    staging_root.mkdir(parents=True, exist_ok=True)
    proposal_id = f"{now_iso().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:12]}"
    stage = staging_root / proposal_id
    stage.mkdir(mode=0o700)
    written: list[dict[str, Any]] = []
    try:
        for index, artifact in enumerate(artifacts):
            name = str(artifact.get("name") or f"candidate-{index + 1}.dat")
            # Artifacts are names, not paths supplied to a shell.  Keep them
            # within the proposal directory even if a caller includes "../".
            name_path = Path(name)
            if name_path.is_absolute() or ".." in name_path.parts:
                raise ValueError(f"proposal artifact 名称越界: {name}")
            candidate = (stage / name_path).resolve()
            if stage not in candidate.parents:
                raise ValueError(f"proposal artifact 路径越界: {name}")
            data = artifact.get("data", b"")
            if isinstance(data, str):
                data = data.encode("utf-8")
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError(f"proposal artifact 数据不是 bytes/str: {name}")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(candidate, bytes(data), overwrite=False)
            written.append(
                {
                    "index": index,
                    "name": str(candidate.relative_to(stage)),
                    "sha256": sha256_file(candidate),
                    "bytes": candidate.stat().st_size,
                    "label": artifact.get("label") or f"candidate-{index + 1}",
                    "metadata": artifact.get("metadata") or {},
                }
            )
        primary = source_paths[0]
        target = Path(requested_output).expanduser().resolve() if requested_output else None
        if target is None:
            fmt = str((metadata or {}).get("output_format") or "cif")
            target = _default_target(primary, fmt)
        for source in source_paths:
            ensure_not_source(source, target)
        manifest = {
            "schema_version": 1,
            "proposal_id": proposal_id,
            "created_at": now_iso(),
            "operation": operation,
            "source_files": [
                {"path": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size}
                for source in source_paths
            ],
            "requested_output": str(target),
            "stage_directory": str(stage),
            "artifacts": written,
            "metadata": metadata or {},
        }
        manifest_path = stage / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return {
            "ok": True,
            "proposal_id": proposal_id,
            "manifest_path": str(manifest_path),
            **manifest,
        }
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = resolve_input_path(path, must_exist=True, directory=False)
    if manifest_path.name != "manifest.json":
        raise ValueError("proposal_apply 需要 proposal 目录中的 manifest.json")
    import orjson

    try:
        manifest = orjson.loads(manifest_path.read_bytes())
    except Exception as exc:
        raise ValueError(f"proposal manifest 不是有效 JSON: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的 proposal schema_version")
    return manifest, manifest_path.parent


def apply_proposal(
    manifest_path: str | Path,
    *,
    candidate_index: int = 0,
    destination: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest, stage = load_manifest(manifest_path)
    lock = FileLock(str(stage / ".apply.lock"), timeout=30)
    with lock:
        source_rows = manifest.get("source_files") or []
        if not source_rows:
            raise ValueError("proposal 没有 source_files")
        for row in source_rows:
            source = resolve_input_path(row["path"], must_exist=True, directory=False)
            actual = sha256_file(source)
            if actual != row.get("sha256"):
                raise RuntimeError(
                    "源文件已变化，拒绝应用 proposal: "
                    f"{source}（期望 {row.get('sha256')}，当前 {actual}）"
                )
        artifacts = manifest.get("artifacts") or []
        if not 0 <= int(candidate_index) < len(artifacts):
            raise IndexError(f"candidate_index 超出范围: {candidate_index}")
        artifact = artifacts[int(candidate_index)]
        candidate = (stage / str(artifact["name"])).resolve()
        if stage not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"proposal artifact 不存在或越界: {candidate}")
        if sha256_file(candidate) != artifact.get("sha256"):
            raise RuntimeError("proposal artifact 已变化，拒绝应用")
        target = (
            Path(destination).expanduser().resolve()
            if destination
            else Path(manifest["requested_output"]).expanduser().resolve()
        )
        for row in source_rows:
            ensure_not_source(Path(row["path"]).expanduser().resolve(), target)
        if target.exists() and not overwrite:
            raise FileExistsError(f"目标已存在，拒绝覆盖: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = candidate.read_bytes()
        atomic_write_bytes(target, data, overwrite=overwrite)
        audit = {
            "schema_version": 1,
            "applied_at": now_iso(),
            "proposal_id": manifest.get("proposal_id"),
            "operation": manifest.get("operation"),
            "source_files": source_rows,
            "selected_artifact": artifact,
            "output_path": str(target),
            "output_sha256": sha256_file(target),
            "metadata": manifest.get("metadata") or {},
        }
        audit_path = Path(str(target) + ".audit.json")
        atomic_write_json(audit_path, audit, overwrite=overwrite)
        return {
            "ok": True,
            "proposal_id": manifest.get("proposal_id"),
            "output_path": str(target),
            "audit_path": str(audit_path),
            "output_sha256": audit["output_sha256"],
            "selected_artifact": artifact,
        }
