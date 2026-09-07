"""Optional H-only ASE calculator adapter.

No ML package is installed by the core preset.  A user may point the service at
an explicit ASE Calculator factory in a separate, validated environment.  The
adapter never moves heavy atoms and records model identity/energy so candidates
from different models are not compared as if their absolute energies were
commensurate.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

import numpy as np

from .models import StructureData
from .parsers import from_ase


def _load_factory(spec: str, kwargs: dict[str, Any]) -> Any:
    if ":" not in spec:
        raise ValueError("calculator_factory 必须是 module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory(**kwargs)


def _calculator_from_config(config: dict[str, Any]) -> tuple[Any, str]:
    factory = config.get("calculator_factory") or config.get("factory")
    kwargs = dict(config.get("calculator_kwargs") or config.get("kwargs") or {})
    if factory:
        return _load_factory(str(factory), kwargs), str(factory)
    backend = str(config.get("backend") or "").lower()
    model_path = config.get("model_path")
    if backend == "mace":
        if not model_path:
            raise ValueError(
                "MACE 后端必须显式提供 model_path；为避免文件处理服务隐式下载权重，"
                "请配置本地模型或使用 calculator_factory"
            )
        try:
            from mace.calculators import mace_mp

            return mace_mp(model=str(model_path), default_dtype="float64"), f"mace:{model_path}"
        except Exception as exc:
            raise RuntimeError(f"MACE 后端不可用: {exc}") from exc
    if backend in {"fairchem", "uma", "ocp"}:
        raise RuntimeError(
            "Fairchem/UMA 接口随版本变化，请通过 calculator_factory 提供 ASE Calculator"
        )
    raise ValueError("ML 配置需要 calculator_factory 或 backend='mace'")


def refine_hydrogen_only(
    model: StructureData, config: dict[str, Any] | None
) -> tuple[StructureData, dict[str, Any]]:
    config = dict(config or {})
    if not config.get("enabled"):
        return model, {"status": "disabled", "reason": "ML 默认关闭"}
    if model.cell is None:
        return model, {"status": "unavailable", "reason": "ML H-only 需要周期晶胞"}
    try:
        from ase.constraints import FixAtoms
        from ase.optimize import BFGS

        from .writers import _ase_atoms

        atoms = _ase_atoms(model)
        heavy = [i for i, atom in enumerate(model.atoms) if not atom.is_hydrogen]
        atoms.set_constraint(FixAtoms(indices=heavy))
        calculator, model_id = _calculator_from_config(config)
        atoms.calc = calculator
        optimizer = BFGS(atoms, logfile=None)
        max_steps = int(config.get("max_steps", 100))
        fmax = float(config.get("fmax", 0.05))
        optimizer.run(fmax=fmax, steps=max_steps)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        refined = from_ase(
            atoms,
            format=model.format,
            source_path=model.source_path,
            metadata=deepcopy(model.metadata),
        )
        # ASE arrays carry enough information for a generic round trip, but
        # generated-H provenance (host index, placement kind, disorder notes)
        # belongs to the canonical model.  Copy those records back by atom
        # index so ML refinement cannot erase the audit trail.
        for source_atom, refined_atom in zip(model.atoms, refined.atoms, strict=True):
            refined_atom.label = source_atom.label
            refined_atom.occupancy = source_atom.occupancy
            refined_atom.charge = source_atom.charge
            refined_atom.properties = deepcopy(source_atom.properties)
        refined.metadata.update({"ml_refined": True, "ml_model_id": model_id})
        return refined, {
            "status": "ok",
            "model_id": model_id,
            "energy_eV": energy,
            "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0,
            "max_steps": max_steps,
            "fmax": fmax,
            "heavy_atoms_fixed": len(heavy),
        }
    except Exception as exc:
        if config.get("allow_failure", True):
            return model, {
                "status": "unavailable",
                "reason": str(exc),
                "backend": config.get("backend"),
            }
        raise


def evaluate_models(
    models: list[StructureData], config: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate path frames with one explicit ASE calculator.

    This is a single-point descriptor pass for ranking file-derived candidates.
    It intentionally does not optimize coordinates, construct an NEB band, or
    compare energies from different calculator/model identities.
    """

    config = dict(config or {})
    if not config.get("enabled"):
        return [], {"status": "disabled", "reason": "ML 默认关闭"}
    if not models:
        return [], {"status": "unavailable", "reason": "没有待评估的路径帧"}
    try:
        from .writers import _ase_atoms

        calculator, model_id = _calculator_from_config(config)
        rows: list[dict[str, Any]] = []
        for frame_index, model in enumerate(models):
            atoms = _ase_atoms(model)
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            force_norms = np.linalg.norm(forces, axis=1) if len(forces) else np.asarray([])
            rows.append(
                {
                    "frame_index": frame_index,
                    "energy_eV": energy,
                    "max_force_eV_A": float(np.max(force_norms)) if len(force_norms) else 0.0,
                    "rms_force_eV_A": float(np.sqrt(np.mean(force_norms**2)))
                    if len(force_norms)
                    else 0.0,
                    "model_id": model_id,
                }
            )
        return rows, {
            "status": "ok",
            "model_id": model_id,
            "frame_count": len(rows),
            "purpose": "same-model single-point ranking only",
        }
    except Exception as exc:
        if config.get("allow_failure", True):
            return [], {
                "status": "unavailable",
                "reason": str(exc),
                "backend": config.get("backend"),
                "purpose": "geometry-only ranking retained",
            }
        raise
