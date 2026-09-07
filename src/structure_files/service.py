"""High-level service actions exposed to the MCP and CLI front ends."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .geometry import build_bond_graph, graph_to_dict, validate_geometry
from .hydrogen import propose_hydrogens, score_candidate
from .ml import refine_hydrogen_only
from .operations import transform_structure
from .outcar import OutcarResult, parse_outcar
from .parsers import detect_format, expand_symmetry, parse_structure
from .proposals import apply_proposal, create_proposal
from .ts_search import search_transition_state_candidates
from .utils import (
    dumps,
    extension_for_format,
    resolve_input_path,
    safe_stem,
    sha256_file,
)
from .writers import cif_text, structure_text


def _fmt(value: str | None) -> str:
    return str(value or "").strip().lower()


def _decode_value(value: Any) -> Any:
    """Decode JSON-shaped MCP strings while accepting compact ``1 0 0`` forms."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text)
        except Exception:
            return value
    if "," in text or " " in text or ";" in text:
        bits = [x for x in text.replace(",", " ").replace(";", " ").split() if x]
        try:
            return [float(x) for x in bits]
        except ValueError:
            return value
    return value


def _composition_dict_from_pymatgen(structure: Any) -> dict[str, float]:
    try:
        values = structure.composition.get_el_amt_dict()
    except Exception:
        return {}
    return {str(key): float(value) for key, value in sorted(values.items())}


def _site_composition_from_pymatgen(structure: Any) -> dict[str, float]:
    """Count one entry per parsed site, unlike the occupancy-weighted formula."""

    counts: dict[str, float] = {}
    for site in getattr(structure, "sites", []):
        species = getattr(site, "species", None)
        if species is None:
            continue
        try:
            # A disordered site remains one parsed site.  Its first species is
            # used only for the unweighted parser-shape comparison; the full
            # occupancy-weighted composition is retained separately below.
            symbol = next(iter(species)).symbol
        except Exception:
            symbol = getattr(getattr(site, "specie", None), "symbol", None)
        if symbol:
            text = str(symbol)
            counts[text] = counts.get(text, 0.0) + 1.0
    return dict(sorted(counts.items()))


def _composition_dict_from_symbols(symbols: list[str]) -> dict[str, float]:
    return {
        str(symbol): float(symbols.count(symbol))
        for symbol in sorted(set(symbols))
    }


def _parser_agreement(
    result: dict[str, Any], left: str = "pymatgen", right: str = "ase"
) -> bool | None:
    """Compare independent parser outputs without relying on formula ordering."""

    if left not in result or right not in result:
        return None
    a, b = result[left], result[right]
    count_equal = a.get("n_atoms") == b.get("n_atoms")
    composition_equal = a.get("composition") == b.get("composition")
    weighted_equal = a.get("composition_weighted") == b.get("composition_weighted")
    result["atom_count_agree"] = bool(count_equal)
    result["composition_agree"] = bool(composition_equal)
    if "composition_weighted" in a and "composition_weighted" in b:
        result["weighted_composition_agree"] = bool(weighted_equal)
    return bool(count_equal and composition_equal)


def _independent_stats(
    path: Path, fmt: str, *, _normalize_cif: bool = True
) -> dict[str, Any]:
    result: dict[str, Any] = {"available": True, "format": fmt}
    try:
        if fmt == "cif":
            from pymatgen.io.cif import CifParser

            structs = CifParser(str(path), occupancy_tolerance=1.0).parse_structures(
                primitive=False, check_occu=False
            )
            if structs:
                result["pymatgen"] = {
                    "n_atoms": len(structs[0]),
                    "formula": structs[0].composition.formula,
                    "composition": _site_composition_from_pymatgen(structs[0]),
                    "composition_weighted": _composition_dict_from_pymatgen(structs[0]),
                    "lattice": [float(x) for x in structs[0].lattice.abc],
                }
        elif fmt == "poscar":
            from pymatgen.io.vasp import Poscar

            try:
                obj = Poscar.from_file(str(path), check_for_potcar=False)
            except TypeError:  # compatibility with older Pymatgen releases
                obj = Poscar.from_file(str(path), check_for_POTCAR=False)
            result["pymatgen"] = {
                "n_atoms": len(obj.structure),
                "formula": obj.structure.composition.formula,
                "composition": _site_composition_from_pymatgen(obj.structure),
                "composition_weighted": _composition_dict_from_pymatgen(obj.structure),
            }
        elif fmt in {"xyz", "extxyz"}:
            from ase import io as ase_io

            obj = ase_io.read(str(path), index=-1)
            symbols = obj.get_chemical_symbols()
            result["pymatgen"] = {
                "n_atoms": len(obj),
                "formula": "".join(symbols),
                "composition": _composition_dict_from_symbols(symbols),
                "composition_weighted": _composition_dict_from_symbols(symbols),
            }
    except Exception as exc:
        result["pymatgen_error"] = str(exc)
    try:
        import ase.io

        if fmt == "cif":
            atoms = ase.io.read(str(path), format="cif", index=0)
        elif fmt == "poscar":
            atoms = ase.io.read(str(path), format="vasp", index=-1)
        else:
            atoms = ase.io.read(str(path), format=fmt, index=-1)
        symbols = atoms.get_chemical_symbols()
        result["ase"] = {
            "n_atoms": len(atoms),
            "symbols": symbols,
            "formula": atoms.get_chemical_formula(),
            "composition": _composition_dict_from_symbols(symbols),
            "composition_weighted": _composition_dict_from_symbols(symbols),
        }
    except Exception as exc:
        result["ase_error"] = str(exc)
    direct_agreement = _parser_agreement(result)

    # A valid CIF may omit its symmetry-operation loop (Pymatgen then defaults
    # to P1), or declare a full setting string that ASE does not recognize.
    # Emit a temporary, in-memory-equivalent CIF containing Gemmi's explicit
    # operations and ASE-compatible short name.  The source is never changed;
    # both the raw parser results above and this normalized cross-check remain
    # visible in the audit.
    if _normalize_cif and fmt == "cif":
        try:
            from .parsers import parse_cif

            model = parse_cif(path)
            operations = list(model.metadata.get("symmetry_operations") or [])
            short_name = model.metadata.get("canonical_short_space_group")
            closed = bool(
                operations and bool(expand_symmetry(model).metadata.get("symmetry_closed"))
            )
            if operations and short_name and (direct_agreement is not True or closed):
                normalized = model.copy()
                if closed:
                    normalized.metadata["declared_space_group"] = "P 1"
                    normalized.metadata["symmetry_operations"] = ["x,y,z"]
                    normalized.metadata["normalization_representation"] = (
                        "symmetry-closed input emitted as explicit P1"
                    )
                else:
                    normalized.metadata["declared_space_group"] = str(short_name)
                    normalized.metadata["normalization_representation"] = (
                        "asymmetric-unit input emitted with Gemmi operations"
                    )
                normalized.metadata["expanded_symmetry"] = False
                # Import lazily to keep parser diagnostics available in minimal
                # environments where the writer dependencies are absent.
                text = cif_text(normalized, expand_p1=False)
                with tempfile.TemporaryDirectory(prefix="dsh-cif-normalized-") as temp_dir:
                    temp_path = Path(temp_dir) / "normalized.cif"
                    temp_path.write_text(text, encoding="utf-8")
                    normalized_stats = _independent_stats(
                        temp_path, "cif", _normalize_cif=False
                    )
                normalized_stats.pop("path", None)
                normalized_stats["method"] = (
                    "symmetry-closed input emitted as explicit P1"
                    if closed
                    else "Gemmi operations + ASE-compatible short space-group name"
                )
                normalized_stats["canonical_space_group"] = model.metadata.get(
                    "canonical_space_group"
                )
                normalized_stats["canonical_short_space_group"] = short_name
                result["normalized"] = normalized_stats
                result["normalized_agreement"] = _parser_agreement(
                    normalized_stats
                )
        except Exception as exc:
            result["normalization_error"] = str(exc)
    return result


def _cif_representation(model: Any) -> dict[str, Any]:
    metadata = model.metadata
    raw = int(metadata.get("raw_atom_count", model.n_atoms))
    expanded_count = None
    operations = metadata.get("symmetry_operations") or []
    symmetry_closed = False
    if operations:
        try:
            expanded_model = expand_symmetry(model)
            expanded_count = expanded_model.n_atoms
            symmetry_closed = bool(expanded_model.metadata.get("symmetry_closed"))
        except Exception:
            expanded_count = None
    occupancy = [float(a.occupancy) for a in model.atoms if abs(a.occupancy - 1.0) > 1e-8]
    disorder = [
        a.label
        for a in model.atoms
        if any(k in a.properties for k in ("disorder_group", "alt_id", "disorder_assembly"))
    ]
    if metadata.get("expanded_symmetry"):
        kind = "expanded_or_viewer_oriented"
    elif disorder or occupancy:
        # Occupancy/disorder is the chemically important classification.  A
        # P1/identity operation can make those rows symmetry-closed as well,
        # but that must not hide the fact that the model is not an ordered
        # structure suitable for POSCAR export without an explicit policy.
        kind = "disorder_or_partial_occupancy"
    elif symmetry_closed:
        kind = "pre_expanded_or_symmetry_closed"
    elif operations and expanded_count and expanded_count > raw:
        kind = "asymmetric_unit_candidate"
    else:
        kind = "P1_or_already_expanded"
    return {
        "classification": kind,
        "raw_atom_count": raw,
        "declared_space_group": metadata.get("declared_space_group"),
        "symmetry_operation_count": len(operations),
        "expanded_atom_count_estimate": expanded_count,
        "symmetry_closed_under_operations": symmetry_closed,
        "partial_occupancy_sites": occupancy,
        "disorder_labels": disorder,
    }


def _outcar_summary(result: OutcarResult) -> dict[str, Any]:
    last = result.steps[-1] if result.steps else None
    return {
        "path": result.path,
        "nions": result.nions,
        "elements": result.elements,
        "complete_steps": len(result.steps),
        "truncated_position_block": result.truncated_position_block,
        "warnings": result.warnings,
        "last_step": None
        if last is None
        else {
            "index": last.index,
            "energy_free_eV": last.energy_free_eV,
            "energy_sigma0_eV": last.energy_sigma0_eV,
            "has_forces": last.forces is not None,
            "has_cell": last.cell is not None,
        },
    }


def _composition_signature(model: Any) -> tuple[tuple[str, float], ...]:
    """Canonical composition key used to gate optional ML comparisons."""

    return tuple(
        sorted((str(atom.element), round(float(atom.occupancy), 8)) for atom in model.atoms)
    )


def _rank_hydrogen_candidates(
    records: list[dict[str, Any]], *, ml_requested: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank H candidates without comparing incompatible compositions.

    Geometry scores are comparable across the deterministic enumerator, while
    ML energies are only meaningful for candidates with identical composition
    evaluated by the same model.  We therefore sort by ML energy *within* each
    such group and order groups by their best geometry candidate.
    """

    groups: dict[tuple[tuple[str, float], ...], list[dict[str, Any]]] = {}
    for record in records:
        key = _composition_signature(record["model"])
        record["composition_signature"] = [list(row) for row in key]
        groups.setdefault(key, []).append(record)
    group_order = sorted(
        groups,
        key=lambda key: min(
            (float(row["base_score"]), int(row["original_index"])) for row in groups[key]
        ),
    )
    audit_groups: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    comparable_count = 0
    for group_index, key in enumerate(group_order):
        members = groups[key]
        ml_ready = (
            ml_requested
            and len(members) > 1
            and all(
                row["ml"].get("status") == "ok"
                and row["ml"].get("energy_eV") is not None
                for row in members
            )
        )
        if ml_ready:
            comparable_count += 1
            members.sort(
                key=lambda row: (
                    float(row["ml"]["energy_eV"]),
                    float(row["base_score"]),
                    int(row["original_index"]),
                )
            )
            for rank, row in enumerate(members, start=1):
                row["ml_rank_within_composition"] = rank
                row["ranking_basis"] = "same-model ML energy within identical composition"
        else:
            members.sort(key=lambda row: (float(row["base_score"]), int(row["original_index"])))
            for row in members:
                row["ml_rank_within_composition"] = None
                row["ranking_basis"] = "geometry score; ML comparison withheld"
        audit_groups.append(
            {
                "group_index": group_index,
                "composition": [list(row) for row in key],
                "candidate_count": len(members),
                "ml_comparable": bool(ml_ready),
                "model_ids": sorted(
                    {
                        str(row["ml"].get("model_id"))
                        for row in members
                        if row["ml"].get("model_id") is not None
                    }
                ),
            }
        )
        ordered.extend(members)
    return ordered, {
        "requested": bool(ml_requested),
        "comparison_scope": "same calculator model and identical composition only",
        "groups": audit_groups,
        "comparable_group_count": comparable_count,
    }


class StructureService:
    def scan(self, path: str, *, recursive: bool = True, max_files: int = 1000) -> dict[str, Any]:
        target = resolve_input_path(path, must_exist=True)
        files = [target] if target.is_file() else []
        if target.is_dir():
            iterator = target.rglob("*") if recursive else target.glob("*")
            for candidate in iterator:
                if len(files) >= int(max_files):
                    break
                if candidate.is_file() and ".dsh-structure-proposals" not in candidate.parts:
                    files.append(candidate)
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for candidate in files:
            fmt = detect_format(candidate)
            row: dict[str, Any] = {
                "path": str(candidate),
                "name": candidate.name,
                "format": fmt,
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
            if fmt in {"cif", "poscar", "xyz", "extxyz"}:
                try:
                    model = parse_structure(candidate, fmt)
                    row.update(
                        {"parse_ok": True, "n_atoms": model.n_atoms, "formula": model.formula()}
                    )
                except Exception as exc:
                    row.update({"parse_ok": False, "parse_error": str(exc)})
            elif fmt == "outcar":
                try:
                    row.update({"parse_ok": True, **_outcar_summary(parse_outcar(candidate))})
                except Exception as exc:
                    row.update({"parse_ok": False, "parse_error": str(exc)})
            else:
                row["parse_ok"] = False
            rows.append(row)
            counts[fmt] = counts.get(fmt, 0) + 1
        return {
            "ok": True,
            "root": str(target),
            "n_files": len(rows),
            "format_counts": counts,
            "files": rows,
        }

    def inspect(
        self, path: str, *, expand_symmetry: bool = False, include_atoms: bool = False
    ) -> dict[str, Any]:
        target = resolve_input_path(path, must_exist=True, directory=False)
        fmt = detect_format(target)
        if fmt == "outcar":
            result = parse_outcar(target)
            return {
                "ok": True,
                "format": fmt,
                "sha256": sha256_file(target),
                "summary": _outcar_summary(result),
            }
        model = parse_structure(target, fmt)
        if expand_symmetry and fmt == "cif":
            model = globals()["expand_symmetry"](model)
        data = model.as_dict(include_coords=include_atoms)
        data.update({"ok": True, "sha256": sha256_file(target), "format": fmt})
        if fmt == "cif":
            data["cif_representation"] = _cif_representation(parse_structure(target, fmt))
        return data

    def validate(
        self,
        path: str,
        *,
        expand_symmetry: bool = True,
        contact_limit: int = 30,
        symprec: float = 0.1,
    ) -> dict[str, Any]:
        target = resolve_input_path(path, must_exist=True, directory=False)
        fmt = detect_format(target)
        if fmt == "outcar":
            result = parse_outcar(target)
            return {
                "ok": bool(result.steps),
                "format": fmt,
                "sha256": sha256_file(target),
                "summary": _outcar_summary(result),
            }
        raw = parse_structure(target, fmt)
        analyzed = globals()["expand_symmetry"](raw) if fmt == "cif" and expand_symmetry else raw
        geometry = validate_geometry(analyzed, contact_limit=contact_limit, symprec=symprec)
        independent = _independent_stats(target, fmt)
        warnings: list[str] = []
        direct_parser_agreement = independent.get("atom_count_agree")
        normalized_parser_agreement = independent.get("normalized_agreement")
        parser_agreement = (
            normalized_parser_agreement
            if normalized_parser_agreement is not None
            else direct_parser_agreement
        )
        expected_composition = {
            str(element): float(amount)
            for element, amount in analyzed.composition(weighted=False).items()
        }
        expected_parser_match: bool | None = None
        parser_view = independent.get("normalized")
        if parser_view is None:
            parser_view = independent
        if isinstance(parser_view, dict) and "pymatgen" in parser_view and "ase" in parser_view:
            expected_parser_match = all(
                view.get("n_atoms") == analyzed.n_atoms
                and view.get("composition") == expected_composition
                for view in (parser_view["pymatgen"], parser_view["ase"])
            )
        if direct_parser_agreement is False and normalized_parser_agreement is True:
            warnings.append(
                "原始 CIF 的 Pymatgen/ASE 表示不同；已用 Gemmi 生成的规范化对称操作完成交叉解析，"
                "规范化结果一致"
            )
        elif parser_agreement is False:
            warnings.append(
                "Pymatgen 与 ASE 的原子数或元素组成不一致；不能把该文件称为已清洗结构"
            )
        if expected_parser_match is False:
            warnings.append(
                "独立解析器结果与当前 CIF 表示分类不一致；请检查 asymmetric unit、"
                "预展开副本或无序模型"
            )
        elif expected_parser_match is None:
            warnings.append("至少一个独立解析器未返回结构，验证仅作部分诊断")
        if geometry["duplicate_contact_count"]:
            warnings.append(
                "发现小于 0.50 Å 的周期重复/重叠接触，需要结合原始 CIF 对称性和化学角色判断"
            )
        if geometry["suspicious_contact_count"]:
            warnings.append("发现疑似异常短接触；数值阈值只是筛选，不可代替键级和配位化学审查")
        if any(a.occupancy < 0.999 for a in analyzed.atoms):
            warnings.append("存在部分占位率；本输出未自动有序化")
        parser_gate = parser_agreement is True and expected_parser_match is True
        return {
            "ok": bool(not geometry["duplicate_contact_count"] and parser_gate),
            "format": fmt,
            "sha256": sha256_file(target),
            "raw": raw.as_dict(include_coords=False),
            "analyzed": analyzed.as_dict(include_coords=False),
            "independent_parsers": independent,
            "geometry": geometry,
            "warnings": warnings,
            "policy": {
                "source_preserved": True,
                "symmetry_expansion_informational": fmt == "cif" and expand_symmetry,
                "partial_occupancy_preserved": True,
                "independent_parser_agreement": bool(parser_gate),
                "parser_agreement_scope": (
                    "normalized symmetry representation"
                    if normalized_parser_agreement is not None
                    else "source representation"
                ),
            },
        }

    def convert(
        self,
        input_path: str,
        output_format: str,
        *,
        output_path: str | None = None,
        output_dir: str | None = None,
        expand_p1: bool = False,
        occupancy_policy: str = "reject",
    ) -> dict[str, Any]:
        source = resolve_input_path(input_path, must_exist=True, directory=False)
        fmt = _fmt(output_format)
        if fmt not in {"cif", "poscar", "vasp", "extxyz", "xyz"}:
            raise ValueError("output_format 仅支持 cif/poscar/vasp/extxyz/xyz")
        source_fmt = detect_format(source)
        model = parse_structure(source)
        if source_fmt == "cif" and fmt != "cif" and not expand_p1:
            operations = list(model.metadata.get("symmetry_operations") or [])
            if len(operations) > 1 and not model.metadata.get("symmetry_closed"):
                raise ValueError(
                    "CIF 含非平凡空间群操作，POSCAR/XYZ/extXYZ 无法保存对称性；"
                    "请明确设置 expand_p1=True 后再转换"
                )
        if expand_p1 and detect_format(source) == "cif":
            model = globals()["expand_symmetry"](model)
        text = structure_text(model, fmt, expand_p1=expand_p1, occupancy_policy=occupancy_policy)
        suffix = extension_for_format("extxyz" if fmt == "xyz" else fmt)
        target = output_path or str(
            (Path(output_dir).expanduser().resolve() if output_dir else source.parent)
            / f"{safe_stem(source)}.converted{suffix}"
        )
        proposal = create_proposal(
            operation="convert",
            sources=[source],
            artifacts=[
                {
                    "name": f"candidate-1{suffix}",
                    "data": text,
                    "label": "converted",
                    "metadata": {"format": fmt, "n_atoms": model.n_atoms},
                }
            ],
            output_dir=output_dir or str(source.parent),
            requested_output=target,
            metadata={
                "output_format": fmt,
                "expand_p1": expand_p1,
                "occupancy_policy": occupancy_policy,
                "n_atoms": model.n_atoms,
                "formula": model.formula(),
            },
        )
        return {"ok": True, "operation": "convert", "input_path": str(source), "proposal": proposal}

    def transform(
        self, input_path: str, output_format: str = "cif", **kwargs: Any
    ) -> dict[str, Any]:
        source = resolve_input_path(input_path, must_exist=True, directory=False)
        fmt = _fmt(output_format)
        source_fmt = detect_format(source)
        model = parse_structure(source)
        normalized = {k: _decode_value(v) for k, v in kwargs.items()}
        if source_fmt == "cif" and fmt != "cif" and not normalized.get("p1", False):
            operations = list(model.metadata.get("symmetry_operations") or [])
            if len(operations) > 1 and not model.metadata.get("symmetry_closed"):
                raise ValueError(
                    "CIF 含非平凡空间群操作，POSCAR/XYZ/extXYZ 无法保存对称性；"
                    "请明确设置 p1=True 后再转换"
                )
        if (
            source_fmt == "cif"
            and normalized.get("atom_translation") is not None
            and not normalized.get("p1", False)
        ):
            operations = list(model.metadata.get("symmetry_operations") or [])
            if len(operations) > 1 and not model.metadata.get("symmetry_closed"):
                raise ValueError(
                    "对非 P1 asymmetric unit 进行选定原子平移会破坏对称性；"
                    "请先设置 p1=True，再平移完整 P1 原子表"
                )
        transformed = transform_structure(
            model,
            **{
                k: v
                for k, v in normalized.items()
                if k
                in {
                    "p1",
                    "supercell",
                    "basis",
                    "origin_shift",
                    "atom_selection",
                    "atom_translation",
                    "translation_coordinate_type",
                    "connected_fragment",
                    "wrap",
                    "unwrap",
                    "extract",
                }
            },
        )
        text = structure_text(
            transformed,
            fmt,
            expand_p1=bool(normalized.get("p1", False)),
            occupancy_policy=str(normalized.get("occupancy_policy", "reject")),
        )
        suffix = extension_for_format("extxyz" if fmt == "xyz" else fmt)
        target = normalized.get("output_path") or str(
            (
                Path(normalized["output_dir"]).expanduser().resolve()
                if normalized.get("output_dir")
                else source.parent
            )
            / f"{safe_stem(source)}.transformed{suffix}"
        )
        proposal = create_proposal(
            operation="transform",
            sources=[source],
            artifacts=[
                {
                    "name": f"candidate-1{suffix}",
                    "data": text,
                    "label": "transformed",
                    "metadata": {
                        "format": fmt,
                        "n_atoms": transformed.n_atoms,
                        "formula": transformed.formula(),
                    },
                }
            ],
            output_dir=normalized.get("output_dir") or str(source.parent),
            requested_output=target,
            metadata={
                "output_format": fmt,
                "transform": {
                    k: v
                    for k, v in normalized.items()
                    if k not in {"output_path", "output_dir", "occupancy_policy"}
                },
                "n_atoms": transformed.n_atoms,
                "formula": transformed.formula(),
            },
        )
        return {
            "ok": True,
            "operation": "transform",
            "input_path": str(source),
            "proposal": proposal,
            "preview": transformed.as_dict(include_coords=False),
        }

    def bond_graph(
        self,
        input_path: str,
        *,
        expand_symmetry: bool = False,
        tolerance: float = 0.45,
        include_hydrogen: bool = True,
    ) -> dict[str, Any]:
        source = resolve_input_path(input_path, must_exist=True, directory=False)
        model = parse_structure(source)
        if expand_symmetry and detect_format(source) == "cif":
            model = globals()["expand_symmetry"](model)
        graph = build_bond_graph(
            model, tolerance=float(tolerance), include_hydrogen=include_hydrogen
        )
        return {
            "ok": True,
            "input_path": str(source),
            "format": model.format,
            "structure": model.as_dict(include_coords=False),
            "graph": graph_to_dict(graph, model),
        }

    def bond_propose(
        self,
        input_path: str,
        *,
        output_path: str | None = None,
        output_dir: str | None = None,
        expand_symmetry: bool = True,
        tolerance: float = 0.45,
        metal_tolerance: float = 0.65,
        include_hydrogen: bool = True,
        include_coordination: bool = True,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Create a CIF proposal with explicitly marked inferred bonds.

        ``bond_graph`` remains the read-only inspection route.  This action
        is the opt-in writer for the user's "补化学键" request: it emits a
        standard ``_geom_bond`` loop, but keeps ``_geom_bond_type`` unknown and
        stores the heuristic kind/order/confidence/image in DSH-prefixed
        columns.  That preserves a useful viewer annotation without claiming
        an experimental bond order.
        """

        source = resolve_input_path(input_path, must_exist=True, directory=False)
        if detect_format(source) != "cif":
            raise ValueError("bond_propose 目前只对 CIF 生成带 _geom_bond 的 proposal")
        model = parse_structure(source)
        if expand_symmetry:
            model = globals()["expand_symmetry"](model)
            operations = list(model.metadata.get("symmetry_operations") or [])
            if (
                not operations
                and model.metadata.get("space_group_number") not in (None, 1)
                and not model.metadata.get("symmetry_closed")
            ):
                raise ValueError(
                    "无法取得非 P1 CIF 的对称操作；拒绝生成缺少周期位点的键注释。"
                    "请提供显式 symmetry loop 或先转换为完整 P1 文件"
                )
        else:
            operations = list(model.metadata.get("symmetry_operations") or [])
            if len(operations) > 1 and not model.metadata.get("symmetry_closed"):
                raise ValueError(
                    "非 P1 CIF 的 atom loop 仍是 asymmetric unit；bond_propose 必须先展开对称性，"
                    "否则 _geom_bond 不能覆盖完整周期结构"
                )
        confidence_floor = float(min_confidence)
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("min_confidence 必须在 0 和 1 之间")
        graph = build_bond_graph(
            model,
            tolerance=float(tolerance),
            metal_tolerance=float(metal_tolerance),
            include_hydrogen=bool(include_hydrogen),
        )
        edges: list[dict[str, Any]] = []
        for i, j, data in graph.edges(data=True):
            if not include_coordination and data.get("kind") == "coordination":
                continue
            if float(data.get("confidence", 0.0)) < confidence_floor:
                continue
            edges.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "label_i": model.atoms[int(i)].label,
                    "label_j": model.atoms[int(j)].label,
                    **{
                        key: (value.tolist() if hasattr(value, "tolist") else value)
                        for key, value in data.items()
                    },
                }
            )
        edges.sort(key=lambda row: (int(row["i"]), int(row["j"])))
        geometry = validate_geometry(model, contact_limit=30)
        validation_summary = {
            "minimum_distance_A": geometry.get("minimum_distance"),
            "duplicate_contact_count": geometry.get("duplicate_contact_count", 0),
            "suspicious_contact_count": geometry.get("suspicious_contact_count", 0),
            "coordination": geometry.get("coordination", []),
        }
        # The generated atom list is complete after the optional expansion;
        # force identity-only P1 so a later reader cannot multiply it again.
        text = cif_text(model, expand_p1=True, bond_edges=edges)
        target = output_path or str(
            (Path(output_dir).expanduser().resolve() if output_dir else source.parent)
            / f"{safe_stem(source)}.with-bonds.cif"
        )
        graph_dict = graph_to_dict(graph, model)
        proposal = create_proposal(
            operation="bond_propose",
            sources=[source],
            artifacts=[
                {
                    "name": "candidate-1.cif",
                    "data": text,
                    "label": "CIF with inferred periodic bond annotations",
                    "metadata": {
                        "bond_count": len(edges),
                        "include_coordination": bool(include_coordination),
                        "min_confidence": confidence_floor,
                    },
                }
            ],
            output_dir=output_dir or str(source.parent),
            requested_output=target,
            metadata={
                "output_format": "cif",
                "expand_symmetry": bool(expand_symmetry),
                "bond_count": len(edges),
                "include_hydrogen": bool(include_hydrogen),
                "include_coordination": bool(include_coordination),
                "tolerance": float(tolerance),
                "metal_tolerance": float(metal_tolerance),
                "min_confidence": confidence_floor,
                "annotation_scope": (
                    "inferred periodic covalent/coordination candidates; "
                    "not experimental bond orders"
                ),
                "validation": validation_summary,
                "graph": graph_dict,
            },
        )
        warnings = [
            "_geom_bond_type 保持 '?'；所有边均为半径/周期距离启发式推断，"
            "请在使用前审查键级、价态和金属配位。"
        ]
        if validation_summary["duplicate_contact_count"]:
            warnings.append("输入/展开模型含小于 0.50 Å 的周期重复接触；proposal 未替你删除原子。")
        if validation_summary["suspicious_contact_count"]:
            warnings.append("输入/展开模型含疑似异常短接触；请结合原始 CIF 和化学角色审查。")
        return {
            "ok": True,
            "operation": "bond_propose",
            "input_path": str(source),
            "bond_count": len(edges),
            "edges": edges,
            "validation": validation_summary,
            "warnings": warnings,
            "proposal": proposal,
        }

    def hydrogen_propose(
        self,
        input_path: str,
        *,
        output_path: str | None = None,
        output_dir: str | None = None,
        top_k: int = 8,
        max_candidates: int = 128,
        orientation_count: int = 12,
        hetero_policy: str = "enumerate",
        include_unprotonated_hetero: bool = True,
        expand_symmetry: bool = True,
        ml: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = resolve_input_path(input_path, must_exist=True, directory=False)
        model = parse_structure(source)
        if expand_symmetry and detect_format(source) == "cif":
            model = globals()["expand_symmetry"](model)
        # Without an ML calculator, geometry top-k is enough and avoids
        # materializing unnecessary candidates.  When ML is explicitly
        # enabled, keep the whole deterministic beam (up to max_candidates)
        # until after H-only refinement so a geometry-ranked tail cannot hide a
        # lower-energy orientation.  The final response is still trimmed to
        # the user's requested top_k below.
        ml_requested = bool(ml and ml.get("enabled"))
        generation_top_k = max(int(top_k), int(max_candidates)) if ml_requested else int(top_k)
        proposal_result = propose_hydrogens(
            model,
            top_k=generation_top_k,
            max_candidates=max_candidates,
            orientation_count=orientation_count,
            hetero_policy=hetero_policy,
            include_unprotonated_hetero=include_unprotonated_hetero,
        )
        candidates = proposal_result.get("candidates", [])
        if not candidates:
            return {
                "ok": True,
                "operation": "hydrogen_propose",
                "input_path": str(source),
                **{k: v for k, v in proposal_result.items() if k != "candidates"},
                "candidates": [],
            }
        artifacts: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        ml_success = False
        evaluated: list[dict[str, Any]] = []
        for original_index, item in enumerate(candidates):
            original_model = item["model"]
            candidate_model = original_model
            base_score = float(item["score"])
            ml_info = {"status": "disabled", "reason": "ML 默认关闭"}
            if ml_requested:
                candidate_model, ml_info = refine_hydrogen_only(candidate_model, ml)
                if ml_info.get("status") == "ok":
                    refined_score, refined_detail = score_candidate(
                        candidate_model, item.get("placements", [])
                    )
                    if refined_detail.get("hard_clashes") or refined_detail.get(
                        "host_bond_violations"
                    ):
                        # A calculator is advisory here.  Never write a
                        # refined structure that introduces a sub-0.72 Å
                        # non-host contact or detaches a generated H from its
                        # intended host; keep the geometry candidate and make
                        # the rejection explicit in the audit.
                        ml_info = {
                            **ml_info,
                            "status": "rejected_geometry",
                            "reason": (
                                "ML refinement introduced hard H contact(s) or violated "
                                "the intended H-host distance; "
                                "original candidate retained"
                            ),
                            "hard_clash_count": len(refined_detail["hard_clashes"]),
                            "hard_clashes": refined_detail["hard_clashes"],
                            "host_bond_violation_count": len(
                                refined_detail.get("host_bond_violations", [])
                            ),
                            "host_bond_violations": refined_detail.get(
                                "host_bond_violations", []
                            ),
                        }
                        candidate_model = original_model
                    else:
                        ml_success = True
                        item["score"], item["detail"] = refined_score, refined_detail
            evaluated.append(
                {
                    "item": item,
                    "model": candidate_model,
                    "ml": ml_info,
                    "base_score": base_score,
                    "original_index": original_index,
                }
            )
        ranked_records, ml_ranking = _rank_hydrogen_candidates(
            evaluated, ml_requested=ml_requested
        )
        # Candidate enumeration can be wider than the requested response when
        # ML ranking is active.  Keep the audit's comparison scope complete but
        # write/return only the requested number of candidates.
        ranked_records = ranked_records[: max(1, int(top_k))]
        for index, record in enumerate(ranked_records, start=1):
            item = record["item"]
            candidate_model = record["model"]
            ml_info = record["ml"]
            name = f"candidate-{index:02d}.cif"
            artifacts.append(
                {
                    "name": name,
                    "data": cif_text(candidate_model, expand_p1=True),
                    "label": f"H candidate {index}",
                    "metadata": {
                        "score": item["score"],
                        "base_geometry_score": record["base_score"],
                        "ranking_basis": record["ranking_basis"],
                        "ml_rank_within_composition": record[
                            "ml_rank_within_composition"
                        ],
                        "composition_signature": record["composition_signature"],
                        "added_hydrogens": item["added_hydrogens"],
                        "ml": ml_info,
                    },
                }
            )
            summaries.append(
                {
                    "index": index - 1,
                    "score": float(item["score"]),
                    "base_geometry_score": float(record["base_score"]),
                    "ranking_basis": record["ranking_basis"],
                    "ml_rank_within_composition": record["ml_rank_within_composition"],
                    "composition_signature": record["composition_signature"],
                    "added_hydrogens": item["added_hydrogens"],
                    "unprotonated_sites": item.get("unprotonated_sites", []),
                    "ml": ml_info,
                    "detail": item["detail"],
                }
            )
        proposal = create_proposal(
            operation="hydrogen_propose",
            sources=[source],
            artifacts=artifacts,
            output_dir=output_dir or str(source.parent),
            requested_output=output_path or str(source.parent / f"{safe_stem(source)}.with-H.cif"),
            metadata={
                "output_format": "cif",
                "state": proposal_result.get("state"),
                "ml": ml or {"enabled": False},
                "ml_ranking": ml_ranking,
                "decisions": proposal_result.get("decisions", []),
                "generated_candidate_count": len(candidates),
                "candidate_count": len(artifacts),
            },
        )
        state = "model_preferred" if ml_success else proposal_result.get("state")
        return {
            "ok": True,
            "operation": "hydrogen_propose",
            "input_path": str(source),
            "state": state,
            "decisions": proposal_result.get("decisions", []),
            "generated_candidate_count": len(candidates),
            "candidate_count": len(summaries),
            "candidates": summaries,
            "proposal": proposal,
            "ml_ranking": ml_ranking,
        }

    def proposal_apply(
        self,
        manifest_path: str,
        *,
        candidate_index: int = 0,
        destination: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return apply_proposal(
            manifest_path,
            candidate_index=candidate_index,
            destination=destination,
            overwrite=overwrite,
        )

    def ts_search(
        self,
        reactant_path: str,
        product_path: str,
        *,
        output_format: str = "cif",
        output_path: str | None = None,
        output_dir: str | None = None,
        n_frames: int = 17,
        top_k: int = 3,
        mapping: Any = None,
        cell_mode: str = "require_equal",
        cell_tolerance: float = 1e-5,
        bond_tolerance: float = 0.45,
        min_bond_change: float = 0.30,
        include_hydrogen: bool = True,
        include_trajectory: bool = True,
        ml: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate file-level TS-like candidates from reactant/product files.

        This action intentionally creates interpolation candidates only.  It
        does not run a saddle-point optimizer, NEB, VASP, DFT, or job workflow.
        """

        reactant = resolve_input_path(reactant_path, must_exist=True, directory=False)
        product = resolve_input_path(product_path, must_exist=True, directory=False)
        r_model = parse_structure(reactant)
        p_model = parse_structure(product)
        search = search_transition_state_candidates(
            r_model,
            p_model,
            mapping=mapping,
            n_frames=int(n_frames),
            top_k=int(top_k),
            cell_mode=cell_mode,
            cell_tolerance=float(cell_tolerance),
            bond_tolerance=float(bond_tolerance),
            min_bond_change=float(min_bond_change),
            include_hydrogen=bool(include_hydrogen),
            ml=ml,
        )
        frames = search["frames"]
        selected = search["selected"]
        fmt = _fmt(output_format) or "cif"
        if fmt not in {"cif", "poscar", "vasp", "extxyz", "xyz"}:
            raise ValueError("ts_search output_format 仅支持 cif/poscar/extxyz/xyz")
        if fmt in {"cif", "poscar", "vasp"} and any(frame.cell is None for frame in frames):
            raise ValueError("无晶胞的 TS 路径只能输出 extxyz/xyz")
        suffix = extension_for_format("extxyz" if fmt == "xyz" else fmt)
        artifacts: list[dict[str, Any]] = []
        candidate_summaries: list[dict[str, Any]] = []
        for rank, row in enumerate(selected, start=1):
            frame_index = int(row["frame_index"])
            frame = frames[frame_index]
            text = structure_text(frame, fmt, expand_p1=True, occupancy_policy="reject")
            artifact_name = f"candidate-{rank:02d}{suffix}"
            artifacts.append(
                {
                    "name": artifact_name,
                    "data": text,
                    "label": f"TS-like candidate frame {frame_index}",
                    "metadata": {
                        "frame_index": frame_index,
                        "rank": rank,
                        "ts_score": row["ts_score"],
                        "geometry_score": row["geometry_score"],
                        "ml": row.get("ml"),
                    },
                }
            )
            candidate_summaries.append(
                {
                    "rank": rank,
                    "frame_index": frame_index,
                    "fraction": row["fraction"],
                    "ts_score": row["ts_score"],
                    "geometry_score": row["geometry_score"],
                    "clash_penalty": row["clash_penalty"],
                    "max_displacement_A": row["max_displacement_A"],
                    "ml": row.get("ml"),
                }
            )
        if include_trajectory:
            import io

            import ase.io

            from .writers import _ase_atoms

            ase_frames = []
            score_by_frame = {int(row["frame_index"]): row for row in search["metrics"]}
            score_by_frame.update({int(row["frame_index"]): row for row in search["ranked"]})
            for frame_index, frame in enumerate(frames):
                atoms = _ase_atoms(frame)
                row = score_by_frame[frame_index]
                atoms.info.update(
                    {
                        "ts_frame": frame_index,
                        "ts_fraction": float(row["fraction"]),
                        "ts_score": float(row.get("ts_score", 0.0)),
                        "geometry_score": float(row["geometry_score"]),
                    }
                )
                if row.get("ml", {}).get("energy_eV") is not None:
                    atoms.info["ml_energy_eV"] = float(row["ml"]["energy_eV"])
                ase_frames.append(atoms)
            stream = io.StringIO()
            ase.io.write(stream, ase_frames, format="extxyz")
            artifacts.append(
                {
                    "name": "ts-path.extxyz",
                    "data": stream.getvalue(),
                    "label": "all interpolation frames",
                    "metadata": {"frames": len(frames), "format": "extxyz"},
                }
            )
        report = {
            "schema_version": 1,
            "operation": "ts_search",
            "scope": "file-level interpolation and candidate ranking; no saddle optimization",
            "reactant_path": str(reactant),
            "product_path": str(product),
            "reactant_sha256": sha256_file(reactant),
            "product_sha256": sha256_file(product),
            "state": search["state"],
            "state_reason": search["state_reason"],
            "mapping": search["mapping"],
            "interpolation": search["interpolation"],
            "bond_changes": search["bond_changes"],
            "metrics": search["metrics"],
            "selected": candidate_summaries,
            "ml": search["ml"],
            "ranking": search["ranking"],
            "warnings": search["warnings"],
        }
        artifacts.append(
            {
                "name": "ts-search-report.json",
                "data": dumps(report, indent=True),
                "label": "TS candidate audit report",
                "metadata": {
                    "state": search["state"],
                    "bond_change_count": len(search["bond_changes"]),
                },
            }
        )
        target = output_path or str(
            (Path(output_dir).expanduser().resolve() if output_dir else reactant.parent)
            / f"{safe_stem(reactant)}.ts-candidate{suffix}"
        )
        proposal = create_proposal(
            operation="ts_search",
            sources=[reactant, product],
            artifacts=artifacts,
            output_dir=output_dir or str(reactant.parent),
            requested_output=target,
            metadata={
                "output_format": fmt,
                "n_frames": int(n_frames),
                "top_k": int(top_k),
                "state": search["state"],
                "scope": "file-level candidate search only",
                "ml": ml or {"enabled": False},
            },
        )
        return {
            "ok": True,
            "operation": "ts_search",
            "reactant_path": str(reactant),
            "product_path": str(product),
            "state": search["state"],
            "state_reason": search["state_reason"],
            "mapping": search["mapping"],
            "interpolation": search["interpolation"],
            "bond_changes": search["bond_changes"],
            "candidate_count": len(candidate_summaries),
            "candidates": candidate_summaries,
            "ml": search["ml"],
            "ranking": search["ranking"],
            "warnings": search["warnings"],
            "proposal": proposal,
        }

    def outcar_extract(
        self,
        input_path: str,
        *,
        mode: str = "summary",
        output_format: str | None = None,
        output_path: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        source = resolve_input_path(input_path, must_exist=True, directory=False)
        result = parse_outcar(source)
        summary = _outcar_summary(result)
        mode = _fmt(mode) or "summary"
        if mode in {"summary", "inspect"} or not output_format:
            return {
                "ok": bool(result.steps),
                "operation": "outcar_extract",
                "mode": mode,
                "summary": summary,
            }
        if not result.steps:
            raise ValueError("OUTCAR 没有完整离子步，不能生成候选输出")
        fmt = _fmt(output_format)
        if mode in {"last", "last_complete", "final"}:
            model = result.last_structure
            text = structure_text(model, fmt, expand_p1=True, occupancy_policy="reject")
            suffix = extension_for_format("extxyz" if fmt == "xyz" else fmt)
            target = output_path or str(
                (Path(output_dir).expanduser().resolve() if output_dir else source.parent)
                / f"{safe_stem(source)}.last{suffix}"
            )
            proposal = create_proposal(
                operation="outcar_extract",
                sources=[source],
                artifacts=[
                    {
                        "name": f"candidate-1{suffix}",
                        "data": text,
                        "label": "last complete ionic step",
                        "metadata": {"step": result.steps[-1].index, "format": fmt},
                    }
                ],
                output_dir=output_dir or str(source.parent),
                requested_output=target,
                metadata={
                    "output_format": fmt,
                    "mode": mode,
                    "complete_steps": len(result.steps),
                    "truncated": result.truncated_position_block,
                },
            )
            return {
                "ok": True,
                "operation": "outcar_extract",
                "mode": mode,
                "summary": summary,
                "proposal": proposal,
            }
        if mode in {"trajectory", "all"}:
            if fmt not in {"extxyz", "xyz"}:
                raise ValueError(
                    "OUTCAR trajectory 输出目前要求 extxyz/xyz，以保留每一步的能量和力"
                )
            import io

            import ase.io

            from .writers import _ase_atoms

            frames = []
            for step in result.steps:
                # Reuse the result's element mapping and attach the step metadata.
                temp = OutcarResult(
                    path=str(source), nions=result.nions, elements=result.elements, steps=[step]
                ).last_structure
                frames.append(_ase_atoms(temp))
            stream = io.StringIO()
            ase.io.write(stream, frames, format="extxyz")
            suffix = extension_for_format("extxyz")
            target = output_path or str(
                (Path(output_dir).expanduser().resolve() if output_dir else source.parent)
                / f"{safe_stem(source)}.trajectory{suffix}"
            )
            proposal = create_proposal(
                operation="outcar_extract",
                sources=[source],
                artifacts=[
                    {
                        "name": "trajectory.extxyz",
                        "data": stream.getvalue(),
                        "label": "complete ionic trajectory",
                        "metadata": {"frames": len(frames)},
                    }
                ],
                output_dir=output_dir or str(source.parent),
                requested_output=target,
                metadata={
                    "output_format": "extxyz",
                    "mode": mode,
                    "complete_steps": len(result.steps),
                    "truncated": result.truncated_position_block,
                },
            )
            return {
                "ok": True,
                "operation": "outcar_extract",
                "mode": mode,
                "summary": summary,
                "proposal": proposal,
            }
        raise ValueError(f"未知 OUTCAR mode: {mode}")

    def compare(
        self, input_a: str, input_b: str, *, ignore_hydrogen: bool = False, primitive: bool = False
    ) -> dict[str, Any]:
        path_a = resolve_input_path(input_a, must_exist=True, directory=False)
        path_b = resolve_input_path(input_b, must_exist=True, directory=False)
        a = parse_structure(path_a)
        b = parse_structure(path_b)
        if ignore_hydrogen:
            a.atoms = [x for x in a.atoms if not x.is_hydrogen]
            b.atoms = [x for x in b.atoms if not x.is_hydrogen]
        result: dict[str, Any] = {
            "ok": True,
            "a": {
                "path": str(path_a),
                "format": a.format,
                "n_atoms": a.n_atoms,
                "formula": a.formula(),
                "composition": a.composition(),
            },
            "b": {
                "path": str(path_b),
                "format": b.format,
                "n_atoms": b.n_atoms,
                "formula": b.formula(),
                "composition": b.composition(),
            },
            "ignore_hydrogen": ignore_hydrogen,
            "primitive_requested": primitive,
        }
        result["composition_equal"] = a.composition() == b.composition()
        result["atom_count_equal"] = a.n_atoms == b.n_atoms
        if a.cell is not None and b.cell is not None:
            from pymatgen.core import Lattice

            la, lb = Lattice(a.cell), Lattice(b.cell)
            result["lattice_a"] = {
                "abc": list(map(float, la.abc)),
                "angles": list(map(float, la.angles)),
            }
            result["lattice_b"] = {
                "abc": list(map(float, lb.abc)),
                "angles": list(map(float, lb.angles)),
            }
            result["lattice_delta"] = {
                "abc_abs_A": [abs(float(x - y)) for x, y in zip(la.abc, lb.abc, strict=True)],
                "angles_abs_deg": [
                    abs(float(x - y)) for x, y in zip(la.angles, lb.angles, strict=True)
                ],
            }
            try:
                from pymatgen.analysis.structure_matcher import StructureMatcher

                sa, sb = a.to_pymatgen(), b.to_pymatgen()
                if primitive:
                    sa = sa.get_primitive_structure()
                    sb = sb.get_primitive_structure()
                matcher = StructureMatcher(
                    ltol=0.2, stol=0.5, angle_tol=5.0, scale=True, attempt_supercell=True
                )
                result["structure_match"] = bool(matcher.fit(sa, sb))
                mapping = matcher.get_mapping(sa, sb)
                result["mapping"] = None if mapping is None else [int(x) for x in mapping]
            except Exception as exc:
                result["structure_match_error"] = str(exc)
        else:
            result["structure_match"] = False
            result["structure_match_reason"] = "至少一个输入没有周期晶胞"
        return result
