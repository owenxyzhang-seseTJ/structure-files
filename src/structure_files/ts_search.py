"""File-level transition-state candidate search.

This module deliberately stops before a saddle-point calculation.  It aligns a
reactant/product pair, generates periodic-safe interpolation frames, detects
endpoint bond changes, and ranks interior frames by reaction-coordinate
centrality.  An explicitly configured ASE calculator (for example a MACE
model) may add same-model energy/force descriptors, but no optimizer, NEB, DFT,
VASP, or other theory workflow is started here.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np

from .geometry import (
    build_bond_graph,
    covalent_radius,
    minimum_image,
    shortest_contacts,
)
from .models import AtomRecord, StructureData

CELL_MODES = {"require_equal", "reactant", "product", "linear"}


def _element_key(element: str) -> str:
    """Return the first element for a simple ordered/disordered species."""

    return str(element).split("/")[0].strip().capitalize()


def _labels_unique(model: StructureData) -> bool:
    labels = model.labels()
    return len(labels) == len(set(labels))


def _parse_mapping_text(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"mapping 不是有效 JSON: {exc}") from exc
    pairs: list[tuple[str, str]] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError("mapping 文本应为 i:j,i:j 或 label:label")
        left, right = (part.strip() for part in token.split(":", 1))
        pairs.append((left, right))
    return {left: right for left, right in pairs}


def _resolve_atom_index(value: Any, labels: list[str], n_atoms: int, side: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{side} mapping 索引不能是布尔值")
    if isinstance(value, (int, np.integer)):
        index = int(value)
    else:
        text = str(value).strip()
        try:
            index = int(text)
        except ValueError:
            if text not in labels:
                raise ValueError(f"{side} mapping 找不到标签: {text}") from None
            index = labels.index(text)
    if index < 0 or index >= n_atoms:
        raise IndexError(f"{side} mapping 索引越界: {index}")
    return index


def resolve_atom_mapping(
    reactant: StructureData,
    product: StructureData,
    mapping: Any = None,
) -> tuple[list[int], dict[str, Any]]:
    """Map each reactant index to one product index with element checks.

    Labels are preferred when both files carry the same unique labels.  If
    labels differ, the deterministic fallback is occurrence order within each
    element.  A user mapping can be a list, a dictionary, JSON text, or a
    compact ``left:right,left:right`` string.
    """

    if reactant.n_atoms != product.n_atoms:
        raise ValueError(
            f"TS search 要求反应物/产物原子数相同: {reactant.n_atoms} != {product.n_atoms}"
        )
    n_atoms = reactant.n_atoms
    reactant_labels = reactant.labels()
    product_labels = product.labels()
    parsed = _parse_mapping_text(mapping)
    resolved: list[int] | None = None
    mapping_mode = "element_occurrence"
    if parsed is not None:
        mapping_mode = "user"
        resolved = [-1] * n_atoms
        if isinstance(parsed, (list, tuple)):
            if len(parsed) != n_atoms:
                raise ValueError("mapping 列表长度必须等于反应物原子数")
            resolved = [
                _resolve_atom_index(value, product_labels, n_atoms, "product") for value in parsed
            ]
        elif isinstance(parsed, dict):
            for left, right in parsed.items():
                i = _resolve_atom_index(left, reactant_labels, n_atoms, "reactant")
                j = _resolve_atom_index(right, product_labels, n_atoms, "product")
                if resolved[i] != -1:
                    raise ValueError(f"mapping 重复指定反应物原子: {left}")
                resolved[i] = j
            if any(index < 0 for index in resolved):
                raise ValueError("mapping 必须覆盖每个反应物原子")
        else:
            raise ValueError("mapping 需要列表或字典")
    elif (
        _labels_unique(reactant)
        and _labels_unique(product)
        and set(reactant_labels) == set(product_labels)
    ):
        mapping_mode = "label"
        resolved = [product_labels.index(label) for label in reactant_labels]
    else:
        buckets: dict[str, list[int]] = {}
        for j, atom in enumerate(product.atoms):
            buckets.setdefault(_element_key(atom.element), []).append(j)
        used: dict[str, int] = {}
        resolved = []
        for atom in reactant.atoms:
            key = _element_key(atom.element)
            offset = used.get(key, 0)
            if offset >= len(buckets.get(key, [])):
                raise ValueError(f"反应物/产物元素组成不一致，缺少 {key}")
            resolved.append(buckets[key][offset])
            used[key] = offset + 1
    assert resolved is not None
    if len(set(resolved)) != n_atoms:
        raise ValueError("mapping 必须是一一映射，不能重复使用产物原子")
    mismatches = []
    for i, j in enumerate(resolved):
        left = _element_key(reactant.atoms[i].element)
        right = _element_key(product.atoms[j].element)
        if left != right:
            mismatches.append(
                {
                    "reactant_index": i,
                    "product_index": j,
                    "reactant_element": left,
                    "product_element": right,
                }
            )
    if mismatches:
        raise ValueError(f"mapping 存在元素不一致: {mismatches[:5]}")
    return resolved, {
        "mode": mapping_mode,
        "reactant_to_product": resolved,
        "pairs": [
            {
                "reactant_index": i,
                "product_index": j,
                "reactant_label": reactant.atoms[i].label,
                "product_label": product.atoms[j].label,
                "element": _element_key(reactant.atoms[i].element),
            }
            for i, j in enumerate(resolved)
        ],
    }


def _cell_delta(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale)


def _reorder_product(
    product: StructureData, mapping: list[int], labels: list[str]
) -> StructureData:
    out = product.copy()
    out.atoms = [product.atoms[j].copy() for j in mapping]
    for atom, label in zip(out.atoms, labels, strict=True):
        atom.label = label
    return out


def _alignment(
    reactant: StructureData,
    product: StructureData,
    mapping: list[int],
    *,
    cell_mode: str,
    cell_tolerance: float,
) -> dict[str, Any]:
    mode = str(cell_mode or "require_equal").strip().lower()
    if mode not in CELL_MODES:
        raise ValueError(f"cell_mode 必须是 {sorted(CELL_MODES)} 之一")
    if (reactant.cell is None) != (product.cell is None):
        raise ValueError("反应物和产物必须同时有晶胞，或同时没有晶胞")
    if reactant.cell is None:
        if not reactant.coords_are_cartesian or not product.coords_are_cartesian:
            raise ValueError("无晶胞输入只能使用笛卡尔坐标；分数坐标缺少长度基准")
        r_cart = reactant.coords_cartesian.copy()
        p_cart = product.coords_cartesian[mapping].copy()
        return {
            "mapping": mapping,
            "r_cart": r_cart,
            "p_cart": p_cart,
            "r_cell": None,
            "p_cell": None,
            "common_cell": None,
            "cell_mode": "none",
            "warnings": ["输入没有周期晶胞；按笛卡尔坐标线性插值，不能进行周期最小镜像"],
        }

    r_cell = np.asarray(reactant.cell, dtype=float)
    p_cell = np.asarray(product.cell, dtype=float)
    relative_delta = _cell_delta(r_cell, p_cell)
    warnings: list[str] = []
    if mode == "require_equal":
        if relative_delta is None or relative_delta > float(cell_tolerance):
            raise ValueError(
                "cell_mode='require_equal' 要求反应物/产物晶胞一致；"
                f"相对差异 {relative_delta!r} 超过 {cell_tolerance}"
            )
        common_cell = r_cell
    elif mode == "reactant":
        common_cell = r_cell
        if relative_delta and relative_delta > float(cell_tolerance):
            warnings.append("产物坐标被映射到反应物晶胞；请确认晶胞变化不是反应坐标的一部分")
    elif mode == "product":
        common_cell = p_cell
        if relative_delta and relative_delta > float(cell_tolerance):
            warnings.append("反应物坐标被映射到产物晶胞；请确认晶胞变化不是反应坐标的一部分")
    else:
        common_cell = r_cell
        if relative_delta and relative_delta > float(cell_tolerance):
            warnings.append("linear 晶胞插值只用于候选帧生成，不代表晶胞鞍点路径")

    # ``linear`` is the only mode where the lattice itself is part of the
    # interpolation.  In that case each endpoint must retain its own
    # fractional coordinates; converting both to one Cartesian frame first
    # would mix a cell change into the atomic reaction coordinate.
    if mode == "linear":
        ref_frac_r = reactant.frac_coords.copy()
        ref_frac_p = product.frac_coords[mapping].copy()
    else:
        r_cart_native = reactant.coords_cartesian.copy()
        p_cart_native = product.coords_cartesian[mapping].copy()
        ref_frac_r = r_cart_native @ np.linalg.inv(common_cell)
        ref_frac_p = p_cart_native @ np.linalg.inv(common_cell)

    # Unwrap product atoms against the reactant endpoint so an atom crossing a
    # periodic boundary travels the short path rather than through the cell.
    p_frac_unwrapped = ref_frac_p.copy()
    from pymatgen.core import Lattice

    lattice = Lattice(r_cell if mode == "linear" else common_cell)
    for index in range(reactant.n_atoms):
        _, image = lattice.get_distance_and_image(ref_frac_r[index], ref_frac_p[index])
        p_frac_unwrapped[index] = ref_frac_p[index] + np.asarray(image, dtype=float)
    if mode == "linear":
        r_cart = ref_frac_r @ r_cell
        p_cart = p_frac_unwrapped @ p_cell
    else:
        r_cart = ref_frac_r @ common_cell
        p_cart = p_frac_unwrapped @ common_cell
    return {
        "mapping": mapping,
        "r_cart": r_cart,
        "p_cart": p_cart,
        "r_cell": r_cell,
        "p_cell": p_cell,
        "common_cell": common_cell,
        "r_frac": ref_frac_r,
        "p_frac": p_frac_unwrapped,
        "cell_mode": mode,
        "relative_cell_delta": relative_delta,
        "warnings": warnings,
    }


def _frame_cell(alignment: dict[str, Any], fraction: float) -> np.ndarray | None:
    if alignment["common_cell"] is None:
        return None
    mode = alignment["cell_mode"]
    if mode == "linear":
        return (1.0 - fraction) * alignment["r_cell"] + fraction * alignment["p_cell"]
    return np.asarray(alignment["common_cell"], dtype=float).copy()


def interpolate_frames(
    reactant: StructureData,
    product: StructureData,
    mapping: list[int],
    *,
    n_frames: int = 17,
    cell_mode: str = "require_equal",
    cell_tolerance: float = 1e-5,
) -> tuple[list[StructureData], list[np.ndarray], dict[str, Any]]:
    """Return wrapped frame models, unwrapped Cartesian coordinates and audit."""

    count = int(n_frames)
    if count < 3:
        raise ValueError("n_frames 至少为 3，才能排除反应物/产物端点")
    alignment = _alignment(
        reactant,
        product,
        mapping,
        cell_mode=cell_mode,
        cell_tolerance=cell_tolerance,
    )
    frames: list[StructureData] = []
    unwrapped: list[np.ndarray] = []
    for frame_index in range(count):
        fraction = frame_index / (count - 1)
        cell = _frame_cell(alignment, fraction)
        if alignment["cell_mode"] == "linear":
            if cell is None or abs(float(np.linalg.det(cell))) < 1e-12:
                raise ValueError(f"第 {frame_index} 帧的线性插值晶胞退化")
            frac = (1.0 - fraction) * alignment["r_frac"] + fraction * alignment["p_frac"]
            cart = frac @ cell
        else:
            cart = (1.0 - fraction) * alignment["r_cart"] + fraction * alignment["p_cart"]
        atoms: list[AtomRecord] = []
        if cell is None:
            for source_atom, xyz in zip(reactant.atoms, cart, strict=True):
                props = deepcopy(source_atom.properties)
                props.update({"ts_frame": frame_index, "ts_fraction": fraction})
                atoms.append(
                    AtomRecord(
                        source_atom.label,
                        source_atom.element,
                        xyz,
                        source_atom.occupancy,
                        source_atom.charge,
                        props,
                    )
                )
            model = StructureData(
                atoms=atoms,
                cell=None,
                pbc=(False, False, False),
                coords_are_cartesian=True,
                format="ts-search",
                source_path=reactant.source_path,
                metadata={"ts_frame": frame_index, "ts_fraction": fraction},
            )
        else:
            frac = cart @ np.linalg.inv(cell)
            frac_wrapped = frac - np.floor(frac)
            for source_atom, xyz in zip(reactant.atoms, frac_wrapped, strict=True):
                props = deepcopy(source_atom.properties)
                props.update({"ts_frame": frame_index, "ts_fraction": fraction})
                atoms.append(
                    AtomRecord(
                        source_atom.label,
                        source_atom.element,
                        xyz,
                        source_atom.occupancy,
                        source_atom.charge,
                        props,
                    )
                )
            model = StructureData(
                atoms=atoms,
                cell=cell,
                pbc=(True, True, True),
                coords_are_cartesian=False,
                format="ts-search",
                source_path=reactant.source_path,
                metadata={"ts_frame": frame_index, "ts_fraction": fraction},
            )
        frames.append(model)
        unwrapped.append(cart)
    audit = {
        "n_frames": count,
        "cell_mode": alignment["cell_mode"],
        "relative_cell_delta": alignment.get("relative_cell_delta"),
        "warnings": alignment["warnings"],
        "periodic": alignment["common_cell"] is not None,
    }
    return frames, unwrapped, audit


def _endpoint_bond_changes(
    reactant: StructureData,
    product: StructureData,
    *,
    tolerance: float,
    min_bond_change: float,
    include_hydrogen: bool,
) -> list[dict[str, Any]]:
    r_graph = build_bond_graph(reactant, tolerance=tolerance, include_hydrogen=include_hydrogen)
    p_graph = build_bond_graph(product, tolerance=tolerance, include_hydrogen=include_hydrogen)
    pairs = {tuple(sorted((int(i), int(j)))) for i, j in r_graph.edges}
    pairs.update(tuple(sorted((int(i), int(j)))) for i, j in p_graph.edges)
    changes: list[dict[str, Any]] = []
    for i, j in sorted(pairs):
        d_r, image_r, _ = minimum_image(reactant, i, j)
        d_p, image_p, _ = minimum_image(product, i, j)
        edge_r = r_graph.get_edge_data(i, j)
        edge_p = p_graph.get_edge_data(i, j)
        if edge_r is None and edge_p is not None:
            kind = "formed"
        elif edge_r is not None and edge_p is None:
            kind = "broken"
        elif abs(d_p - d_r) >= float(min_bond_change):
            kind = "shortened" if d_p < d_r else "lengthened"
        else:
            continue
        edge = edge_p or edge_r or {}
        changes.append(
            {
                "i": i,
                "j": j,
                "label_i": reactant.atoms[i].label,
                "label_j": reactant.atoms[j].label,
                "element_i": _element_key(reactant.atoms[i].element),
                "element_j": _element_key(reactant.atoms[j].element),
                "change": kind,
                "distance_reactant_A": float(d_r),
                "distance_product_A": float(d_p),
                "delta_A": float(d_p - d_r),
                "image_reactant": image_r.tolist(),
                "image_product": image_p.tolist(),
                "endpoint_kind": edge.get("kind"),
                "endpoint_bond_order": edge.get("bond_order"),
                "endpoint_confidence": edge.get("confidence"),
            }
        )
    return changes


def _progress(change: dict[str, Any], distance: float) -> float:
    d_r = float(change["distance_reactant_A"])
    d_p = float(change["distance_product_A"])
    kind = change["change"]
    if kind in {"formed", "shortened"}:
        denominator = d_r - d_p
        value = (d_r - distance) / denominator if abs(denominator) > 1e-8 else 0.5
    else:
        denominator = d_p - d_r
        value = (distance - d_r) / denominator if abs(denominator) > 1e-8 else 0.5
    return float(np.clip(value, 0.0, 1.0))


def _clash_metrics(model: StructureData) -> tuple[float, list[dict[str, Any]]]:
    penalty = 0.0
    severe: list[dict[str, Any]] = []
    for contact in shortest_contacts(model, limit=model.n_atoms * max(model.n_atoms - 1, 1)):
        i, j = int(contact["i"]), int(contact["j"])
        if i >= j:
            continue
        radius_sum = covalent_radius(model.atoms[i].element) + covalent_radius(
            model.atoms[j].element
        )
        hard = max(0.45, 0.52 * radius_sum)
        soft = max(hard + 0.05, 0.72 * radius_sum)
        distance = float(contact["distance"])
        if distance < hard:
            penalty += 10.0 + 20.0 * (hard - distance)
            severe.append({**contact, "severity": "hard", "hard_limit_A": hard})
        elif distance < soft:
            penalty += (soft - distance) / max(soft - hard, 1e-8)
            if len(severe) < 20:
                severe.append({**contact, "severity": "soft", "soft_limit_A": soft})
    return float(penalty), severe[:20]


def frame_metrics(
    frames: list[StructureData],
    unwrapped_cart: list[np.ndarray],
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for index, (model, cart) in enumerate(zip(frames, unwrapped_cart, strict=True)):
        fraction = index / max(len(frames) - 1, 1)
        progress_rows: list[dict[str, Any]] = []
        progress_values: list[float] = []
        for change in changes:
            distance, image, _ = minimum_image(model, int(change["i"]), int(change["j"]))
            value = _progress(change, distance)
            progress_values.append(value)
            progress_rows.append(
                {
                    "i": change["i"],
                    "j": change["j"],
                    "labels": [change["label_i"], change["label_j"]],
                    "distance_A": float(distance),
                    "image": image.tolist(),
                    "progress": value,
                }
            )
        if progress_values:
            values = np.asarray(progress_values, dtype=float)
            centrality = float(1.0 - np.mean(np.abs(2.0 * values - 1.0)))
            synchrony = float(1.0 - min(1.0, np.std(values) * 2.0))
            geometry_score = max(0.0, centrality * (0.5 + 0.5 * synchrony))
        else:
            centrality = synchrony = geometry_score = 0.0
        clash_penalty, severe = _clash_metrics(model)
        displacement = np.linalg.norm(cart - unwrapped_cart[0], axis=1)
        metrics.append(
            {
                "frame_index": index,
                "fraction": float(fraction),
                "interior": 0 < index < len(frames) - 1,
                "geometry_score": float(geometry_score),
                "reaction_coordinate_centrality": centrality,
                "reaction_coordinate_synchrony": synchrony,
                "clash_penalty": clash_penalty,
                "severe_contacts": severe,
                "max_displacement_A": float(np.max(displacement)) if len(displacement) else 0.0,
                "rms_displacement_A": float(np.sqrt(np.mean(displacement**2)))
                if len(displacement)
                else 0.0,
                "bond_progress": progress_rows,
            }
        )
    return metrics


def rank_frames(
    metrics: list[dict[str, Any]],
    *,
    ml_rows: list[dict[str, Any]] | None = None,
    geometry_weight: float = 0.70,
    ml_weight: float = 0.30,
    clash_weight: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign comparable scores only within this one path/model."""

    rows = [dict(row) for row in metrics]
    ml_rows = ml_rows or []
    energies = [float(row["energy_eV"]) for row in ml_rows if row.get("energy_eV") is not None]
    use_ml = bool(energies) and len(ml_rows) == len(rows)
    min_energy = min(energies) if energies else None
    max_energy = max(energies) if energies else None
    energy_span = (
        (max_energy - min_energy) if min_energy is not None and max_energy is not None else 0.0
    )
    ml_by_frame = {int(row["frame_index"]): row for row in ml_rows}
    gw = max(0.0, float(geometry_weight))
    mw = max(0.0, float(ml_weight)) if use_ml else 0.0
    if gw + mw <= 0:
        gw = 1.0
    total = gw + mw
    for row in rows:
        frame_index = int(row["frame_index"])
        ml_row = ml_by_frame.get(frame_index)
        if ml_row:
            row["ml"] = ml_row
        if use_ml and ml_row and energy_span > 1e-12:
            row["ml_energy_peak_score"] = float(
                (float(ml_row["energy_eV"]) - min_energy) / energy_span
            )
        else:
            row["ml_energy_peak_score"] = None
        geometry_component = max(0.0, float(row["geometry_score"]))
        energy_component = float(row["ml_energy_peak_score"] or 0.0)
        row["ts_score"] = float(
            (gw * geometry_component + mw * energy_component) / total
            - float(clash_weight) * min(1.0, float(row["clash_penalty"]) / 10.0)
        )
    ranked = sorted(
        (row for row in rows if row.get("interior")),
        key=lambda row: (-float(row["ts_score"]), int(row["frame_index"])),
    )
    audit = {
        "ml_used_for_ranking": use_ml,
        "geometry_weight": gw / total,
        "ml_weight": mw / total,
        "clash_weight": float(clash_weight),
        "energy_min_eV": min_energy,
        "energy_max_eV": max_energy,
        "energy_span_eV": energy_span,
        "ranking_scope": "interior frames of this reactant/product pair only",
    }
    return ranked, audit


def search_transition_state_candidates(
    reactant: StructureData,
    product: StructureData,
    *,
    mapping: Any = None,
    n_frames: int = 17,
    top_k: int = 3,
    cell_mode: str = "require_equal",
    cell_tolerance: float = 1e-5,
    bond_tolerance: float = 0.45,
    min_bond_change: float = 0.30,
    include_hydrogen: bool = True,
    ml: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and rank TS-like interpolation frames without saddle optimization."""

    atom_mapping, mapping_audit = resolve_atom_mapping(reactant, product, mapping)
    product_aligned = _reorder_product(product, atom_mapping, reactant.labels())
    frames, unwrapped, interpolation_audit = interpolate_frames(
        reactant,
        product,
        atom_mapping,
        n_frames=n_frames,
        cell_mode=cell_mode,
        cell_tolerance=cell_tolerance,
    )
    changes = _endpoint_bond_changes(
        reactant,
        product_aligned,
        tolerance=float(bond_tolerance),
        min_bond_change=float(min_bond_change),
        include_hydrogen=bool(include_hydrogen),
    )
    metrics = frame_metrics(frames, unwrapped, changes)
    ml_rows: list[dict[str, Any]] = []
    ml_audit: dict[str, Any] = {
        "status": "disabled",
        "reason": "ML 默认关闭；候选搜索先使用几何路径",
    }
    if ml and ml.get("enabled"):
        from .ml import evaluate_models

        ml_rows, ml_audit = evaluate_models(frames, ml)
    ranked, ranking_audit = rank_frames(metrics, ml_rows=ml_rows)
    selected = ranked[: max(1, int(top_k))]
    if not changes:
        state = "no_endpoint_bond_change"
        state_reason = "端点启发式键图没有达到变化阈值；插值帧不应被解释为过渡态。"
    else:
        state = "candidate_frames"
        state_reason = "已发现端点键变化并生成内部候选帧；仍需独立鞍点/实验验证。"
    return {
        "frames": frames,
        "unwrapped_cart": unwrapped,
        "product_aligned": product_aligned,
        "mapping": mapping_audit,
        "interpolation": interpolation_audit,
        "bond_changes": changes,
        "metrics": metrics,
        "ranked": ranked,
        "selected": selected,
        "ml": ml_audit,
        "ranking": ranking_audit,
        "state": state,
        "state_reason": state_reason,
        "warnings": interpolation_audit.get("warnings", []),
    }
