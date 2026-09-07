"""Valence-guided hydrogen placement and candidate ranking.

The algorithm is intentionally an enumerator, not a structure solver.  It can
place routine C--H atoms from local valence and geometry, while exposing
alternative protonation/orientation candidates for O/N/S sites and defects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import (
    build_bond_graph,
    covalent_radius,
    minimum_image,
    target_valence,
    valence_summary,
)
from .models import AtomRecord, StructureData, is_metal
from .operations import wrap_structure

H_BOND_LENGTHS = {"C": 1.09, "N": 1.01, "O": 0.98, "S": 1.34, "P": 1.42, "Si": 1.48, "B": 1.20}
ACCEPTORS = {"N", "O", "S", "F", "Cl", "Br", "I"}
HETERO = {"N", "O", "S", "P"}


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-10:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return value / norm


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    trial = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, trial), np.array([0.0, 0.0, 1.0]))
    v = _unit(np.cross(axis, u), np.array([0.0, 1.0, 0.0]))
    return u, v


def _rotate_about(axis: np.ndarray, vector: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1 - math.cos(angle))
    )


def _neighbor_directions(model: StructureData, graph: Any, host: int) -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    for other in graph.neighbors(host):
        _, _, vector = minimum_image(model, host, other)
        if np.linalg.norm(vector) > 1e-8:
            directions.append(_unit(vector))
    return directions


def _preferred_axis(neighbor_dirs: list[np.ndarray]) -> np.ndarray:
    if not neighbor_dirs:
        return np.array([1.0, 0.0, 0.0])
    return _unit(-np.sum(np.asarray(neighbor_dirs), axis=0), fallback=np.array([1.0, 0.0, 0.0]))


def _orientation_sets(
    model: StructureData,
    graph: Any,
    host: int,
    n_hydrogen: int,
    *,
    orientation_count: int = 12,
    hetero: bool = False,
) -> list[list[np.ndarray]]:
    dirs = _neighbor_directions(model, graph, host)
    axis = _preferred_axis(dirs)
    u, v = _basis(axis)
    count = max(1, int(orientation_count))
    result: list[list[np.ndarray]] = []
    if n_hydrogen <= 0:
        return [[]]
    if n_hydrogen == 1:
        # Carbon valence completion is usually determinate.  Heteroatom
        # protonation is sampled around the preferred axis to expose hydrogen-
        # bond and disorder alternatives without claiming one is observed.
        cone = math.radians(18.0 if hetero else 3.0)
        for k in range(1 if not hetero else count):
            phi = 2 * math.pi * k / max(1, count)
            lateral = math.cos(phi) * u + math.sin(phi) * v
            direction = _unit(math.cos(cone) * axis + math.sin(cone) * lateral)
            result.append([direction])
        return result
    if n_hydrogen == 2:
        theta = math.radians(54.75)
        for k in range(count):
            phi = 2 * math.pi * k / count
            lateral = math.cos(phi) * u + math.sin(phi) * v
            d1 = _unit(math.cos(theta) * axis + math.sin(theta) * lateral)
            d2 = _unit(math.cos(theta) * axis - math.sin(theta) * lateral)
            result.append([d1, d2])
        return result
    if n_hydrogen == 3:
        # Three tetrahedral directions around the negative sum of existing
        # neighbors.  Rotate the set about its axis to represent free methyl/
        # ammonium orientation without random numbers.
        theta = math.acos(-1.0 / 3.0)
        base = []
        for k in range(3):
            phi = 2 * math.pi * k / 3
            base.append(
                _unit(
                    math.cos(theta) * axis
                    + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * v)
                )
            )
        for k in range(count):
            angle = 2 * math.pi * k / count
            result.append([_unit(_rotate_about(axis, d, angle)) for d in base])
        return result
    # Four H is only sensible for an isolated carbon/metal hydride and is
    # underdetermined in this file-only workflow; return one tetrahedral set.
    tetra = [
        np.array([1, 1, 1]),
        np.array([1, -1, -1]),
        np.array([-1, 1, -1]),
        np.array([-1, -1, 1]),
    ]
    return [[_unit(x) for x in tetra[:n_hydrogen]]]


def _current_valence(graph: Any, host: int) -> float:
    return sum(
        float(data.get("bond_order", 1.0))
        for _, _, data in graph.edges(host, data=True)
        if data.get("kind") == "covalent"
    )


def _host_groups(
    model: StructureData, graph: Any, *, hetero_policy: str = "enumerate"
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return mutually-exclusive placement groups and diagnostic decisions."""
    groups: list[list[dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []
    handled: set[int] = set()
    # Detect common carboxylate/sulfonate-like pairs: one proton belongs to one
    # of multiple singly bonded heteroatoms, not to every oxygen independently.
    for center, atom in enumerate(model.atoms):
        if atom.element not in {"C", "S", "P"}:
            continue
        oxy = [
            j
            for j in graph.neighbors(center)
            if model.atoms[j].element == "O" and _current_valence(graph, j) < 2.1
        ]
        if len(oxy) >= 2:
            options: list[dict[str, Any]] = []
            for host in oxy:
                if host in handled:
                    continue
                options.append(
                    {"host": host, "n_h": 1, "kind": "multi_oxygen_protonation", "hetero": True}
                )
                handled.add(host)
            if options:
                groups.append(options)
                decisions.append(
                    {
                        "type": "mutually_exclusive_protonation",
                        "center": model.atoms[center].label,
                        "hosts": [model.atoms[x["host"]].label for x in options],
                    }
                )
    for host, atom in enumerate(model.atoms):
        if host in handled or atom.is_hydrogen or is_metal(atom.element):
            continue
        target = target_valence(atom.element)
        if target is None:
            continue
        valence = _current_valence(graph, host)
        deficit = int(round(target - valence))
        if deficit <= 0 or deficit > 3:
            continue
        is_hetero = atom.element in HETERO
        if is_hetero and hetero_policy == "acidic_only" and atom.element == "N":
            continue
        if is_hetero:
            kind = "heteroatom_protonation"
        else:
            kind = "implicit_valence_hydrogen"
        groups.append([{"host": host, "n_h": deficit, "kind": kind, "hetero": is_hetero}])
        decisions.append(
            {
                "type": kind,
                "host": atom.label,
                "element": atom.element,
                "valence": valence,
                "target": target,
                "deficit": deficit,
            }
        )
    return groups, decisions


def _add_hydrogens(model: StructureData, placements: list[dict[str, Any]]) -> StructureData:
    out = model.copy()
    if out.cell is None:
        raise ValueError("补氢需要晶胞，以便输出周期一致的 CIF/POSCAR")
    for placement in placements:
        host = int(placement["host"])
        if host < 0:
            continue
        host_atom = out.atoms[host]
        host_cart = out.coords_cartesian[host]
        for h_index, direction in enumerate(placement["directions"], start=1):
            length = H_BOND_LENGTHS.get(
                host_atom.element, covalent_radius(host_atom.element) + covalent_radius("H")
            )
            cart = host_cart + float(length) * np.asarray(direction, dtype=float)
            frac = cart @ np.linalg.inv(np.asarray(out.cell, dtype=float))
            label = f"H_{host_atom.label}_{h_index}"
            used = {a.label for a in out.atoms}
            if label in used:
                suffix = 2
                while f"{label}_{suffix}" in used:
                    suffix += 1
                label = f"{label}_{suffix}"
            props = {
                "generated_hydrogen": True,
                "host_index": host,
                "host_label": host_atom.label,
                "placement_kind": placement.get("kind"),
                "orientation_index": placement.get("orientation_index"),
            }
            out.atoms.append(AtomRecord(label, "H", frac, 1.0, properties=props))
    out.metadata["generated_hydrogens"] = len(out.atoms) - model.n_atoms
    out.metadata["hydrogen_proposal"] = True
    # Existing source _geom_bond rows describe the pre-H atom set and may be
    # incomplete for an asymmetric unit.  Do not carry stale annotations into
    # a candidate; callers can run bond_propose on the chosen model again.
    out.metadata.pop("geom_bonds", None)
    out.metadata["geom_bond_present"] = False
    out.metadata["geom_bond_invalidated_reason"] = "hydrogen coordinates changed the bond set"
    return wrap_structure(out)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.linalg.norm(a), np.linalg.norm(b)
    if aa < 1e-10 or bb < 1e-10:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / aa / bb, -1.0, 1.0))))


def score_candidate(
    model: StructureData, placements: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    graph = build_bond_graph(model, include_hydrogen=True)
    score = 0.0
    reasons: list[dict[str, Any]] = []
    hard_clashes: list[dict[str, Any]] = []
    host_bond_violations: list[dict[str, Any]] = []
    for placement in placements:
        if (
            int(placement.get("host", -1)) < 0
            and placement.get("kind") == "unprotonated_alternative"
        ):
            # Keep a no-protonation alternative visible for disordered/charged
            # sites, but rank a chemically completed valence model ahead of it.
            score += 8.0
    # Every generated H must have exactly one intended host bond and no severe
    # unrelated contact.  Distances are reported explicitly for auditability.
    for atom_index, atom in enumerate(model.atoms):
        if not atom.properties.get("generated_hydrogen"):
            continue
        host = atom.properties.get("host_index")
        if host is None:
            score += 100.0
            continue
        distance, _, vector = minimum_image(model, int(host), atom_index)
        ideal = H_BOND_LENGTHS.get(model.atoms[int(host)].element, 1.1)
        # A fixed-heavy-atom ML optimizer is still free to detach a proton or
        # collapse it onto the host.  Treat that as a hard geometry failure so
        # a low model energy can never outrank a chemically connected H
        # candidate.  The bounds are intentionally broad enough for ordinary
        # thermal/crystallographic variation and are only a screening gate.
        host_lower = max(0.72, 0.70 * ideal)
        host_upper = max(1.35, 1.45 * ideal)
        if distance < host_lower or distance > host_upper:
            score += 1000.0 + 100.0 * min(
                abs(distance - host_lower), abs(distance - host_upper)
            )
            host_bond_violations.append(
                {
                    "h": atom.label,
                    "host": model.atoms[int(host)].label,
                    "distance_A": float(distance),
                    "ideal_A": float(ideal),
                    "allowed_range_A": [float(host_lower), float(host_upper)],
                }
            )
        score += min(20.0, 8.0 * abs(distance - ideal))
        nearest = []
        for j in range(model.n_atoms):
            if j == atom_index or j == int(host):
                continue
            d, _, v = minimum_image(model, atom_index, j)
            nearest.append((d, j, v))
            if d < 0.72:
                score += 1000.0 + 100.0 * (0.72 - d)
                hard_clashes.append(
                    {
                        "h": atom.label,
                        "other": model.atoms[j].label,
                        "distance_A": float(d),
                    }
                )
            elif d < 1.15 and not model.atoms[j].is_hydrogen:
                score += 30.0 * (1.15 - d)
        if nearest:
            dmin, jmin, _ = min(nearest, key=lambda x: x[0])
            reasons.append(
                {
                    "h": atom.label,
                    "nearest_nonhost": model.atoms[jmin].label,
                    "distance": float(dmin),
                }
            )
        # Hydrogen-bond geometry reward; it only ranks candidates and never
        # changes a crystallographic protonation assignment.
        for j, acceptor in enumerate(model.atoms):
            if j in {atom_index, int(host)} or acceptor.element not in ACCEPTORS:
                continue
            d_ha, _, vec_ha = minimum_image(model, atom_index, j)
            if 1.35 <= d_ha <= 2.60:
                h_to_host = -vector
                angle = _angle(h_to_host, vec_ha)
                if angle >= 120.0:
                    score -= min(2.0, (angle - 120.0) / 30.0 + 0.4)
                    reasons.append(
                        {
                            "h": atom.label,
                            "acceptor": acceptor.label,
                            "distance": d_ha,
                            "angle": angle,
                        }
                    )
    valence = valence_summary(model, graph)
    for row in valence:
        if row["label"] in {p.get("host_label") for p in placements} and row["deficit"] is not None:
            deficit = float(row["deficit"])
            if deficit > 0.25:
                score += 25.0 * deficit
            elif deficit < -0.35:
                score += 20.0 * abs(deficit)
    return float(score), {
        "reasons": reasons,
        "valence": valence,
        "hard_clashes": hard_clashes[:50],
        "host_bond_violations": host_bond_violations[:50],
    }


def propose_hydrogens(
    model: StructureData,
    *,
    top_k: int = 8,
    max_candidates: int = 128,
    orientation_count: int = 12,
    hetero_policy: str = "enumerate",
    include_unprotonated_hetero: bool = True,
) -> dict[str, Any]:
    if model.cell is None:
        raise ValueError("hydrogen_propose 需要周期晶胞；无晶胞文件请先转换为带晶胞的结构")
    graph = build_bond_graph(model, include_hydrogen=True)
    groups, decisions = _host_groups(model, graph, hetero_policy=hetero_policy)
    if not groups:
        return {
            "candidates": [],
            "decisions": decisions,
            "state": "underdetermined",
            "reason": "未发现可由当前价态启发式安全补入的氢位点；这不等于实验结构无氢。",
        }
    option_groups: list[list[dict[str, Any]]] = []
    for options in groups:
        built: list[dict[str, Any]] = []
        for option in options:
            host = int(option["host"])
            orientation_sets = _orientation_sets(
                model,
                graph,
                host,
                int(option["n_h"]),
                orientation_count=orientation_count,
                hetero=bool(option.get("hetero")),
            )
            for orientation_index, directions in enumerate(orientation_sets):
                built.append(
                    {**option, "directions": directions, "orientation_index": orientation_index}
                )
        if include_unprotonated_hetero and any(x.get("hetero") for x in options):
            built.append(
                {
                    "host": -1,
                    "unprotonated_host": int(options[0]["host"]),
                    "n_h": 0,
                    "kind": "unprotonated_alternative",
                    "directions": [],
                    "orientation_index": None,
                    "hetero": True,
                }
            )
        option_groups.append(built)
    beam: list[tuple[list[dict[str, Any]], float]] = [([], 0.0)]
    for options in option_groups:
        expanded: list[tuple[list[dict[str, Any]], float]] = []
        for previous, _ in beam:
            for option in options:
                placements = previous + [option]
                expanded.append((placements, 0.0))
        # Cheap deterministic pruning before full geometry scoring.
        beam = expanded[:max_candidates]
    candidates: list[dict[str, Any]] = []
    for placements, _ in beam:
        candidate_model = _add_hydrogens(model, placements)
        score, detail = score_candidate(candidate_model, placements)
        added = [
            {
                "host": model.atoms[int(p["host"])].label,
                "element": model.atoms[int(p["host"])].element,
                "n_h": len(p.get("directions", [])),
                "kind": p.get("kind"),
                "orientation_index": p.get("orientation_index"),
            }
            for p in placements
            if int(p.get("host", -1)) >= 0
        ]
        unprotonated = [
            model.atoms[int(p["unprotonated_host"])].label
            for p in placements
            if p.get("kind") == "unprotonated_alternative"
            and p.get("unprotonated_host") is not None
        ]
        candidates.append(
            {
                "model": candidate_model,
                "score": score,
                "detail": detail,
                # Retain the internal host/direction records so an optional
                # ML refinement can be re-audited against the same intended
                # H--host bonds after coordinates move.
                "placements": placements,
                "added_hydrogens": added,
                "unprotonated_sites": unprotonated,
            }
        )
    candidates.sort(key=lambda x: (x["score"], str(x["added_hydrogens"])))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in candidates:
        signature = (
            tuple((x["host"], x["orientation_index"], x["n_h"]) for x in item["added_hydrogens"]),
            tuple(item.get("unprotonated_sites", [])),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
        if len(unique) >= max(1, int(top_k)):
            break
    hetero_ambiguous = any(
        d.get("type") in {"heteroatom_protonation", "mutually_exclusive_protonation"}
        for d in decisions
    )
    state = "ambiguous" if hetero_ambiguous or len(unique) > 1 else "determinate_by_valence"
    return {
        "candidates": unique,
        "decisions": decisions,
        "state": state,
        "option_group_count": len(option_groups),
    }
