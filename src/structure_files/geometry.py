"""Periodic geometry, bond heuristics, and chemistry-aware validation."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .models import StructureData, atomic_number, is_metal

# Pymatgen removed the old covalent_radius attribute in some recent releases;
# keep a local, auditable first-pass table.  These are covalent radii in Å and
# are intentionally used as a screening heuristic, never as a bond-order oracle.
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Li": 1.28,
    "Na": 1.66,
    "K": 2.03,
    "Mg": 1.41,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Al": 1.21,
    "Ga": 1.22,
    "In": 1.42,
    "Tl": 1.45,
    "Sn": 1.40,
    "Pb": 1.46,
}

TARGET_VALENCE = {
    "H": 1.0,
    "B": 3.0,
    "C": 4.0,
    "N": 3.0,
    "O": 2.0,
    "F": 1.0,
    "Si": 4.0,
    "P": 3.0,
    "S": 2.0,
    "Cl": 1.0,
    "Br": 1.0,
    "I": 1.0,
}


def covalent_radius(element: str) -> float:
    key = element.capitalize()
    if key in COVALENT_RADII:
        return COVALENT_RADII[key]
    try:
        from pymatgen.core import Element

        radius = getattr(Element(key), "covalent_radius", None)
        if radius is not None:
            return float(radius)
    except Exception:
        pass
    return 1.0


def target_valence(element: str) -> float | None:
    return TARGET_VALENCE.get(element.capitalize())


def cell_lengths(cell: np.ndarray | None) -> list[float] | None:
    if cell is None:
        return None
    return [float(np.linalg.norm(row)) for row in np.asarray(cell)]


def minimum_image(model: StructureData, i: int, j: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Return distance, integer image for j, and Cartesian vector i→j(image)."""
    if model.cell is None:
        delta = model.coords_cartesian[j] - model.coords_cartesian[i]
        return float(np.linalg.norm(delta)), np.zeros(3, dtype=int), delta
    from pymatgen.core import Lattice

    lattice = Lattice(np.asarray(model.cell, dtype=float))
    fi = model.frac_coords[i]
    fj = model.frac_coords[j]
    distance, image = lattice.get_distance_and_image(fi, fj)
    image = np.asarray(image, dtype=int)
    vector = (fj + image - fi) @ np.asarray(model.cell, dtype=float)
    return float(distance), image, vector


def _estimated_bond_order(
    element_i: str, element_j: str, distance: float, radius_sum: float
) -> float:
    pair = {element_i.capitalize(), element_j.capitalize()}
    # A short contact relative to the covalent sum is a useful indication of a
    # multiple bond for valence completion.  Aromatic bonds stay fractional.
    if pair <= {"C", "N", "O", "S"} and distance < radius_sum - 0.22:
        return 2.0
    if pair == {"C"} and distance < 1.28:
        return 3.0
    if pair <= {"C", "N"} and distance < 1.30:
        return 2.0
    if pair <= {"C", "N"} and distance < 1.43:
        return 1.5
    return 1.0


def build_bond_graph(
    model: StructureData,
    *,
    tolerance: float = 0.45,
    include_hydrogen: bool = True,
    metal_tolerance: float = 0.65,
    max_distance: float | None = None,
) -> nx.Graph:
    """Build a periodic first-pass covalent/coordination graph.

    Each edge carries ``image`` (the periodic image applied to node ``j``),
    ``distance``, ``bond_order`` and ``kind``.  The graph is deliberately
    conservative: callers must inspect functional groups and oxidation states
    before treating an edge as a chemically final bond.
    """
    graph = nx.Graph()
    for i, atom in enumerate(model.atoms):
        graph.add_node(i, label=atom.label, element=atom.element, occupancy=atom.occupancy)
    for i, atom_i in enumerate(model.atoms):
        for j in range(i + 1, model.n_atoms):
            atom_j = model.atoms[j]
            if not include_hydrogen and (atom_i.is_hydrogen or atom_j.is_hydrogen):
                continue
            distance, image, _ = minimum_image(model, i, j)
            if max_distance is not None and distance > max_distance:
                continue
            if distance < 0.25:
                # Exact duplicates are reported by validation, not added as
                # bonds, because they would corrupt valence completion.
                continue
            ri, rj = covalent_radius(atom_i.element), covalent_radius(atom_j.element)
            metal_pair = is_metal(atom_i.element) or is_metal(atom_j.element)
            threshold = ri + rj + (metal_tolerance if metal_pair else tolerance)
            if distance > threshold:
                continue
            # Do not turn two isolated metals into a normal covalent bond just
            # because their generous radii overlap.
            if metal_pair and is_metal(atom_i.element) and is_metal(atom_j.element):
                if distance > ri + rj - 0.1:
                    continue
            kind = (
                "coordination"
                if metal_pair and not (atom_i.is_hydrogen or atom_j.is_hydrogen)
                else "covalent"
            )
            order = (
                1.0
                if kind == "coordination"
                else _estimated_bond_order(atom_i.element, atom_j.element, distance, ri + rj)
            )
            confidence = max(0.0, min(1.0, 1.0 - abs(distance - (ri + rj)) / max(threshold, 1e-6)))
            graph.add_edge(
                i,
                j,
                image=image.tolist(),
                distance=float(distance),
                threshold=float(threshold),
                bond_order=float(order),
                kind=kind,
                confidence=float(confidence),
            )
    return graph


def graph_to_dict(graph: nx.Graph, model: StructureData) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    for i, j, data in graph.edges(data=True):
        edges.append(
            {
                "i": i,
                "j": j,
                "label_i": model.atoms[i].label,
                "label_j": model.atoms[j].label,
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()},
            }
        )
    return {
        "nodes": [
            {"index": i, "label": atom.label, "element": atom.element, "occupancy": atom.occupancy}
            for i, atom in enumerate(model.atoms)
        ],
        "edges": sorted(edges, key=lambda x: (x["i"], x["j"])),
        "n_nodes": model.n_atoms,
        "n_edges": len(edges),
        "components": [
            sorted(int(x) for x in component) for component in nx.connected_components(graph)
        ],
    }


def shortest_contacts(model: StructureData, *, limit: int = 30) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for i in range(model.n_atoms):
        for j in range(i + 1, model.n_atoms):
            distance, image, vector = minimum_image(model, i, j)
            contacts.append(
                {
                    "i": i,
                    "j": j,
                    "label_i": model.atoms[i].label,
                    "label_j": model.atoms[j].label,
                    "element_i": model.atoms[i].element,
                    "element_j": model.atoms[j].element,
                    "distance": float(distance),
                    "image": image.tolist(),
                    "vector": [float(x) for x in vector],
                }
            )
    contacts.sort(key=lambda x: x["distance"])
    return contacts[: max(0, int(limit))]


def periodic_duplicate_contacts(
    model: StructureData, threshold: float = 0.50
) -> list[dict[str, Any]]:
    return [
        x
        for x in shortest_contacts(model, limit=model.n_atoms * max(model.n_atoms - 1, 1))
        if x["distance"] < threshold
    ]


def coordination_summary(
    model: StructureData, graph: nx.Graph | None = None
) -> list[dict[str, Any]]:
    graph = graph or build_bond_graph(model)
    rows: list[dict[str, Any]] = []
    for i, atom in enumerate(model.atoms):
        if not is_metal(atom.element):
            continue
        neighbors = []
        for j in graph.neighbors(i):
            data = graph.edges[i, j]
            neighbors.append(
                {
                    "index": int(j),
                    "label": model.atoms[j].label,
                    "element": model.atoms[j].element,
                    "distance": float(data["distance"]),
                    "image": list(data.get("image", [0, 0, 0])),
                }
            )
        rows.append(
            {
                "index": i,
                "label": atom.label,
                "element": atom.element,
                "coordination_number": len(neighbors),
                "neighbors": neighbors,
            }
        )
    return rows


def spglib_summary(model: StructureData, *, symprec: float = 0.1) -> dict[str, Any]:
    if model.cell is None or model.coords_are_cartesian or not model.atoms:
        return {"available": False, "reason": "无周期晶胞或无原子"}
    try:
        import spglib

        numbers = [atomic_number(atom.element.split("/")[0]) for atom in model.atoms]
        if any(number <= 0 for number in numbers):
            return {"available": False, "reason": "存在未知元素，跳过 Spglib"}
        cell = (
            np.asarray(model.cell, dtype=float),
            model.frac_coords,
            np.asarray(numbers, dtype=int),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            return {"available": True, "found": False, "symprec": symprec}

        # New and old Spglib return either a dataclass-like object or a dict.
        def get(key: str, default: Any = None) -> Any:
            try:
                value = getattr(dataset, key)
                return default if value is None else value
            except Exception:
                try:
                    return dataset[key]
                except Exception:
                    return default

        return {
            "available": True,
            "found": True,
            "symprec": symprec,
            "number": int(get("number", 0) or 0),
            "international": str(get("international", "")),
            "hall": str(get("hall", "")),
            "pointgroup": str(get("pointgroup", "")),
            "n_operations": int(len(get("rotations", []))),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def valence_summary(model: StructureData, graph: nx.Graph | None = None) -> list[dict[str, Any]]:
    graph = graph or build_bond_graph(model)
    rows: list[dict[str, Any]] = []
    for i, atom in enumerate(model.atoms):
        valence = sum(
            float(data.get("bond_order", 1.0))
            for _, _, data in graph.edges(i, data=True)
            if data.get("kind") == "covalent"
        )
        target = target_valence(atom.element)
        rows.append(
            {
                "index": i,
                "label": atom.label,
                "element": atom.element,
                "covalent_valence": round(valence, 4),
                "target_valence": target,
                "deficit": None if target is None else round(float(target - valence), 4),
                "n_neighbors": int(graph.degree(i)),
            }
        )
    return rows


def validate_geometry(
    model: StructureData, *, contact_limit: int = 30, symprec: float = 0.1
) -> dict[str, Any]:
    graph = build_bond_graph(model)
    contacts = shortest_contacts(model, limit=contact_limit)
    duplicates = [c for c in contacts if c["distance"] < 0.50]
    suspicious = [
        c
        for c in contacts
        if c["distance"] < (1.20 if c["element_i"] == "H" or c["element_j"] == "H" else 1.00)
    ]
    out_of_range = []
    if model.cell is not None and not model.coords_are_cartesian:
        for i, xyz in enumerate(model.frac_coords):
            if np.any(xyz < -1e-6) or np.any(xyz >= 1.0 + 1e-6):
                out_of_range.append(
                    {"index": i, "label": model.atoms[i].label, "coords": xyz.tolist()}
                )
    return {
        "minimum_distance": contacts[0]["distance"] if contacts else None,
        "shortest_contacts": contacts,
        "duplicate_contact_count": len(duplicates),
        "suspicious_contact_count": len(suspicious),
        "out_of_range_fractional_count": len(out_of_range),
        "out_of_range_fractional": out_of_range[:50],
        "bond_graph": graph_to_dict(graph, model),
        "valence": valence_summary(model, graph),
        "coordination": coordination_summary(model, graph),
        "spglib": spglib_summary(model, symprec=float(symprec)),
    }
