"""Small, JSON-friendly internal structure model.

The model deliberately keeps fractional coordinates, occupancies, disorder
fields, and source metadata separate.  Pymatgen/ASE objects are adapters, not
the provenance record, so a parser quirk cannot silently discard an occupancy
or an alternate-location label.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AtomRecord:
    label: str
    element: str
    coords: np.ndarray | list[float] | tuple[float, float, float]
    occupancy: float = 1.0
    charge: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(3)
        raw_element = str(self.element).strip()
        if "/" in raw_element:
            self.element = "/".join(
                part.strip().capitalize() for part in raw_element.split("/") if part.strip()
            )
        else:
            self.element = raw_element.capitalize()
        self.occupancy = float(self.occupancy if self.occupancy is not None else 1.0)
        if not np.isfinite(self.coords).all():
            raise ValueError(f"坐标不是有限数: {self.label}")

    @property
    def is_hydrogen(self) -> bool:
        return self.element == "H"

    def copy(self) -> "AtomRecord":
        return AtomRecord(
            label=self.label,
            element=self.element,
            coords=self.coords.copy(),
            occupancy=self.occupancy,
            charge=self.charge,
            properties=deepcopy(self.properties),
        )

    def as_dict(self, coords_are_cartesian: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "element": self.element,
            "coords": [float(x) for x in self.coords],
            "occupancy": float(self.occupancy),
        }
        if self.charge is not None:
            out["charge"] = float(self.charge)
        if self.properties:
            out["properties"] = deepcopy(self.properties)
        out["coordinate_type"] = "cartesian" if coords_are_cartesian else "fractional"
        return out


@dataclass
class StructureData:
    """Canonical structure representation used by all file operations."""

    atoms: list[AtomRecord]
    cell: np.ndarray | list[list[float]] | None = None
    pbc: tuple[bool, bool, bool] = (True, True, True)
    coords_are_cartesian: bool = False
    format: str = "unknown"
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atoms = list(self.atoms)
        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
            if not np.isfinite(self.cell).all() or abs(float(np.linalg.det(self.cell))) < 1e-12:
                raise ValueError("晶胞矩阵为空、非有限或退化")
        self.pbc = tuple(bool(x) for x in self.pbc)  # type: ignore[assignment]
        if self.cell is None:
            self.pbc = (False, False, False)

    def copy(self) -> "StructureData":
        return StructureData(
            atoms=[a.copy() for a in self.atoms],
            cell=None if self.cell is None else self.cell.copy(),
            pbc=self.pbc,
            coords_are_cartesian=self.coords_are_cartesian,
            format=self.format,
            source_path=self.source_path,
            metadata=deepcopy(self.metadata),
        )

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def frac_coords(self) -> np.ndarray:
        if self.coords_are_cartesian:
            if self.cell is None:
                raise ValueError("无晶胞，不能把笛卡尔坐标转成分数坐标")
            return np.asarray(self.coords_cartesian) @ np.linalg.inv(self.cell)
        return np.asarray([a.coords for a in self.atoms], dtype=float)

    @property
    def coords_cartesian(self) -> np.ndarray:
        if self.coords_are_cartesian:
            return np.asarray([a.coords for a in self.atoms], dtype=float)
        if self.cell is None:
            raise ValueError("无晶胞，不能把分数坐标转成笛卡尔坐标")
        return np.asarray([a.coords for a in self.atoms], dtype=float) @ self.cell

    def set_frac_coords(self, coords: Iterable[Iterable[float]]) -> None:
        vals = np.asarray(list(coords), dtype=float)
        if vals.shape != (self.n_atoms, 3):
            raise ValueError("坐标数量与原子数不一致")
        self.coords_are_cartesian = False
        for atom, xyz in zip(self.atoms, vals, strict=True):
            atom.coords = xyz.copy()

    def set_cart_coords(self, coords: Iterable[Iterable[float]]) -> None:
        vals = np.asarray(list(coords), dtype=float)
        if vals.shape != (self.n_atoms, 3):
            raise ValueError("坐标数量与原子数不一致")
        self.coords_are_cartesian = True
        for atom, xyz in zip(self.atoms, vals, strict=True):
            atom.coords = xyz.copy()

    def composition(self, weighted: bool = True) -> dict[str, float]:
        result: dict[str, float] = {}
        for atom in self.atoms:
            amount = atom.occupancy if weighted else 1.0
            result[atom.element] = result.get(atom.element, 0.0) + amount
        return dict(sorted(result.items()))

    def formula(self, weighted: bool = True) -> str:
        comp = self.composition(weighted=weighted)
        keys = list(comp)
        if "C" in comp:
            keys = ["C"] + (["H"] if "H" in comp else []) + [k for k in keys if k not in {"C", "H"}]
        parts: list[str] = []
        for element in keys:
            amount = comp[element]
            if abs(amount - round(amount)) < 1e-6:
                text = str(int(round(amount)))
            else:
                text = f"{amount:.5g}"
            parts.append(element + ("" if text == "1" else text))
        return "".join(parts) or "(empty)"

    def labels(self) -> list[str]:
        return [a.label for a in self.atoms]

    def as_dict(self, include_coords: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "source_path": self.source_path,
            "n_atoms": self.n_atoms,
            "formula": self.formula(),
            "composition": self.composition(),
            "pbc": list(self.pbc),
            "coords_are_cartesian": self.coords_are_cartesian,
            "metadata": deepcopy(self.metadata),
        }
        if self.cell is not None:
            result["cell"] = [[float(x) for x in row] for row in self.cell]
        if include_coords:
            result["atoms"] = [a.as_dict(self.coords_are_cartesian) for a in self.atoms]
        return result

    def to_pymatgen(self):
        """Return a Pymatgen Structure when a periodic cell is available."""
        from pymatgen.core import Lattice, Structure

        if self.cell is None:
            raise ValueError("无晶胞的结构不能构造 Pymatgen Structure")
        species: list[Any] = []
        for atom in self.atoms:
            if abs(atom.occupancy - 1.0) < 1e-8:
                species.append(atom.element)
            else:
                species.append({atom.element: atom.occupancy})
        props: dict[str, list[Any]] = {"_dsh_label": [a.label for a in self.atoms]}
        for key in sorted({k for a in self.atoms for k in a.properties}):
            props[key] = [a.properties.get(key) for a in self.atoms]
        return Structure(
            Lattice(np.asarray(self.cell, dtype=float)),
            species,
            self.frac_coords,
            coords_are_cartesian=False,
            site_properties=props,
            to_unit_cell=False,
        )

    @classmethod
    def from_pymatgen(
        cls, structure: Any, *, format: str = "pymatgen", source_path: str | None = None
    ) -> "StructureData":
        atoms: list[AtomRecord] = []
        labels = structure.site_properties.get("_dsh_label", [])
        for i, site in enumerate(structure.sites):
            comp = getattr(site, "species", None)
            element = (
                str(site.specie.symbol) if hasattr(site, "specie") else str(next(iter(comp)).symbol)
            )
            occupancy = float(getattr(site, "occupancy", 1.0))
            if comp is not None and len(comp) > 1:
                # Keep mixed species explicit instead of inventing one ordered atom.
                element = "/".join(sorted(str(s.symbol) for s in comp))
                occupancy = float(sum(float(v) for v in comp.values()))
            props: dict[str, Any] = {}
            for key, values in structure.site_properties.items():
                if key == "_dsh_label":
                    continue
                if i < len(values):
                    props[key] = values[i]
            atoms.append(
                AtomRecord(
                    label=str(labels[i] if i < len(labels) else f"{element}{i + 1}"),
                    element=element,
                    coords=np.asarray(site.frac_coords, dtype=float),
                    occupancy=occupancy,
                    properties=props,
                )
            )
        return cls(
            atoms=atoms,
            cell=np.asarray(structure.lattice.matrix, dtype=float),
            pbc=(True, True, True),
            coords_are_cartesian=False,
            format=format,
            source_path=source_path,
        )


def atomic_number(element: str) -> int:
    try:
        from pymatgen.core import Element

        return int(Element(element).Z)
    except Exception:
        # Enough of a fallback for diagnostics when the optional chemistry
        # dependency is unavailable.
        fallback = {
            "H": 1,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Br": 35,
            "I": 53,
        }
        return fallback.get(element.capitalize(), 0)


def is_metal(element: str) -> bool:
    try:
        from pymatgen.core import Element

        return bool(Element(element).is_metal)
    except Exception:
        return element.capitalize() in {
            "Li",
            "Na",
            "K",
            "Rb",
            "Cs",
            "Mg",
            "Ca",
            "Sr",
            "Ba",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Al",
            "Ga",
            "In",
            "Tl",
            "Sn",
            "Pb",
            "Bi",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Th",
            "U",
        }
