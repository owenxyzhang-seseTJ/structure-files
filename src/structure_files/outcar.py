"""Read-only OUTCAR extraction.

This parser intentionally does not attempt to repair, rerun, or rewrite an
OUTCAR.  It collects only complete POSITION/TOTAL-FORCE blocks and associates
the nearest preceding lattice and energy records.  A truncated final block is
reported as truncated and omitted from the complete trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import AtomRecord, StructureData

FLOAT = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?"


@dataclass
class OutcarStep:
    index: int
    positions_cartesian: np.ndarray
    forces: np.ndarray | None = None
    cell: np.ndarray | None = None
    energy_free_eV: float | None = None
    energy_sigma0_eV: float | None = None
    stress_kbar: np.ndarray | None = None


@dataclass
class OutcarResult:
    path: str
    nions: int | None
    elements: list[str]
    steps: list[OutcarStep] = field(default_factory=list)
    lattice: np.ndarray | None = None
    energies: list[float] = field(default_factory=list)
    truncated_position_block: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def last_structure(self) -> StructureData:
        if not self.steps:
            raise ValueError("OUTCAR 没有完整离子步")
        step = self.steps[-1]
        elements = self.elements or ["X"] * len(step.positions_cartesian)
        if len(elements) != len(step.positions_cartesian):
            elements = (elements + ["X"] * len(step.positions_cartesian))[
                : len(step.positions_cartesian)
            ]
        atoms: list[AtomRecord] = []
        if step.cell is not None and abs(float(np.linalg.det(step.cell))) > 1e-10:
            frac = step.positions_cartesian @ np.linalg.inv(step.cell)
            for i, (element, xyz) in enumerate(zip(elements, frac, strict=True)):
                props: dict[str, Any] = {"outcar_step": step.index}
                if step.forces is not None:
                    props["force_eV_A"] = [float(x) for x in step.forces[i]]
                atoms.append(AtomRecord(f"{element}{i + 1}", element, xyz, properties=props))
            model = StructureData(
                atoms=atoms,
                cell=step.cell,
                pbc=(True, True, True),
                coords_are_cartesian=False,
                format="outcar",
                source_path=self.path,
                metadata={
                    "outcar_step": step.index,
                    "energy_free_eV": step.energy_free_eV,
                    "energy_sigma0_eV": step.energy_sigma0_eV,
                    "stress_kbar": None
                    if step.stress_kbar is None
                    else [float(x) for x in step.stress_kbar],
                    "truncated_position_block": self.truncated_position_block,
                },
            )
        else:
            for i, (element, xyz) in enumerate(
                zip(elements, step.positions_cartesian, strict=True)
            ):
                props = {"outcar_step": step.index}
                if step.forces is not None:
                    props["force_eV_A"] = [float(x) for x in step.forces[i]]
                atoms.append(AtomRecord(f"{element}{i + 1}", element, xyz, properties=props))
            model = StructureData(
                atoms=atoms,
                cell=None,
                pbc=(False, False, False),
                coords_are_cartesian=True,
                format="outcar",
                source_path=self.path,
                metadata={
                    "outcar_step": step.index,
                    "energy_free_eV": step.energy_free_eV,
                    "energy_sigma0_eV": step.energy_sigma0_eV,
                    "stress_kbar": None
                    if step.stress_kbar is None
                    else [float(x) for x in step.stress_kbar],
                    "truncated_position_block": self.truncated_position_block,
                },
            )
        return model


def _numbers(line: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in re.findall(FLOAT, line)]


def _find_lattice(lines: list[str], start: int) -> tuple[np.ndarray | None, int]:
    for i in range(start, min(len(lines), start + 5)):
        vals = _numbers(lines[i])
        if len(vals) >= 6:
            return np.asarray([vals[:3], vals[3:6], vals[6:9]], dtype=float) if len(
                vals
            ) >= 9 else None, i
    if start + 3 < len(lines):
        rows = [_numbers(lines[start + k]) for k in range(1, 4)]
        if all(len(row) >= 3 for row in rows):
            return np.asarray([row[:3] for row in rows], dtype=float), start + 3
    return None, start


def parse_outcar(path: str | Path) -> OutcarResult:
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    nions: int | None = None
    for line in lines:
        match = re.search(r"NIONS\s*=\s*(\d+)", line)
        if match:
            nions = int(match.group(1))
            break
    elements: list[str] = []
    for line in lines:
        match = re.search(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", line)
        if match and match.group(1) not in elements:
            elements.append(match.group(1))
    counts: list[int] = []
    for line in lines:
        if "ions per type" in line.lower():
            vals = re.findall(r"\d+", line.split("=")[-1])
            counts = [int(x) for x in vals]
            break
    if counts and elements and sum(counts) == (nions or sum(counts)):
        expanded: list[str] = []
        for element, count in zip(elements, counts, strict=True):
            expanded.extend([element] * count)
        elements = expanded
    result = OutcarResult(path=str(p), nions=nions, elements=elements)
    current_cell: np.ndarray | None = None
    current_free: float | None = None
    current_sigma: float | None = None
    current_stress: np.ndarray | None = None
    step_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if "direct lattice vectors" in line.lower():
            rows: list[list[float]] = []
            for j in range(i + 1, min(len(lines), i + 6)):
                vals = _numbers(lines[j])
                if len(vals) >= 6:
                    rows.append(vals[:3])
                if len(rows) == 3:
                    break
            if len(rows) == 3:
                current_cell = np.asarray(rows, dtype=float)
                result.lattice = current_cell.copy()
            i += 1
            continue
        match = re.search(r"free\s+energy\s+TOTEN\s*=\s*(%s)" % FLOAT, line, re.IGNORECASE)
        if match:
            current_free = float(match.group(1).replace("D", "E").replace("d", "e"))
            i += 1
            continue
        match = re.search(r"energy\(sigma->0\)\s*=\s*(%s)" % FLOAT, line, re.IGNORECASE)
        if match:
            current_sigma = float(match.group(1).replace("D", "E").replace("d", "e"))
            result.energies.append(current_sigma)
            i += 1
            continue
        if "in kB" in line and "stress" in line.lower():
            vals = _numbers(line)
            if len(vals) >= 6:
                current_stress = np.asarray(vals[-6:], dtype=float)
        if "POSITION" in line and "TOTAL-FORCE" in line:
            rows: list[list[float]] = []
            j = i + 1
            # Skip the dashed separator and blank lines.
            while j < len(lines) and len(rows) < (nions or 10**9):
                vals = _numbers(lines[j])
                if len(vals) >= 6:
                    rows.append(vals[:6])
                elif rows and ("---" in lines[j] or not lines[j].strip()):
                    break
                elif rows and len(vals) == 0:
                    break
                j += 1
            if nions is not None and len(rows) == nions:
                step = OutcarStep(
                    index=step_index,
                    positions_cartesian=np.asarray([row[:3] for row in rows], dtype=float),
                    forces=np.asarray([row[3:6] for row in rows], dtype=float),
                    cell=None if current_cell is None else current_cell.copy(),
                    energy_free_eV=current_free,
                    energy_sigma0_eV=current_sigma,
                    stress_kbar=None if current_stress is None else current_stress.copy(),
                )
                result.steps.append(step)
                step_index += 1
            else:
                result.truncated_position_block = True
                result.warnings.append(
                    f"第 {step_index} 个 POSITION 块不完整，已忽略 ({len(rows)}/{nions or '?'})"
                )
            i = max(i + 1, j)
            continue
        i += 1
    if nions is None:
        result.warnings.append("未找到 NIONS；仅在能完整识别位置行时提取")
    if not result.steps:
        result.warnings.append("没有可提取的完整离子步")
    return result
