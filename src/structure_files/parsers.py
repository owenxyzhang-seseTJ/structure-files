"""CIF/POSCAR/XYZ readers and symmetry handling."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .models import AtomRecord, StructureData
from .utils import parse_cif_number

FORMAT_ALIASES = {
    "cif": "cif",
    "mmcif": "cif",
    "mcif": "cif",
    "poscar": "poscar",
    "contcar": "poscar",
    "vasp": "poscar",
    "xyz": "xyz",
    "extxyz": "extxyz",
    "outcar": "outcar",
}


def detect_format(path: str | Path) -> str:
    p = Path(path)
    name = p.name.upper()
    suffix = p.suffix.lower().lstrip(".")
    if name in {"POSCAR", "CONTCAR"} or name.startswith("POSCAR.") or name.startswith("CONTCAR."):
        return "poscar"
    if name == "OUTCAR" or name.startswith("OUTCAR."):
        return "outcar"
    if suffix in {"cif", "mcif", "mmcif"}:
        return "cif"
    if suffix in {"vasp", "poscar", "contcar"}:
        return "poscar"
    if suffix in {"xyz", "extxyz"}:
        return "extxyz" if suffix == "extxyz" else "xyz"
    # Content sniffing for files that were renamed without an extension.
    try:
        head = p.read_text(encoding="utf-8", errors="replace")[:4096]
    except Exception:
        return "unknown"
    if "_atom_site_" in head or "data_" in head and "_cell_" in head:
        return "cif"
    if "direct lattice vectors" in head or "POSITION" in head and "TOTAL-FORCE" in head:
        return "outcar"
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    if len(lines) >= 8 and re.fullmatch(r"\d+", lines[0] or ""):
        return "xyz"
    if len(lines) >= 5:
        return "poscar"
    return "unknown"


def _first_value(block: Any, tag: str, default: Any = None) -> Any:
    try:
        column = block.find_values(tag)
        if column and len(column):
            return column[0]
    except Exception:
        pass
    return default


def _has_atom_rows(block: Any) -> bool:
    """Return whether a Gemmi block contains a crystallographic atom loop.

    publCIF and deposition files often put a ``publication_text`` or audit
    block before the actual crystallographic block.  Selecting ``doc[0]`` in
    that situation makes a perfectly valid CIF look malformed.  Keep this
    test deliberately narrow: an atom *loop* with labels and coordinates is
    the representation we can safely convert into :class:`StructureData`.
    """

    for tag in ("_atom_site_label", "_atom_site_type_symbol"):
        try:
            if block.find_loop(tag):
                return True
        except Exception:
            continue
    return False


def _select_cif_block(doc: Any) -> tuple[Any, int, str]:
    """Choose the block containing the crystallographic atom loop.

    If a document contains several atom-bearing blocks, prefer one that also
    has a unit cell and then the first such block.  The choice is recorded in
    metadata so callers can audit rather than silently trusting a parser's
    default block selection.
    """

    blocks = list(doc)
    if not blocks:
        raise ValueError("CIF 文档没有 data block")
    atom_blocks = [(index, block) for index, block in enumerate(blocks) if _has_atom_rows(block)]
    if not atom_blocks:
        # Preserve the old failure mode with a useful block name.  The caller
        # will still try the Pymatgen fallback and report both parser errors.
        return blocks[0], 0, "no_atom_block_fallback"
    with_cell = [
        (index, block)
        for index, block in atom_blocks
        if _first_value(block, "_cell_length_a") not in (None, "", ".", "?")
    ]
    selected = with_cell or atom_blocks
    index, block = selected[0]
    if len(atom_blocks) == 1:
        reason = "唯一含 atom_site loop 的 block"
    elif with_cell:
        reason = "多个 atom block 中优先选择含晶胞参数的 block"
    else:
        reason = "多个 atom block 中选择首个"
    return block, int(index), reason


def _element_from_text(value: Any, label: str = "X") -> str:
    text = str(value or "").strip().strip("'").strip('"')
    text = re.sub(r"[+\-].*$", "", text)
    if text:
        # Type symbols may be decorated (e.g. C1, Zn2+, FE).  Prefer a
        # valid two-letter token before falling back to a one-letter symbol;
        # this avoids interpreting ``CA``/``FE`` as C/F when capitalization
        # was lost in an upstream export.
        m = re.search(r"([A-Za-z]{1,2})", text)
        if m:
            raw = m.group(1)
            candidates = [raw]
            if len(raw) == 2:
                candidates.append(raw[0].upper() + raw[1].lower())
            candidates.append(raw[0].upper())
            for token in candidates:
                try:
                    from pymatgen.core import Element

                    return str(Element(token).symbol)
                except Exception:
                    if token in {"D", "T"}:
                        return "H"
                    continue
    m = re.match(r"([A-Za-z]{1,2})", str(label))
    if m:
        token = m.group(1).capitalize()
        try:
            from pymatgen.core import Element

            Element(token)
            return token
        except Exception:
            pass
    return "X"


def _get_row_value(
    loop: Any, indices: dict[str, int], i: int, names: list[str], default: Any = None
) -> Any:
    for name in names:
        j = indices.get(name.lower())
        if j is not None:
            try:
                return loop[i, j]
            except Exception:
                return default
    return default


def _parse_geom_bonds(block: Any) -> list[dict[str, Any]]:
    """Read an existing CIF ``_geom_bond`` loop without assigning new bonds."""

    try:
        column = block.find_loop("_geom_bond_atom_site_label_1")
        if not column:
            return []
        loop = column.get_loop()
    except Exception:
        return []
    indices = {str(tag).lower(): i for i, tag in enumerate(loop.tags)}
    records: list[dict[str, Any]] = []
    for row_index in range(loop.length()):
        left = _get_row_value(
            loop, indices, row_index, ["_geom_bond_atom_site_label_1"], None
        )
        right = _get_row_value(
            loop, indices, row_index, ["_geom_bond_atom_site_label_2"], None
        )
        if left in (None, "", ".", "?") or right in (None, "", ".", "?"):
            continue
        record: dict[str, Any] = {
            "label_1": str(left).strip("'\""),
            "label_2": str(right).strip("'\""),
            "distance_A": parse_cif_number(
                _get_row_value(loop, indices, row_index, ["_geom_bond_distance"], None)
            ),
        }
        for key, names in {
            "site_symmetry_2": ["_geom_bond_site_symmetry_2"],
            "type": ["_geom_bond_type"],
            "publ_flag": ["_geom_bond_publ_flag"],
            "dsh_kind": ["_dsh_geom_bond_kind"],
            "dsh_order": ["_dsh_geom_bond_order"],
            "dsh_confidence": ["_dsh_geom_bond_confidence"],
            "dsh_image_x": ["_dsh_geom_bond_image_x"],
            "dsh_image_y": ["_dsh_geom_bond_image_y"],
            "dsh_image_z": ["_dsh_geom_bond_image_z"],
        }.items():
            value = _get_row_value(loop, indices, row_index, names, None)
            if value not in (None, "", ".", "?"):
                parsed = parse_cif_number(value)
                record[key] = (
                    parsed
                    if parsed is not None and key.startswith("dsh_")
                    else str(value).strip("'\"")
                )
        records.append(record)
    return records


def _cell_from_parameters(values: dict[str, Any]) -> np.ndarray | None:
    a = parse_cif_number(values.get("a"))
    b = parse_cif_number(values.get("b"))
    c = parse_cif_number(values.get("c"))
    alpha = np.deg2rad(parse_cif_number(values.get("alpha"), 90.0) or 90.0)
    beta = np.deg2rad(parse_cif_number(values.get("beta"), 90.0) or 90.0)
    gamma = np.deg2rad(parse_cif_number(values.get("gamma"), 90.0) or 90.0)
    if not all(x is not None and x > 0 for x in (a, b, c)):
        return None
    # Pymatgen's Lattice.from_parameters uses row vectors.  Reproduce it here
    # so Gemmi remains the source parser and no symmetry/occupancy information
    # is lost in a fallback conversion.
    ax = float(a)
    bx = float(b) * np.cos(gamma)
    by = float(b) * np.sin(gamma)
    cx = float(c) * np.cos(beta)
    sin_gamma = np.sin(gamma)
    cy = float(c) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cz_sq = float(c) ** 2 - cx**2 - cy**2
    if cz_sq <= 0:
        return None
    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, np.sqrt(cz_sq)]], dtype=float)


def _space_group_details(symbol: Any) -> dict[str, Any]:
    """Return Gemmi's canonical names for a CIF space-group symbol.

    CIF writers use several valid spellings for the same group (for example
    ``P 21/c`` and the full setting ``P 1 21/c 1``).  Keeping both the
    declared and canonical spellings lets downstream parsers use the form
    they support without changing the source file or its provenance.
    """

    raw = str(symbol or "").strip().strip("'\"")
    if not raw or raw in {".", "?"}:
        return {}
    try:
        import gemmi

        group = gemmi.find_spacegroup_by_name(raw)
        if group is None:
            return {}
        short_name = group.short_name()
        return {
            "number": int(group.number),
            "hm": str(group.hm),
            "short_name": str(short_name),
            "hall": str(group.hall),
            "crystal_system": str(group.crystal_system_str()),
        }
    except Exception:
        return {}


def _atom_matches(
    left: AtomRecord, right: AtomRecord, cell: np.ndarray, tolerance_angstrom: float
) -> bool:
    """Compare two raw rows while retaining disorder/occupancy semantics."""

    if left.element != right.element:
        return False
    if abs(float(left.occupancy) - float(right.occupancy)) > 1e-6:
        return False
    for key in ("disorder_group", "disorder_assembly", "alt_id"):
        if left.properties.get(key) != right.properties.get(key):
            return False
    return _periodic_close(left.coords, right.coords, cell, tolerance_angstrom)


def _symmetry_closed(
    model: StructureData, ops: list[str], *, tolerance_angstrom: float = 0.02
) -> bool:
    """Whether the atom rows are already closed under the declared symmetry.

    A CIF may contain a complete, viewer-oriented set of sites while still
    declaring its crystallographic space group.  Expanding such a file again
    would multiply every atom.  Closure is checked against element,
    occupancy, disorder role, and periodic coordinates rather than distance
    alone, so distinct atoms at a short contact are not silently removed.
    """

    if model.cell is None or model.coords_are_cartesian or not ops:
        return False
    try:
        from pymatgen.core.operations import SymmOp
    except Exception:
        return False
    parsers = getattr(SymmOp, "from_xyz_string", None)
    if parsers is None:
        parsers = SymmOp.from_xyz_str
    cell = np.asarray(model.cell, dtype=float)
    for atom in model.atoms:
        for op_text in ops:
            try:
                frac = np.asarray(parsers(op_text).operate(atom.coords), dtype=float)
            except Exception:
                return False
            generated = AtomRecord(
                atom.label,
                atom.element,
                frac - np.floor(frac),
                atom.occupancy,
                atom.charge,
                atom.properties,
            )
            if not any(
                _atom_matches(generated, candidate, cell, tolerance_angstrom)
                for candidate in model.atoms
            ):
                return False
    return True


def parse_cif(path: str | Path) -> StructureData:
    """Parse a CIF with Gemmi while preserving raw atom/symmetry metadata."""
    p = Path(path).expanduser().resolve()
    try:
        import gemmi
    except (
        Exception
    ) as exc:  # pragma: no cover - dependency is core, fallback is useful in stripped envs
        return _parse_cif_with_pymatgen(p, exc)

    try:
        doc = gemmi.cif.read_file(str(p))
        block, block_index, block_selection_reason = _select_cif_block(doc)
        values = {
            "a": _first_value(block, "_cell_length_a"),
            "b": _first_value(block, "_cell_length_b"),
            "c": _first_value(block, "_cell_length_c"),
            "alpha": _first_value(block, "_cell_angle_alpha"),
            "beta": _first_value(block, "_cell_angle_beta"),
            "gamma": _first_value(block, "_cell_angle_gamma"),
        }
        cell = _cell_from_parameters(values)
        declared_space_group = _first_value(block, "_symmetry_space_group_name_H-M")
        if declared_space_group is None:
            declared_space_group = _first_value(block, "_space_group_name_H-M_alt")
        declared_space_group = (
            str(declared_space_group).strip().strip("'\"")
            if declared_space_group not in (None, "", ".", "?")
            else None
        )
        space_group_details = _space_group_details(declared_space_group)
        sym_ops: list[str] = []
        for tag in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            try:
                col = block.find_loop(tag)
                if col:
                    loop = col.get_loop()
                    sym_ops = [
                        str(loop[i, 0]).strip().strip("'").strip('"') for i in range(loop.length())
                    ]
                    if sym_ops:
                        break
            except Exception:
                continue
        symmetry_source = "explicit_loop" if sym_ops else None
        # CIFs commonly provide only a Hermann–Mauguin symbol.  Gemmi knows
        # both short and full symbols (for example ``P21/c`` and
        # ``P 1 21/c 1``), so use it to materialize the operations while
        # retaining the declared symbol as provenance.  Without this step a
        # valid asymmetric unit would silently remain unexpanded.
        if not sym_ops and declared_space_group not in (None, "", ".", "?"):
            try:
                space_group = gemmi.find_spacegroup_by_name(str(declared_space_group))
                if space_group is not None:
                    sym_ops = [str(op.triplet()) for op in space_group.operations().sym_ops]
                    symmetry_source = "space_group_symbol"
            except Exception:
                # Keep parsing the atom loop; validate/inspect will report
                # that symmetry could not be expanded rather than rejecting a
                # coordinate-bearing CIF solely for missing operations.
                symmetry_source = None
        atom_col = None
        for tag in ("_atom_site_label", "_atom_site_type_symbol"):
            try:
                atom_col = block.find_loop(tag)
                if atom_col:
                    break
            except Exception:
                pass
        if not atom_col:
            raise ValueError("CIF 没有 _atom_site 原子循环")
        loop = atom_col.get_loop()
        indices = {str(tag).lower(): i for i, tag in enumerate(loop.tags)}
        atoms: list[AtomRecord] = []
        for i in range(loop.length()):
            label = str(_get_row_value(loop, indices, i, ["_atom_site_label"], f"X{i + 1}")).strip(
                "'\""
            )
            symbol = _get_row_value(
                loop, indices, i, ["_atom_site_type_symbol", "_atom_site_symbol"], None
            )
            element = _element_from_text(symbol, label)
            x = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_fract_x"], None))
            y = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_fract_y"], None))
            z = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_fract_z"], None))
            coords_are_cartesian = False
            if None in (x, y, z):
                x = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_cartn_x"], None))
                y = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_cartn_y"], None))
                z = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_cartn_z"], None))
                coords_are_cartesian = True
            if None in (x, y, z):
                raise ValueError(f"CIF 原子 {label} 缺少三维坐标")
            occupancy = parse_cif_number(
                _get_row_value(loop, indices, i, ["_atom_site_occupancy"], 1.0), 1.0
            )
            if occupancy is None or occupancy < 0:
                raise ValueError(f"CIF 原子 {label} 的 occupancy 无效")
            charge = parse_cif_number(_get_row_value(loop, indices, i, ["_atom_site_charge"], None))
            props: dict[str, Any] = {"source_index": i}
            for key, names in {
                "disorder_group": ["_atom_site_disorder_group"],
                "disorder_assembly": ["_atom_site_disorder_assembly"],
                "alt_id": ["_atom_site_label_alt_id"],
                "multiplicity": ["_atom_site_symmetry_multiplicity"],
            }.items():
                val = _get_row_value(loop, indices, i, names, None)
                if val not in (None, "", ".", "?"):
                    props[key] = str(val).strip("'\"")
            atoms.append(AtomRecord(label, element, [x, y, z], occupancy, charge, props))
        if coords_are_cartesian and cell is not None:
            cart = np.asarray([a.coords for a in atoms], dtype=float)
            frac = cart @ np.linalg.inv(cell)
            for atom, xyz in zip(atoms, frac, strict=True):
                atom.coords = xyz
            coords_are_cartesian = False
        geom_bonds = _parse_geom_bonds(block)
        metadata: dict[str, Any] = {
            "cif_block": str(block.name),
            "cif_block_index": block_index,
            "cif_block_count": len(doc),
            "cif_block_names": [str(item.name) for item in doc],
            "cif_block_selection_reason": block_selection_reason,
            "declared_space_group": declared_space_group,
            "symmetry_operations": sym_ops,
            "symmetry_operations_source": symmetry_source,
            "cell_parameters": values,
            "raw_atom_count": len(atoms),
            "occupancy_present": "_atom_site_occupancy" in indices,
            "disorder_present": any("disorder" in k or "alt_id" in k for k in indices),
            "geom_bond_present": bool(geom_bonds),
            "geom_bonds": geom_bonds,
        }
        if space_group_details:
            metadata["space_group_number"] = space_group_details["number"]
            metadata["canonical_space_group"] = space_group_details["hm"]
            metadata["canonical_short_space_group"] = space_group_details["short_name"]
            metadata["space_group_hall"] = space_group_details["hall"]
            metadata["space_group_crystal_system"] = space_group_details["crystal_system"]
        model = StructureData(
            atoms=atoms,
            cell=cell,
            pbc=(True, True, True) if cell is not None else (False, False, False),
            coords_are_cartesian=coords_are_cartesian,
            format="cif",
            source_path=str(p),
            metadata=metadata,
        )
        if sym_ops and _symmetry_closed(model, sym_ops):
            model.metadata.update(
                {
                    "symmetry_closed": True,
                    "expansion_skipped": True,
                    "expansion_skip_reason": "input atom rows are closed under declared symmetry",
                }
            )
        return model
    except Exception as exc:
        return _parse_cif_with_pymatgen(p, exc)


def _parse_cif_with_pymatgen(path: Path, original_error: Exception | None = None) -> StructureData:
    try:
        from pymatgen.io.cif import CifParser

        # Pymatgen also defaults to the first data block.  When Gemmi can read
        # the document but a later conversion step fails, feed Pymatgen a
        # temporary copy of the selected crystallographic block instead of a
        # leading publCIF/audit block.  The source remains untouched.
        parse_path = path
        selected_block_name: str | None = None
        selected_index: int | None = None
        temporary_name: str | None = None
        try:
            import gemmi

            doc = gemmi.cif.read_file(str(path))
            block, selected_index, _ = _select_cif_block(doc)
            selected_block_name = str(block.name)
            if len(doc) > 1 or selected_index != 0:
                handle = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".cif", prefix="dsh-cif-block-", delete=False
                )
                temporary_name = handle.name
                try:
                    handle.write(block.as_string())
                finally:
                    handle.close()
                parse_path = Path(temporary_name)
        except Exception:
            # If Gemmi itself cannot read the malformed file, let Pymatgen try
            # the original path and preserve both errors in the final message.
            parse_path = path
        try:
            parser = CifParser(str(parse_path), occupancy_tolerance=1.0)
            structures = parser.parse_structures(primitive=False, check_occu=False)
            if not structures:
                raise ValueError("Pymatgen 未解析出结构")
            model = StructureData.from_pymatgen(structures[0], format="cif", source_path=str(path))
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        model.metadata["fallback_parser"] = "pymatgen"
        if selected_block_name is not None:
            model.metadata.update(
                {
                    "cif_block": selected_block_name,
                    "cif_block_index": selected_index,
                    "cif_block_count": len(doc),
                    "cif_block_names": [str(item.name) for item in doc],
                    "cif_block_selection_reason": "Pymatgen fallback selected Gemmi atom block",
                }
            )
        if original_error:
            model.metadata["gemmi_error"] = str(original_error)
        return model
    except Exception as exc:
        if original_error:
            raise ValueError(f"CIF 解析失败（Gemmi: {original_error}; Pymatgen: {exc}）") from exc
        raise


def parse_poscar(path: str | Path) -> StructureData:
    p = Path(path).expanduser().resolve()
    try:
        from pymatgen.io.vasp import Poscar

        try:
            poscar = Poscar.from_file(str(p), check_for_potcar=False)
        except TypeError:  # compatibility with older Pymatgen releases
            poscar = Poscar.from_file(str(p), check_for_POTCAR=False)
        model = StructureData.from_pymatgen(poscar.structure, format="poscar", source_path=str(p))
        model.metadata.update(
            {
                "comment": poscar.comment,
                "selective_dynamics": poscar.selective_dynamics,
                "velocities_present": poscar.velocities is not None,
                "coordinate_mode": "direct",
            }
        )
        # Preserve the conventional POSCAR species order in metadata.
        model.metadata["site_symbols"] = list(poscar.site_symbols)
        model.metadata["natoms"] = list(poscar.natoms)
        return model
    except Exception as exc:
        try:
            import ase.io

            atoms = ase.io.read(str(p), format="vasp")
            return from_ase(
                atoms, format="poscar", source_path=str(p), metadata={"fallback_parser": "ase"}
            )
        except Exception as ase_exc:
            raise ValueError(f"POSCAR 解析失败（Pymatgen: {exc}; ASE: {ase_exc}）") from ase_exc


def parse_xyz(path: str | Path, fmt: str = "xyz") -> StructureData:
    p = Path(path).expanduser().resolve()
    try:
        import ase.io

        atoms = ase.io.read(str(p), index=-1)
        return from_ase(atoms, format=fmt, source_path=str(p))
    except Exception as exc:
        raise ValueError(f"XYZ/extXYZ 解析失败: {exc}") from exc


def from_ase(
    atoms: Any,
    *,
    format: str,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StructureData:
    cell = (
        np.asarray(atoms.cell.array, dtype=float)
        if getattr(atoms, "cell", None) is not None
        else None
    )
    if cell is not None and abs(float(np.linalg.det(cell))) < 1e-12:
        cell = None
    symbols = list(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if cell is not None:
        frac = positions @ np.linalg.inv(cell)
        coords = frac
        cartesian = False
    else:
        coords = positions
        cartesian = True
    result = StructureData(
        atoms=[
            AtomRecord(f"{s}{i + 1}", s, xyz)
            for i, (s, xyz) in enumerate(zip(symbols, coords, strict=True))
        ],
        cell=cell,
        pbc=tuple(bool(x) for x in getattr(atoms, "pbc", (False, False, False))),
        coords_are_cartesian=cartesian,
        format=format,
        source_path=source_path,
        metadata=metadata or {},
    )
    if getattr(atoms, "arrays", None):
        for key, vals in atoms.arrays.items():
            if key in {"positions", "numbers"}:
                continue
            for atom, value in zip(result.atoms, vals, strict=True):
                try:
                    converted = value.tolist()
                except AttributeError:
                    converted = value
                if key == "occupancy":
                    try:
                        atom.occupancy = float(converted)
                    except (TypeError, ValueError):
                        atom.properties[key] = converted
                elif key in {"dsh_label", "_dsh_label"}:
                    atom.label = str(converted)
                else:
                    atom.properties[key] = converted
    return result


def parse_structure(path: str | Path, fmt: str | None = None) -> StructureData:
    actual = FORMAT_ALIASES.get((fmt or "").lower(), fmt) if fmt else detect_format(path)
    if actual == "cif":
        return parse_cif(path)
    if actual == "poscar":
        return parse_poscar(path)
    if actual in {"xyz", "extxyz"}:
        return parse_xyz(path, actual)
    if actual == "outcar":
        from .outcar import parse_outcar

        return parse_outcar(path).last_structure
    raise ValueError(f"无法识别结构文件格式: {path}")


def _periodic_close(
    frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray, tol_ang: float
) -> bool:
    delta = frac_a - frac_b
    # 27-image search is conservative for normal crystallographic cells and
    # avoids the orthogonal-cell assumption in a simple ``round`` operation.
    # np.ndindex gives 0..2; shift to -1..1.
    best = min(
        float(np.linalg.norm((delta + np.array(t) - 1.0) @ cell)) for t in np.ndindex(3, 3, 3)
    )
    return best <= tol_ang


def expand_symmetry(model: StructureData, *, tolerance_angstrom: float = 0.02) -> StructureData:
    """Expand CIF atom rows by declared symmetry and deduplicate special sites."""
    if model.cell is None or model.coords_are_cartesian:
        return model.copy()
    ops = list(model.metadata.get("symmetry_operations") or [])
    if not ops:
        symbol = model.metadata.get("declared_space_group")
        if symbol:
            try:
                from pymatgen.symmetry.groups import SpaceGroup

                for op in SpaceGroup(str(symbol)).symmetry_ops:
                    method = getattr(op, "as_xyz_string", None) or getattr(op, "as_xyz_str", None)
                    if method is not None:
                        ops.append(str(method()))
            except Exception:
                ops = []
        if not ops and symbol:
            try:
                import gemmi

                space_group = gemmi.find_spacegroup_by_name(str(symbol).strip("'\""))
                if space_group is not None:
                    ops = [str(op.triplet()) for op in space_group.operations().sym_ops]
            except Exception:
                ops = []
    if not ops:
        return model.copy()
    if _symmetry_closed(model, ops, tolerance_angstrom=tolerance_angstrom):
        # The source already contains a symmetry-closed set (often a viewer
        # export).  Keep every row and only record why another expansion was
        # deliberately skipped; deleting or multiplying these rows would
        # change the chemical model.
        out = model.copy()
        out.metadata.update(
            {
                "symmetry_closed": True,
                "expansion_skipped": True,
                "expansion_skip_reason": "input atom rows are closed under declared symmetry",
                "expanded_from_atom_count": model.n_atoms,
                "expanded_atom_count": model.n_atoms,
                "expansion_operation_count": len(ops),
            }
        )
        return out
    try:
        from pymatgen.core.operations import SymmOp
    except Exception:
        return model.copy()
    result_atoms: list[AtomRecord] = []
    for source_index, atom in enumerate(model.atoms):
        for op_index, op_text in enumerate(ops):
            try:
                parser = getattr(SymmOp, "from_xyz_string", None) or SymmOp.from_xyz_str
                frac = np.asarray(parser(op_text).operate(atom.coords), dtype=float)
            except Exception:
                continue
            frac = frac - np.floor(frac)
            props = dict(atom.properties)
            props.update(
                {
                    "source_index": source_index,
                    "symmetry_operation": op_text,
                    "symmetry_index": op_index,
                }
            )
            duplicate = False
            for existing in result_atoms:
                if existing.element != atom.element:
                    continue
                if abs(existing.occupancy - atom.occupancy) > 1e-6:
                    continue
                if existing.properties.get("disorder_group") != atom.properties.get(
                    "disorder_group"
                ):
                    continue
                if existing.properties.get("disorder_assembly") != atom.properties.get(
                    "disorder_assembly"
                ):
                    continue
                if existing.properties.get("alt_id") != atom.properties.get("alt_id"):
                    continue
                if _periodic_close(
                    existing.coords, frac, np.asarray(model.cell), tolerance_angstrom
                ):
                    duplicate = True
                    break
            if not duplicate:
                label = atom.label if op_index == 0 else f"{atom.label}_s{op_index + 1}"
                result_atoms.append(
                    AtomRecord(label, atom.element, frac, atom.occupancy, atom.charge, props)
                )
    # Global label uniqueness is required for broad viewer compatibility.
    used: dict[str, int] = {}
    for atom in result_atoms:
        count = used.get(atom.label, 0) + 1
        used[atom.label] = count
        if count > 1:
            atom.label = f"{atom.label}_{count}"
    out = StructureData(
        atoms=result_atoms,
        cell=model.cell.copy(),
        pbc=model.pbc,
        coords_are_cartesian=False,
        format=model.format,
        source_path=model.source_path,
        metadata=dict(model.metadata),
    )
    out.metadata.update(
        {
            "expanded_symmetry": True,
            "expansion_operation_count": len(ops),
            "expanded_from_atom_count": model.n_atoms,
            "expanded_atom_count": out.n_atoms,
        }
    )
    if out.n_atoms != model.n_atoms or not model.metadata.get("symmetry_closed"):
        # Source _geom_bond rows usually refer only to the asymmetric unit;
        # after expansion their labels/images are incomplete.  Require a
        # fresh bond_graph/bond_propose pass rather than carrying stale rows.
        out.metadata.pop("geom_bonds", None)
        out.metadata["geom_bond_present"] = False
        out.metadata["geom_bond_invalidated_reason"] = (
            "symmetry expansion changed the atom representation"
        )
    return out
