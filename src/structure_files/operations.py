"""Structure transformations used by the proposal service."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from .geometry import build_bond_graph
from .models import AtomRecord, StructureData
from .parsers import expand_symmetry


def _invalidate_geom_bonds(model: StructureData, reason: str) -> None:
    """Drop source bond annotations when a coordinate/topology transform changes rows."""

    model.metadata.pop("geom_bonds", None)
    model.metadata["geom_bond_present"] = False
    model.metadata["geom_bond_invalidated_reason"] = reason


def parse_matrix(value: Any, *, integer: bool = False) -> np.ndarray:
    if value is None:
        raise ValueError("缺少矩阵")
    if isinstance(value, str):
        text = value.replace(";", " ").replace(",", " ")
        vals = [float(x) for x in text.split()]
        if len(vals) == 3:
            matrix = np.diag(vals)
        elif len(vals) == 9:
            matrix = np.asarray(vals, dtype=float).reshape(3, 3)
        else:
            raise ValueError("矩阵需要 3 个对角元素或 9 个元素")
    else:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape == (3,):
            matrix = np.diag(matrix)
        elif matrix.shape == (9,):
            matrix = matrix.reshape(3, 3)
        if matrix.shape != (3, 3):
            raise ValueError("矩阵需要形状 (3,3)")
    if integer and not np.allclose(matrix, np.rint(matrix), atol=1e-8):
        raise ValueError("超胞/基矢矩阵必须是整数矩阵")
    return np.rint(matrix).astype(int) if integer else matrix


def _manual_diagonal_supercell(model: StructureData, matrix: np.ndarray) -> StructureData:
    if model.cell is None or model.coords_are_cartesian:
        raise ValueError("超胞需要有分数坐标的周期晶胞")
    diag = np.diag(matrix).astype(int)
    if not np.allclose(matrix, np.diag(diag)) or np.any(diag <= 0):
        raise ValueError("手工超胞只接受正对角矩阵")
    new_atoms: list[AtomRecord] = []
    for atom in model.atoms:
        for shift in itertools.product(*(range(int(n)) for n in diag)):
            frac = (atom.coords + np.asarray(shift, dtype=float)) / diag
            suffix = "_" + "_".join(str(x) for x in shift)
            props = deepcopy(atom.properties)
            props["supercell_image"] = list(shift)
            new_atoms.append(
                AtomRecord(
                    f"{atom.label}{suffix}", atom.element, frac, atom.occupancy, atom.charge, props
                )
            )
    out = StructureData(
        atoms=new_atoms,
        cell=np.asarray(matrix, dtype=float) @ np.asarray(model.cell, dtype=float),
        pbc=model.pbc,
        coords_are_cartesian=False,
        format=model.format,
        source_path=model.source_path,
        metadata=deepcopy(model.metadata),
    )
    out.metadata.update(
        {
            "supercell_matrix": matrix.tolist(),
            "supercell_determinant": int(round(float(np.linalg.det(matrix)))),
            "supercell_from_atom_count": model.n_atoms,
        }
    )
    _invalidate_geom_bonds(out, "supercell changed atom rows; recompute periodic bonds")
    return out


def apply_supercell(model: StructureData, matrix: Any) -> StructureData:
    matrix_arr = parse_matrix(matrix, integer=True)
    determinant = int(round(float(np.linalg.det(matrix_arr))))
    if determinant == 0:
        raise ValueError("超胞矩阵不可逆（行列式不能为 0）")
    source_atom_count = model.n_atoms
    expanded_source = False
    # A supercell contains every site in the input cell.  If the input is an
    # asymmetric unit, expand it first; otherwise retaining the original
    # space-group loop would make a later CIF reader multiply the supercell a
    # second time.  Symmetry-closed/pre-expanded inputs are already complete.
    if (
        model.metadata.get("symmetry_operations")
        and not model.metadata.get("expanded_symmetry")
        and not model.metadata.get("symmetry_closed")
    ):
        expanded = expand_symmetry(model)
        if expanded.n_atoms != model.n_atoms:
            model = expanded
            expanded_source = True
    try:
        out = _manual_diagonal_supercell(model, matrix_arr)
    except ValueError:
        if model.cell is None or model.coords_are_cartesian:
            raise
        # Pymatgen handles non-diagonal integer matrices and the associated
        # lattice-point enumeration.  Convert back while retaining labels and
        # site properties as far as the parser can represent them.
        pmg = model.to_pymatgen()
        pmg.make_supercell(matrix_arr)
        out = StructureData.from_pymatgen(pmg, format=model.format, source_path=model.source_path)
        out.metadata.update(deepcopy(model.metadata))
        # Pymatgen correctly enumerates lattice points but may duplicate a
        # site-property label.  CIF/POSCAR consumers need globally unique
        # labels, so make the disambiguation deterministic and retain the
        # original label as an audit property.
        label_map: list[dict[str, str]] = []
        used: dict[str, int] = {}
        for index, atom in enumerate(out.atoms):
            original = atom.label or f"{atom.element}{index + 1}"
            occurrence = used.get(original, 0) + 1
            used[original] = occurrence
            label = original if occurrence == 1 else f"{original}_sc{occurrence}"
            while label in {row["output_label"] for row in label_map}:
                occurrence += 1
                used[original] = occurrence
                label = f"{original}_sc{occurrence}"
            atom.label = label
            if occurrence > 1:
                atom.properties["supercell_source_label"] = original
            label_map.append({"output_label": label, "source_label": original})
        out.metadata.update(
            {
                "supercell_matrix": matrix_arr.tolist(),
                "supercell_determinant": determinant,
                "supercell_from_atom_count": model.n_atoms,
                "supercell_label_map": label_map,
            }
        )
    # Replication creates a full atom list in the new cell.  Emit identity-only
    # P1 metadata so the generated CIF is unambiguous and idempotent on read.
    out.metadata.update(
        {
            "declared_space_group": "P 1",
            "canonical_space_group": "P 1",
            "canonical_short_space_group": "P1",
            "space_group_number": 1,
            "symmetry_operations": ["x,y,z"],
            "symmetry_operations_source": "supercell_full_atom_list",
            "supercell_input_atom_count": source_atom_count,
            "supercell_symmetry_expanded_source": expanded_source,
        }
    )
    _invalidate_geom_bonds(out, "supercell changed atom rows; recompute periodic bonds")
    return out


def _transform_symmetry_operations(operations: list[str], transform: np.ndarray) -> list[str]:
    """Conjugate fractional symmetry operations into a new unimodular basis."""

    if not operations:
        return []
    try:
        from pymatgen.core.operations import SymmOp
    except Exception:
        return []
    parser = getattr(SymmOp, "from_xyz_string", None) or getattr(SymmOp, "from_xyz_str", None)
    formatter = getattr(SymmOp, "as_xyz_string", None) or getattr(SymmOp, "as_xyz_str", None)
    constructor = getattr(SymmOp, "from_rotation_and_translation", None)
    if parser is None or formatter is None or constructor is None:
        return []
    inv_transform = np.linalg.inv(np.asarray(transform, dtype=float))
    result: list[str] = []
    fingerprints: set[tuple[float, ...]] = set()
    for text in operations:
        try:
            operation = parser(str(text))
            rotation = np.asarray(operation.rotation_matrix, dtype=float)
            translation = np.asarray(operation.translation_vector, dtype=float)
            # Coordinates are represented as row fractional vectors in this
            # project, whereas SymmOp stores column-vector rotations.  The
            # conjugation below follows f_new = f_old @ inv(T).
            new_rotation_col = inv_transform.T @ rotation @ np.asarray(transform, dtype=float).T
            new_translation = translation @ inv_transform
            new_rotation_col[np.abs(new_rotation_col) < 1e-10] = 0.0
            new_translation[np.abs(new_translation) < 1e-10] = 0.0
            new_translation = new_translation - np.floor(new_translation)
            new_rotation_col = np.where(
                np.isclose(new_rotation_col, np.rint(new_rotation_col), atol=1e-8),
                np.rint(new_rotation_col),
                new_rotation_col,
            )
            new_operation = constructor(new_rotation_col, new_translation)
            new_text = str(formatter(new_operation))
            fingerprint = tuple(
                np.round(
                    np.concatenate((new_rotation_col.ravel(), new_translation)),
                    decimals=8,
                )
            )
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            result.append(new_text)
        except Exception:
            return []
    return result


def apply_basis_change(model: StructureData, matrix: Any, *, wrap: bool = True) -> StructureData:
    if model.cell is None or model.coords_are_cartesian:
        raise ValueError("基矢变换需要有分数坐标的周期晶胞")
    transform = parse_matrix(matrix, integer=True)
    determinant = int(round(float(np.linalg.det(transform))))
    if abs(determinant) != 1:
        raise ValueError(
            "基矢变换矩阵必须是 unimodular（行列式为 ±1）；"
            "改变晶胞体积请使用 supercell"
        )
    out = model.copy()
    old_frac = model.frac_coords.copy()
    out.cell = transform @ np.asarray(model.cell, dtype=float)
    new_frac = old_frac @ np.linalg.inv(transform)
    out.set_frac_coords(new_frac)
    if wrap:
        out.set_frac_coords(out.frac_coords - np.floor(out.frac_coords))
    old_ops = list(model.metadata.get("symmetry_operations") or [])
    if model.metadata.get("expanded_symmetry") or model.metadata.get("symmetry_closed"):
        # A full/pre-expanded atom set no longer needs the source space-group
        # operators.  Emitting identity-only P1 prevents a later CIF reader
        # from multiplying the same rows a second time.
        out.metadata.update(
            {
                "declared_space_group": "P 1",
                "canonical_space_group": "P 1",
                "canonical_short_space_group": "P1",
                "space_group_number": 1,
                "symmetry_operations": ["x,y,z"],
                "symmetry_operations_source": "basis_change_from_symmetry_closed_input",
            }
        )
        transformed_ops = ["x,y,z"]
    else:
        transformed_ops = _transform_symmetry_operations(old_ops, transform)
        if old_ops and not transformed_ops:
            raise ValueError("无法把 CIF 对称操作转换到新基矢；请先 P1 展开后再变换")
        if transformed_ops:
            out.metadata["symmetry_operations"] = transformed_ops
            out.metadata["symmetry_operations_source"] = "basis_transformed"
            out.metadata["basis_change_original_symmetry_operations"] = old_ops
    out.metadata.update(
        {
            "basis_change_matrix": transform.tolist(),
            "basis_change_determinant": determinant,
            "basis_change_transformed_symmetry_operations": transformed_ops,
        }
    )
    _invalidate_geom_bonds(out, "basis change altered the periodic image convention")
    return out


def _selection_indices(model: StructureData, selection: Any) -> list[int]:
    if selection is None or selection == "all":
        return list(range(model.n_atoms))
    if isinstance(selection, str):
        text = selection.strip()
        if text.startswith("labels:"):
            labels = {x.strip() for x in text[7:].split(",") if x.strip()}
            return [i for i, atom in enumerate(model.atoms) if atom.label in labels]
        if text.startswith("elements:"):
            elements = {x.strip().capitalize() for x in text[9:].split(",") if x.strip()}
            return [i for i, atom in enumerate(model.atoms) if atom.element in elements]
        selection = [int(x.strip()) for x in text.split(",") if x.strip()]
    vals = [int(x) for x in selection]
    if any(x < 0 or x >= model.n_atoms for x in vals):
        raise IndexError("selection 中存在越界原子索引")
    return sorted(set(vals))


def translate_atoms(
    model: StructureData,
    selection: Any,
    vector: Iterable[float],
    *,
    coordinate_type: str = "fractional",
    connected_fragment: bool = False,
) -> StructureData:
    out = model.copy()
    selected = set(_selection_indices(out, selection))
    if connected_fragment:
        graph = build_bond_graph(out)
        expanded: set[int] = set()
        for component in __import__("networkx").connected_components(graph):
            if selected.intersection(component):
                expanded.update(component)
        selected = expanded
    vec = np.asarray(list(vector), dtype=float).reshape(3)
    if coordinate_type.lower() in {"cart", "cartesian", "angstrom", "a"}:
        if out.cell is None:
            delta = vec
            cart = out.coords_cartesian.copy()
            cart[list(selected)] += delta
            out.set_cart_coords(cart)
        else:
            vec = vec @ np.linalg.inv(np.asarray(out.cell))
            frac = out.frac_coords.copy()
            frac[list(selected)] += vec
            out.set_frac_coords(frac)
    else:
        frac = out.frac_coords.copy()
        frac[list(selected)] += vec
        out.set_frac_coords(frac)
    out.metadata["translated_indices"] = sorted(selected)
    out.metadata["translation_vector"] = [float(x) for x in np.asarray(vector, dtype=float)]
    out.metadata["translation_coordinate_type"] = coordinate_type
    out.metadata["translation_connected_fragment"] = bool(connected_fragment)
    _invalidate_geom_bonds(out, "atom translation may change bond distances")
    return out


def translate_origin(model: StructureData, shift: Iterable[float]) -> StructureData:
    """Translate every atom and conjugate non-P1 symmetry operations.

    A uniform fractional origin shift is not just a coordinate edit for an
    asymmetric unit: the affine translations in each symmetry operation must
    change as ``t' = s + t - R s``.  Full/P1 atom lists are emitted as P1 by
    policy, while a genuine asymmetric-unit CIF retains its declared group
    with the conjugated operations.
    """

    vec = np.asarray(list(shift), dtype=float).reshape(3)
    out = model.copy()
    out.set_frac_coords(model.frac_coords + vec)
    old_ops = list(model.metadata.get("symmetry_operations") or [])
    if old_ops and not model.metadata.get("expanded_symmetry") and not model.metadata.get(
        "symmetry_closed"
    ):
        try:
            from pymatgen.core.operations import SymmOp

            parser = getattr(SymmOp, "from_xyz_string", None) or SymmOp.from_xyz_str
            formatter = getattr(SymmOp, "as_xyz_string", None) or SymmOp.as_xyz_str
            constructor = SymmOp.from_rotation_and_translation
            transformed: list[str] = []
            for text in old_ops:
                operation = parser(str(text))
                rotation = np.asarray(operation.rotation_matrix, dtype=float)
                translation = np.asarray(operation.translation_vector, dtype=float)
                new_translation = vec + translation - rotation @ vec
                new_translation = new_translation - np.floor(new_translation)
                transformed.append(str(formatter(constructor(rotation, new_translation))))
            out.metadata["symmetry_operations"] = transformed
            out.metadata["symmetry_operations_source"] = "origin_shift_conjugated"
            out.metadata["origin_shift_original_symmetry_operations"] = old_ops
        except Exception as exc:
            raise ValueError(
                "无法把 origin_shift 应用于 CIF 对称操作；请先 p1=True 展开后再平移"
            ) from exc
    elif old_ops:
        out.metadata.update(
            {
                "declared_space_group": "P 1",
                "canonical_space_group": "P 1",
                "canonical_short_space_group": "P1",
                "space_group_number": 1,
                "symmetry_operations": ["x,y,z"],
                "symmetry_operations_source": "origin_shift_from_full_atom_list",
            }
        )
    out.metadata["origin_shift"] = [float(x) for x in vec]
    _invalidate_geom_bonds(out, "origin shift changed the coordinate representation")
    return out


def unwrap_structure(model: StructureData) -> StructureData:
    if model.cell is None or model.coords_are_cartesian:
        return model.copy()
    graph = build_bond_graph(model)
    frac = model.frac_coords.copy()
    unwrapped = np.full_like(frac, np.nan)
    visited: set[int] = set()
    for root in range(model.n_atoms):
        if root in visited:
            continue
        unwrapped[root] = frac[root]
        visited.add(root)
        queue = [root]
        while queue:
            i = queue.pop(0)
            for j in graph.neighbors(i):
                data = graph.edges[i, j]
                image = np.asarray(data.get("image", [0, 0, 0]), dtype=float)
                if j not in visited:
                    unwrapped[j] = unwrapped[i] + frac[j] + image - frac[i]
                    visited.add(j)
                    queue.append(j)
                elif i not in visited:
                    unwrapped[i] = unwrapped[j] + frac[i] - image - frac[j]
    out = model.copy()
    out.set_frac_coords(unwrapped)
    out.metadata["unwrapped"] = True
    _invalidate_geom_bonds(out, "unwrap changed periodic image coordinates")
    return out


def wrap_structure(model: StructureData) -> StructureData:
    out = model.copy()
    if out.cell is not None and not out.coords_are_cartesian:
        out.set_frac_coords(out.frac_coords - np.floor(out.frac_coords))
        out.metadata["wrapped"] = True
    return out


def extract_components(model: StructureData, components: Any = None) -> StructureData:
    graph = build_bond_graph(model)
    groups = [sorted(component) for component in __import__("networkx").connected_components(graph)]
    if components is None:
        raise ValueError("extract_components 需要 components 索引（例如 [0,2]）")
    wanted = {int(x) for x in components}
    if any(x < 0 or x >= len(groups) for x in wanted):
        raise IndexError("components 编号越界")
    indices = sorted(
        i for group_index, group in enumerate(groups) if group_index in wanted for i in group
    )
    out = model.copy()
    out.atoms = [out.atoms[i] for i in indices]
    out.metadata["extracted_component_indices"] = sorted(wanted)
    out.metadata["component_map"] = groups
    _invalidate_geom_bonds(out, "component extraction changed the atom set")
    return out


def transform_structure(
    model: StructureData,
    *,
    p1: bool = False,
    supercell: Any = None,
    basis: Any = None,
    origin_shift: Any = None,
    atom_selection: Any = None,
    atom_translation: Any = None,
    translation_coordinate_type: str = "fractional",
    connected_fragment: bool = False,
    wrap: bool = False,
    unwrap: bool = False,
    extract: Any = None,
) -> StructureData:
    out = expand_symmetry(model) if p1 else model.copy()
    if supercell is not None:
        out = apply_supercell(out, supercell)
    if basis is not None:
        out = apply_basis_change(out, basis, wrap=False)
    if origin_shift is not None:
        out = translate_origin(out, origin_shift)
    if atom_translation is not None:
        out = translate_atoms(
            out,
            atom_selection,
            atom_translation,
            coordinate_type=translation_coordinate_type,
            connected_fragment=connected_fragment,
        )
    if extract is not None:
        out = extract_components(out, extract)
    if unwrap:
        out = unwrap_structure(out)
    if wrap:
        out = wrap_structure(out)
    return out
