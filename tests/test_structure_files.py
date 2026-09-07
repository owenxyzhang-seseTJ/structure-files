from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from structure_files.geometry import build_bond_graph, coordination_summary, minimum_image
from structure_files.mcp_server import server
from structure_files.ml import evaluate_models
from structure_files.models import AtomRecord, StructureData
from structure_files.operations import apply_basis_change, apply_supercell, transform_structure
from structure_files.outcar import parse_outcar
from structure_files.parsers import expand_symmetry, parse_cif, parse_poscar
from structure_files.proposals import apply_proposal
from structure_files.service import StructureService
from structure_files.ts_search import search_transition_state_candidates
from structure_files.writers import cif_text

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


def test_cif_and_poscar_parse_agree() -> None:
    cif = parse_cif(EXAMPLES / "ethanol_no_h.cif")
    poscar = parse_poscar(EXAMPLES / "ethanol_no_h.POSCAR")
    assert cif.n_atoms == poscar.n_atoms == 3
    assert cif.composition() == poscar.composition()
    assert expand_symmetry(cif).n_atoms == 3


def test_outcar_last_complete() -> None:
    result = parse_outcar(EXAMPLES / "OUTCAR.sample")
    assert result.nions == 2
    assert len(result.steps) == 1
    assert result.steps[-1].energy_sigma0_eV == pytest.approx(-10.12)
    assert result.last_structure.n_atoms == 2


def test_hydrogen_proposal_and_sha_gate(tmp_path: Path) -> None:
    service = StructureService()
    result = service.hydrogen_propose(
        str(EXAMPLES / "ethanol_no_h.cif"), output_dir=str(tmp_path), top_k=2
    )
    assert result["ok"] is True
    assert result["candidate_count"] == 2
    manifest = Path(result["proposal"]["manifest_path"])
    applied = service.proposal_apply(str(manifest), destination=str(tmp_path / "candidate.cif"))
    assert Path(applied["output_path"]).is_file()
    assert Path(applied["audit_path"]).is_file()
    # Source mutation invalidates a proposal instead of silently applying it.
    source = EXAMPLES / "ethanol_no_h.cif"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n# changed for test\n")
        with pytest.raises(RuntimeError):
            apply_proposal(manifest, destination=str(tmp_path / "stale.cif"))
    finally:
        source.write_bytes(original)


def test_mcp_tool_inventory() -> None:
    import asyncio

    async def collect():
        return await server.list_tools()

    names = {tool.name for tool in asyncio.run(collect())}
    assert names == {
        "scan",
        "inspect",
        "validate",
        "convert",
        "transform",
        "bond_graph",
        "bond_propose",
        "hydrogen_propose",
        "proposal_apply",
        "outcar_extract",
        "compare",
        "ts_search",
    }


def test_transform_creates_proposal(tmp_path: Path) -> None:
    result = StructureService().transform(
        str(EXAMPLES / "ethanol_no_h.cif"),
        "cif",
        output_dir=str(tmp_path),
        supercell="2 1 1",
        wrap=True,
    )
    assert result["ok"] is True
    assert result["preview"]["n_atoms"] == 6
    manifest = json.loads(Path(result["proposal"]["manifest_path"]).read_text())
    assert manifest["metadata"]["transform"]["supercell"] == [2.0, 1.0, 1.0]


def test_vasp_output_alias_uses_poscar_extension(tmp_path: Path) -> None:
    result = StructureService().convert(
        str(EXAMPLES / "ethanol_no_h.cif"), "vasp", output_dir=str(tmp_path), expand_p1=True
    )
    assert result["ok"] is True
    assert result["proposal"]["requested_output"].endswith(".vasp")


def test_ts_search_generates_ranked_candidates_and_trajectory(tmp_path: Path) -> None:
    service = StructureService()
    result = service.ts_search(
        str(EXAMPLES / "ethanol_no_h.cif"),
        str(EXAMPLES / "ethanol_no_h_product.cif"),
        output_dir=str(tmp_path),
        n_frames=7,
        top_k=2,
    )
    assert result["ok"] is True
    assert result["state"] == "candidate_frames"
    assert result["bond_changes"][0]["change"] == "broken"
    assert result["candidates"][0]["frame_index"] == 3
    manifest = result["proposal"]["manifest_path"]
    names = [row["name"] for row in result["proposal"]["artifacts"]]
    assert names[:2] == ["candidate-01.cif", "candidate-02.cif"]
    assert "ts-path.extxyz" in names
    assert "ts-search-report.json" in names
    applied = service.proposal_apply(manifest, destination=str(tmp_path / "ts.cif"))
    assert Path(applied["output_path"]).is_file()
    assert Path(applied["audit_path"]).is_file()


def test_space_group_symbol_without_operation_loop_expands(tmp_path: Path) -> None:
    source = tmp_path / "p21c.cif"
    source.write_text(
        """data_p21c
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
"""
    )
    model = parse_cif(source)
    assert model.metadata["symmetry_operations_source"] == "space_group_symbol"
    expanded = expand_symmetry(model)
    assert expanded.n_atoms == 4
    assert len(set(expanded.labels())) == expanded.n_atoms
    assert all(np.all((atom.coords >= 0) & (atom.coords < 1)) for atom in expanded.atoms)


def test_cif_writer_preserves_altloc_and_disorder_assembly(tmp_path: Path) -> None:
    source = tmp_path / "disorder.cif"
    source.write_text(
        """data_disorder
_cell_length_a 8
_cell_length_b 8
_cell_length_c 8
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
 _symmetry_equiv_pos_as_xyz
 'x,y,z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 _atom_site_disorder_group
 _atom_site_disorder_assembly
 _atom_site_label_alt_id
 C1A C 0.10 0.10 0.10 0.60 1 A A
 C1B C 0.12 0.10 0.10 0.40 2 A B
"""
    )
    model = parse_cif(source)
    assert model.metadata["disorder_present"] is True
    assert StructureService().inspect(str(source))["cif_representation"]["classification"] == (
        "disorder_or_partial_occupancy"
    )
    output = cif_text(model)
    assert "_atom_site_disorder_assembly" in output
    assert "_atom_site_label_alt_id" in output
    generated = tmp_path / "roundtrip.cif"
    generated.write_text(output)
    roundtrip = parse_cif(generated)
    assert [a.properties.get("alt_id") for a in roundtrip.atoms] == ["A", "B"]
    assert [a.properties.get("disorder_assembly") for a in roundtrip.atoms] == ["A", "A"]


def test_existing_geom_bonds_are_retained_on_cif_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source-bond.cif"
    source.write_text(
        """data_source_bond
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
 _symmetry_equiv_pos_as_xyz
 'x,y,z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.10 0.10 0.10
 O1 O 0.22 0.10 0.10
loop_
 _geom_bond_atom_site_label_1
 _geom_bond_atom_site_label_2
 _geom_bond_distance
 _geom_bond_type
 C1 O1 1.20 SING
""",
        encoding="utf-8",
    )
    model = parse_cif(source)
    assert model.metadata["geom_bond_present"] is True
    assert model.metadata["geom_bonds"][0]["distance_A"] == pytest.approx(1.2)
    output = cif_text(model)
    assert "_geom_bond_atom_site_label_1" in output
    assert "C1 O1 1.2 ? SING" in output


def test_cif_writer_suffixes_duplicate_labels_without_changing_source_model() -> None:
    model = StructureData(
        [
            AtomRecord("C1", "C", [0.1, 0.1, 0.1]),
            AtomRecord("C1", "C", [0.2, 0.1, 0.1]),
        ],
        cell=np.diag([10.0, 10.0, 10.0]),
    )
    text = cif_text(model)
    assert "C1 C 0.1 0.1 0.1" in text
    assert "C1_2 C 0.2 0.1 0.1" in text
    assert model.labels() == ["C1", "C1"]


def test_multiblock_cif_selects_crystallographic_atom_block(tmp_path: Path) -> None:
    source = tmp_path / "publcif.cif"
    source.write_text(
        """data_publication_text
_publ_section_title 'A text-only leading block'
_publ_author_name 'Example'

data_crystal
_cell_length_a 8
_cell_length_b 9
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.1 0.2 0.3
""",
        encoding="utf-8",
    )
    model = parse_cif(source)
    assert model.metadata["cif_block"] == "crystal"
    assert model.metadata["cif_block_index"] == 1
    assert model.metadata["cif_block_count"] == 2
    assert model.n_atoms == 1
    inspection = StructureService().inspect(str(source))
    assert inspection["cif_representation"]["declared_space_group"] == "P 1"


def test_bond_propose_writes_audited_geom_bond_loop(tmp_path: Path) -> None:
    source = tmp_path / "bond.cif"
    source.write_text(
        """data_bond
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
 _symmetry_equiv_pos_as_xyz
 'x,y,z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.10 0.10 0.10
 O1 O 0.22 0.10 0.10
""",
        encoding="utf-8",
    )
    result = StructureService().bond_propose(str(source), output_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["bond_count"] == 1
    assert result["validation"]["duplicate_contact_count"] == 0
    artifact = Path(result["proposal"]["stage_directory"]) / "candidate-1.cif"
    text = artifact.read_text(encoding="utf-8")
    assert "_geom_bond_atom_site_label_1" in text
    assert "_dsh_geom_bond_confidence" in text
    assert "? covalent" in text
    applied = StructureService().proposal_apply(
        result["proposal"]["manifest_path"], destination=str(tmp_path / "bond-out.cif")
    )
    assert Path(applied["output_path"]).is_file()
    roundtrip = parse_cif(applied["output_path"])
    assert roundtrip.n_atoms == 2


def test_bond_propose_rejects_unexpanded_asymmetric_unit(tmp_path: Path) -> None:
    source = tmp_path / "asymmetric.cif"
    source.write_text(
        """data_asymmetric
_cell_length_a 8
_cell_length_b 9
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.10 0.20 0.30
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="必须先展开对称性"):
        StructureService().bond_propose(
            str(source), expand_symmetry=False, output_dir=str(tmp_path)
        )


def test_cli_routes_bond_propose(tmp_path: Path) -> None:
    import subprocess

    out = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "structure_files",
            "bond_propose",
            "--input-path",
            str(EXAMPLES / "ethanol_no_h.cif"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))},
        check=False,
    )
    payload = json.loads(out.stdout)
    assert payload["ok"] is True
    assert payload["operation"] == "bond_propose"


def test_supercell_accepts_flat_matrix_and_makes_labels_unique() -> None:
    model = parse_cif(EXAMPLES / "ethanol_no_h.cif")
    transformed = apply_supercell(model, [1, 1, 0, 0, 2, 0, 0, 0, 1])
    assert transformed.n_atoms == 6
    assert len(set(transformed.labels())) == transformed.n_atoms
    assert transformed.metadata["supercell_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


def test_basis_change_preserves_cartesian_positions() -> None:
    model = parse_cif(EXAMPLES / "ethanol_no_h.cif")
    before = model.coords_cartesian.copy()
    transformed = apply_basis_change(
        model, [[1, 1, 0], [0, 1, 0], [0, 0, 1]], wrap=False
    )
    np.testing.assert_allclose(transformed.coords_cartesian, before, atol=1e-10)


def test_ts_search_explicit_mapping_no_cell_and_linear_cell_mode(tmp_path: Path) -> None:
    reactant_path = tmp_path / "reactant.xyz"
    product_path = tmp_path / "product.xyz"
    reactant_path.write_text("2\nreactant\nC 0 0 0\nO 3 0 0\n")
    product_path.write_text("2\nproduct\nO 0 0 0\nC 1 0 0\n")
    no_cell = StructureService().ts_search(
        str(reactant_path),
        str(product_path),
        output_dir=str(tmp_path),
        output_format="extxyz",
        mapping="0:1,1:0",
        n_frames=5,
        top_k=1,
        include_trajectory=False,
    )
    assert no_cell["ok"] is True
    assert no_cell["interpolation"]["periodic"] is False
    assert no_cell["mapping"]["mode"] == "user"
    assert no_cell["bond_changes"][0]["change"] == "formed"
    with pytest.raises(ValueError, match="无晶胞"):
        StructureService().ts_search(
            str(reactant_path), str(product_path), output_format="cif", n_frames=3
        )

    r_model = StructureData(
        [AtomRecord("C1", "C", [0.10, 0.20, 0.30])], cell=np.diag([10.0, 10.0, 10.0])
    )
    p_model = StructureData(
        [AtomRecord("C1", "C", [0.10, 0.20, 0.30])], cell=np.diag([20.0, 10.0, 10.0])
    )
    linear = search_transition_state_candidates(r_model, p_model, n_frames=3, cell_mode="linear")
    np.testing.assert_allclose(linear["frames"][1].cell[0, 0], 15.0)
    np.testing.assert_allclose(linear["unwrapped_cart"][1][0], [1.5, 2.0, 3.0])


def test_ml_factory_evaluates_all_frames_with_one_model() -> None:
    models = [
        StructureData(
            [AtomRecord("H1", "H", [float(x), 0.0, 0.0])],
            cell=np.diag([8.0, 8.0, 8.0]),
        )
        for x in (0.0, 1.0, 2.0)
    ]
    rows, audit = evaluate_models(
        models,
        {
            "enabled": True,
            "calculator_factory": "tests.fake_calculators:make_calculator",
        },
    )
    assert audit["status"] == "ok"
    assert audit["frame_count"] == 3
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["energy_eV"] < rows[-1]["energy_eV"]


def test_mace_requires_explicit_local_model_path() -> None:
    from structure_files.ml import _calculator_from_config

    with pytest.raises(ValueError, match="显式提供 model_path"):
        _calculator_from_config({"backend": "mace"})


def test_hydrogen_ml_hard_clash_is_rejected_and_original_is_retained(tmp_path: Path) -> None:
    result = StructureService().hydrogen_propose(
        str(EXAMPLES / "ethanol_no_h.cif"),
        output_dir=str(tmp_path),
        top_k=2,
        ml={
            "enabled": True,
            "calculator_factory": "tests.fake_calculators:make_calculator",
        },
    )
    assert result["state"] == "ambiguous"
    assert result["generated_candidate_count"] >= result["candidate_count"]
    assert result["ml_ranking"]["comparable_group_count"] == 0
    assert all(row["ml"]["status"] == "rejected_geometry" for row in result["candidates"])
    assert all(row["score"] < 100.0 for row in result["candidates"])


def test_cli_decodes_ml_json_for_ts_search(tmp_path: Path) -> None:
    import subprocess

    out = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "structure_files",
            "ts_search",
            "--reactant-path",
            str(EXAMPLES / "ethanol_no_h.cif"),
            "--product-path",
            str(EXAMPLES / "ethanol_no_h_product.cif"),
            "--output-dir",
            str(tmp_path),
            "--n-frames",
            "5",
            "--top-k",
            "1",
            "--ml-json",
            '{"enabled":true,"calculator_factory":"tests.fake_calculators:make_calculator"}',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))},
        check=False,
    )
    payload = json.loads(out.stdout)
    assert payload["ok"] is True
    assert payload["ml"]["status"] == "ok"


def test_cif_space_group_normalization_keeps_dual_parser_gate(tmp_path: Path) -> None:
    source = tmp_path / "p21c-label-only.cif"
    source.write_text(
        """data_p21c
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
"""
    )
    result = StructureService().validate(str(source))
    independent = result["independent_parsers"]
    assert independent["atom_count_agree"] is False
    assert independent["normalized_agreement"] is True
    assert independent["normalized"]["canonical_short_space_group"] == "P21/c"
    assert result["ok"] is True
    assert result["policy"]["parser_agreement_scope"] == "normalized symmetry representation"


def test_preexpanded_cif_is_not_multiplied_again(tmp_path: Path) -> None:
    source = tmp_path / "preexpanded.cif"
    source.write_text(
        """data_preexpanded
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1 21/c 1'
loop_
 _symmetry_equiv_pos_as_xyz
 x,y,z
 -x,y+1/2,-z+1/2
 -x,-y,-z
 x,-y+1/2,z+1/2
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
 C2 C 0.87 0.73 0.19
 C3 C 0.87 0.77 0.69
 C4 C 0.13 0.27 0.81
"""
    )
    model = parse_cif(source)
    assert model.metadata["symmetry_closed"] is True
    assert expand_symmetry(model).n_atoms == 4
    inspection = StructureService().inspect(str(source))
    assert inspection["cif_representation"]["classification"] == (
        "pre_expanded_or_symmetry_closed"
    )
    converted = StructureService().convert(str(source), "poscar", output_dir=str(tmp_path))
    assert converted["ok"] is True


def test_cross_boundary_bond_and_metal_coordination_are_reported() -> None:
    model = StructureData(
        [
            AtomRecord("C1", "C", [0.94, 0.5, 0.5]),
            AtomRecord("O1", "O", [0.06, 0.5, 0.5]),
            AtomRecord("Zn1", "Zn", [0.5, 0.5, 0.5]),
            AtomRecord("N1", "N", [0.7, 0.5, 0.5]),
        ],
        cell=np.diag([10.0, 10.0, 10.0]),
    )
    distance, image, _ = minimum_image(model, 0, 1)
    assert distance == pytest.approx(1.2)
    assert image.tolist() == [1, 0, 0]
    graph = build_bond_graph(model)
    assert graph.edges[0, 1]["kind"] == "covalent"
    assert any(
        row["label"] == "Zn1" and row["coordination_number"] == 1
        for row in coordination_summary(model, graph)
    )


def test_basis_change_transforms_symmetry_and_supercell_expands_asymmetric_unit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "p21c.cif"
    source.write_text(
        """data_p21c
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
"""
    )
    model = parse_cif(source)
    changed = apply_basis_change(model, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert changed.metadata["symmetry_operations_source"] == "basis_transformed"
    assert len(changed.metadata["symmetry_operations"]) == 4
    assert "2x+y" in changed.metadata["symmetry_operations"][1]
    supercell = apply_supercell(model, [2, 1, 1])
    assert supercell.n_atoms == 8
    assert supercell.metadata["supercell_symmetry_expanded_source"] is True
    assert supercell.metadata["declared_space_group"] == "P 1"


def test_origin_shift_conjugates_asymmetric_unit_symmetry(tmp_path: Path) -> None:
    source = tmp_path / "p21c-origin.cif"
    source.write_text(
        """data_p21c
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
""",
        encoding="utf-8",
    )
    model = parse_cif(source)
    shifted = transform_structure(model, origin_shift=[0.1, 0.0, 0.0])
    assert shifted.metadata["symmetry_operations_source"] == "origin_shift_conjugated"
    assert shifted.metadata["symmetry_operations"] != model.metadata["symmetry_operations"]
    expanded = expand_symmetry(shifted)
    assert expanded.n_atoms == 4


def test_selected_atom_translation_requires_p1_for_asymmetric_unit(tmp_path: Path) -> None:
    source = tmp_path / "p21c-atom-shift.cif"
    source.write_text(
        """data_p21c
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 101
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 C1 C 0.13 0.23 0.31
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="会破坏对称性"):
        StructureService().transform(
            str(source), atom_selection="0", atom_translation="0.1 0 0"
        )
