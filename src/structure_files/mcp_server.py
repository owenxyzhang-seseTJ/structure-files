"""stdio MCP server for DeepSeek Harness."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from .service import StructureService

server = MCPServer(
    name="structure-files",
    version="0.1.0",
    description=(
        "Chemistry-aware CIF/POSCAR/OUTCAR file processing; file operations only, "
        "no theory calculations."
    ),
    instructions=(
        "Only process structure files. Preserve source files. Any generated structure must go "
        "through a proposal and proposal_apply with SHA-256 verification. Preserve "
        "occupancies/disorder and label hydrogen positions as candidates unless chemistry makes "
        "them determinate. bond_propose writes only explicitly marked inferred bond annotations; "
        "it does not assert experimental bond orders."
    ),
)
service = StructureService()


def _error(action: str, exc: Exception) -> dict[str, Any]:
    return {"ok": False, "action": action, "error_type": type(exc).__name__, "error": str(exc)}


def _json_object(text: str | None) -> dict[str, Any] | None:
    if text is None or not str(text).strip():
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("ml_json 必须是 JSON 对象")
    return value


@server.tool(
    name="scan",
    description=(
        "扫描文件或目录，识别 CIF/POSCAR/CONTCAR/OUTCAR/XYZ，报告大小、SHA-256 和可解析性。只读。"
    ),
)
def scan(path: str, recursive: bool = True, max_files: int = 1000) -> dict[str, Any]:
    try:
        return service.scan(path, recursive=recursive, max_files=max_files)
    except Exception as exc:
        return _error("scan", exc)


@server.tool(
    name="inspect",
    description=(
        "读取单个结构文件的晶胞、坐标、组分、occupancy/disorder、声明对称性和 OUTCAR 摘要。只读。"
    ),
)
def inspect(
    path: str, expand_symmetry: bool = False, include_atoms: bool = False
) -> dict[str, Any]:
    try:
        return service.inspect(path, expand_symmetry=expand_symmetry, include_atoms=include_atoms)
    except Exception as exc:
        return _error("inspect", exc)


@server.tool(
    name="validate",
    description=(
        "双解析和化学感知验证：Pymatgen/ASE、周期最短接触、重复位点、启发式键图、"
        "价态/金属配位和 Spglib 交叉检查。只读。"
    ),
)
def validate(
    path: str, expand_symmetry: bool = True, contact_limit: int = 30, symprec: float = 0.1
) -> dict[str, Any]:
    try:
        return service.validate(
            path, expand_symmetry=expand_symmetry, contact_limit=contact_limit, symprec=symprec
        )
    except Exception as exc:
        return _error("validate", exc)


@server.tool(
    name="convert",
    description=(
        "提出 CIF/POSCAR/extXYZ 互转 proposal；不覆盖源文件。请随后用 proposal_apply "
        "选择并写入新文件。POSCAR 默认拒绝部分 occupancy。"
    ),
)
def convert(
    input_path: str,
    output_format: str,
    output_path: str | None = None,
    output_dir: str | None = None,
    expand_p1: bool = False,
    occupancy_policy: str = "reject",
) -> dict[str, Any]:
    try:
        return service.convert(
            input_path,
            output_format,
            output_path=output_path,
            output_dir=output_dir,
            expand_p1=expand_p1,
            occupancy_policy=occupancy_policy,
        )
    except Exception as exc:
        return _error("convert", exc)


@server.tool(
    name="transform",
    description=(
        "提出结构变换 proposal：P1 对称展开、整数超胞、基矢矩阵、原点/原子平移、"
        "连通片段平移、wrap/unwrap 和组分提取。复杂参数用空格/逗号字符串或 JSON 字符串。"
    ),
)
def transform(
    input_path: str,
    output_format: str = "cif",
    output_path: str | None = None,
    output_dir: str | None = None,
    p1: bool = False,
    supercell: str | None = None,
    basis: str | None = None,
    origin_shift: str | None = None,
    atom_selection: str | None = None,
    atom_translation: str | None = None,
    translation_coordinate_type: str = "fractional",
    connected_fragment: bool = False,
    wrap: bool = False,
    unwrap: bool = False,
    extract: str | None = None,
    occupancy_policy: str = "reject",
) -> dict[str, Any]:
    try:
        return service.transform(
            input_path,
            output_format,
            output_path=output_path,
            output_dir=output_dir,
            p1=p1,
            supercell=supercell,
            basis=basis,
            origin_shift=origin_shift,
            atom_selection=atom_selection,
            atom_translation=atom_translation,
            translation_coordinate_type=translation_coordinate_type,
            connected_fragment=connected_fragment,
            wrap=wrap,
            unwrap=unwrap,
            extract=extract,
            occupancy_policy=occupancy_policy,
        )
    except Exception as exc:
        return _error("transform", exc)


@server.tool(
    name="bond_graph",
    description=(
        "构造周期化学键/金属配位候选图，返回边距离、周期 image、启发式 bond order 和"
        "置信度。只读，不把启发式当成最终化学结论。"
    ),
)
def bond_graph(
    path: str, expand_symmetry: bool = False, tolerance: float = 0.45, include_hydrogen: bool = True
) -> dict[str, Any]:
    try:
        return service.bond_graph(
            path,
            expand_symmetry=expand_symmetry,
            tolerance=tolerance,
            include_hydrogen=include_hydrogen,
        )
    except Exception as exc:
        return _error("bond_graph", exc)


@server.tool(
    name="bond_propose",
    description=(
        "提出带 _geom_bond 的 CIF proposal。边由周期共价半径/金属配位启发式推断，"
        "标准 bond type 保持 '?'，并附带 DSH kind/order/confidence/image 审计字段；"
        "不会覆盖源文件，需随后调用 proposal_apply。"
    ),
)
def bond_propose(
    input_path: str,
    output_path: str | None = None,
    output_dir: str | None = None,
    expand_symmetry: bool = True,
    tolerance: float = 0.45,
    metal_tolerance: float = 0.65,
    include_hydrogen: bool = True,
    include_coordination: bool = True,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    try:
        return service.bond_propose(
            input_path,
            output_path=output_path,
            output_dir=output_dir,
            expand_symmetry=expand_symmetry,
            tolerance=tolerance,
            metal_tolerance=metal_tolerance,
            include_hydrogen=include_hydrogen,
            include_coordination=include_coordination,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        return _error("bond_propose", exc)


@server.tool(
    name="hydrogen_propose",
    description=(
        "基于周期键图/价态枚举补氢和多取向，输出 top-k CIF 候选 proposal、化学状态和"
        "不确定性。普通 C-H 可确定；羧酸 O-H、配位水、μ-OH、缺陷和 N/S 质子化保留"
        "多候选。可选 ml_json 仅用于同模型内 H-only 优化/排序，不能声称唯一实验位置。"
    ),
)
def hydrogen_propose(
    input_path: str,
    output_path: str | None = None,
    output_dir: str | None = None,
    top_k: int = 8,
    max_candidates: int = 128,
    orientation_count: int = 12,
    hetero_policy: str = "enumerate",
    include_unprotonated_hetero: bool = True,
    expand_symmetry: bool = True,
    ml_json: str | None = None,
) -> dict[str, Any]:
    try:
        return service.hydrogen_propose(
            input_path,
            output_path=output_path,
            output_dir=output_dir,
            top_k=top_k,
            max_candidates=max_candidates,
            orientation_count=orientation_count,
            hetero_policy=hetero_policy,
            include_unprotonated_hetero=include_unprotonated_hetero,
            expand_symmetry=expand_symmetry,
            ml=_json_object(ml_json),
        )
    except Exception as exc:
        return _error("hydrogen_propose", exc)


@server.tool(
    name="proposal_apply",
    description=(
        "在核对 proposal 中所有源文件 SHA-256 后，应用指定候选到新的输出路径并生成"
        " .audit.json；默认拒绝覆盖任何现有文件。"
    ),
)
def proposal_apply(
    manifest_path: str,
    candidate_index: int = 0,
    destination: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    try:
        return service.proposal_apply(
            manifest_path,
            candidate_index=candidate_index,
            destination=destination,
            overwrite=overwrite,
        )
    except Exception as exc:
        return _error("proposal_apply", exc)


@server.tool(
    name="ts_search",
    description=(
        "文件级 TS 候选搜索：对反应物/产物结构做原子映射、周期安全插值、端点键变化识别和"
        "内部帧排序，可选同一 ASE/MACE 模型做单点能量/力排序。只生成 TS-like 候选和"
        "proposal，不运行鞍点优化、NEB、VASP、DFT 或提交任务。"
    ),
)
def ts_search(
    reactant_path: str,
    product_path: str,
    output_format: str = "cif",
    output_path: str | None = None,
    output_dir: str | None = None,
    n_frames: int = 17,
    top_k: int = 3,
    mapping: str | None = None,
    cell_mode: str = "require_equal",
    cell_tolerance: float = 1e-5,
    bond_tolerance: float = 0.45,
    min_bond_change: float = 0.30,
    include_hydrogen: bool = True,
    include_trajectory: bool = True,
    ml_json: str | None = None,
) -> dict[str, Any]:
    try:
        return service.ts_search(
            reactant_path,
            product_path,
            output_format=output_format,
            output_path=output_path,
            output_dir=output_dir,
            n_frames=n_frames,
            top_k=top_k,
            mapping=mapping,
            cell_mode=cell_mode,
            cell_tolerance=cell_tolerance,
            bond_tolerance=bond_tolerance,
            min_bond_change=min_bond_change,
            include_hydrogen=include_hydrogen,
            include_trajectory=include_trajectory,
            ml=_json_object(ml_json),
        )
    except Exception as exc:
        return _error("ts_search", exc)


@server.tool(
    name="outcar_extract",
    description=(
        "只读解析 OUTCAR：summary 或最后完整离子步/完整轨迹。截断末步只报告并跳过，"
        "不修复原 OUTCAR；生成 POSCAR/CIF/extXYZ 时仍先建立 proposal。"
    ),
)
def outcar_extract(
    input_path: str,
    mode: str = "summary",
    output_format: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    try:
        return service.outcar_extract(
            input_path,
            mode=mode,
            output_format=output_format,
            output_path=output_path,
            output_dir=output_dir,
        )
    except Exception as exc:
        return _error("outcar_extract", exc)


@server.tool(
    name="compare",
    description=(
        "比较两个结构的组成、原子数、晶胞、Pymatgen StructureMatcher 映射和周期结构相似性。只读。"
    ),
)
def compare(
    input_a: str, input_b: str, ignore_hydrogen: bool = False, primitive: bool = False
) -> dict[str, Any]:
    try:
        return service.compare(
            input_a, input_b, ignore_hydrogen=ignore_hydrogen, primitive=primitive
        )
    except Exception as exc:
        return _error("compare", exc)


def main() -> None:
    # MCP owns stdout. Keep diagnostics on stderr so a log line can never
    # corrupt the stdio JSON-RPC stream.
    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        return
    except Exception as exc:  # pragma: no cover - transport-level failure
        print(f"structure-files MCP stopped: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
