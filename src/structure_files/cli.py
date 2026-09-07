"""Small JSON CLI for smoke tests and reproducible local use."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any

from .service import StructureService
from .utils import dumps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dsh-structure-files JSON CLI")
    parser.add_argument(
        "action",
        choices=[
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
        ],
    )
    parser.add_argument("--json", dest="json_payload", help="JSON object with action arguments")
    parser.add_argument("--input-path")
    parser.add_argument("--path")
    parser.add_argument("--input-a")
    parser.add_argument("--input-b")
    parser.add_argument("--reactant-path")
    parser.add_argument("--product-path")
    parser.add_argument("--mapping")
    parser.add_argument("--ml-json")
    parser.add_argument("--n-frames", type=int, default=17)
    parser.add_argument("--cell-mode")
    parser.add_argument("--min-bond-change", type=float)
    parser.add_argument("--bond-tolerance", type=float)
    parser.add_argument("--metal-tolerance", type=float)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--no-coordination", action="store_true")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--output-format")
    parser.add_argument("--manifest-path")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--destination")
    parser.add_argument("--output-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", default="summary")
    parser.add_argument("--expand-symmetry", action="store_true")
    parser.add_argument("--include-atoms", action="store_true")
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    if args.json_payload:
        payload = json.loads(args.json_payload)
        if not isinstance(payload, dict):
            parser.error("--json 必须是 JSON 对象")
    for key in (
        "input_path",
        "path",
        "input_a",
        "input_b",
        "reactant_path",
        "product_path",
        "mapping",
        "ml_json",
        "output_format",
        "manifest_path",
        "destination",
        "output_path",
        "output_dir",
        "mode",
        "cell_mode",
        "min_bond_change",
        "bond_tolerance",
        "metal_tolerance",
        "min_confidence",
    ):
        value = getattr(args, key, None)
        if value is not None and key not in payload:
            payload[key] = value
    if args.action == "scan" and "recursive" not in payload:
        payload["recursive"] = args.recursive
    if args.action == "inspect":
        payload.setdefault("expand_symmetry", args.expand_symmetry)
        payload.setdefault("include_atoms", args.include_atoms)
    if args.action == "hydrogen_propose":
        payload.setdefault("top_k", args.top_k)
    if args.action == "bond_propose":
        payload.setdefault("include_coordination", not args.no_coordination)
    if args.action == "proposal_apply":
        payload.setdefault("candidate_index", args.candidate_index)
    if args.action == "ts_search":
        payload.setdefault("n_frames", args.n_frames)
        payload.setdefault("top_k", args.top_k)
        payload.setdefault("include_trajectory", not args.no_trajectory)
    service = StructureService()
    functions = {
        "scan": service.scan,
        "inspect": service.inspect,
        "validate": service.validate,
        "convert": service.convert,
        "transform": service.transform,
        "bond_graph": service.bond_graph,
        "bond_propose": service.bond_propose,
        "hydrogen_propose": service.hydrogen_propose,
        "proposal_apply": service.proposal_apply,
        "outcar_extract": service.outcar_extract,
        "compare": service.compare,
        "ts_search": service.ts_search,
    }
    # Keep convenience CLI defaults from leaking into unrelated service calls.
    allowed = set(inspect.signature(functions[args.action]).parameters)
    if args.action == "transform":
        allowed.update(
            {
                "output_path",
                "output_dir",
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
                "occupancy_policy",
            }
        )
    raw_ml_json = payload.get("ml_json")
    payload = {key: value for key, value in payload.items() if key in allowed}
    # The service API calls the optional calculator configuration ``ml`` while
    # the CLI exposes the more explicit ``--ml-json`` flag.  Decode it here so
    # ML-backed hydrogen/TS ranking is available from both front ends and a
    # malformed payload gets the same structured error as other actions.
    if raw_ml_json is not None and "ml" in allowed:
        try:
            decoded_ml = json.loads(raw_ml_json)
        except json.JSONDecodeError as exc:
            result = {
                "ok": False,
                "action": args.action,
                "error_type": type(exc).__name__,
                "error": f"ml-json 不是有效 JSON: {exc}",
            }
            sys.stdout.buffer.write(dumps(result, indent=True) + b"\n")
            return 1
        if not isinstance(decoded_ml, dict):
            result = {
                "ok": False,
                "action": args.action,
                "error_type": "ValueError",
                "error": "ml-json 必须是 JSON 对象",
            }
            sys.stdout.buffer.write(dumps(result, indent=True) + b"\n")
            return 1
        payload["ml"] = decoded_ml
    try:
        result = functions[args.action](**payload)
    except Exception as exc:
        result = {
            "ok": False,
            "action": args.action,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    sys.stdout.buffer.write(dumps(result, indent=True) + b"\n")
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
