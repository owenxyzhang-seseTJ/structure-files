# Structure-files agent instructions

This repository is a chemistry-aware, file-only structure service. Use the
`structure-files` MCP server (or `scripts/run_cli.sh` when MCP is unavailable)
for CIF, POSCAR/CONTCAR, OUTCAR, XYZ and extXYZ work.

## Required workflow

1. Start with `scan` or `inspect`, then run `validate` before interpreting or
   transforming a structure.
2. Preserve the original file and its SHA-256. Never overwrite a source file.
3. For any generated structure, call `convert`, `transform`, `bond_propose`,
   `hydrogen_propose` or `outcar_extract` first. Apply a selected artifact only
   with `proposal_apply` after checking its manifest and audit data.
4. Treat inferred bonds, hydrogen positions and TS-search frames as candidates.
   Do not describe them as experimental bond orders, unique proton locations or
   confirmed transition states.

## Chemistry boundaries

- Preserve CIF occupancies, disorder, alternate locations, guests and periodic
  connectivity. Do not silently order disorder or set occupancy to one.
- Use P1 expansion only after classifying asymmetric-unit, pre-expanded and
  display-image representations. Re-run the bond graph after topology-changing
  operations.
- `hydrogen_propose` enumerates protonation/orientation candidates. ML may be
  enabled only with an explicit local calculator/model and may move H atoms only;
  heavy atoms and the cell remain fixed.
- `ts_search` performs periodic-safe interpolation and same-model single-point
  ranking only. It does not run VASP, DFT, NEB, Sella, saddle optimization,
  submission or monitoring.
- OUTCAR is read-only; a truncated last ionic step is reported and skipped.

Use the proposal manifest and `.audit.json` sidecar as the provenance record.
