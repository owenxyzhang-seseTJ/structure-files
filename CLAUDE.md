# Structure-files workflow for Claude Code

Use the project-scoped `structure-files` MCP server from `.mcp.json` for all
structure-file work. If MCP is unavailable, use `scripts/run_cli.sh` and its
JSON interface. Follow `AGENTS.md` exactly: inspect and validate first, preserve
source files, make every write proposal-only until `proposal_apply`, and keep
chemical/ML/TS claims at candidate level.

This project handles files only. Do not launch VASP, DFT, RASPA, NEB, Sella,
saddle-point optimization, job submission or monitoring from this repository.
