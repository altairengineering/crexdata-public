# FloodWaive exporter — CREXDATA support tool

Command-line tool to reproduce and validate FloodWaive exports for CREXDATA.

It can:

- request a Simulation GeoTIFF export (e.g., **maximum water levels** via `timesteps=[-1]`)
- poll the Export job until completion
- download the resulting file(s) using the filesystem download endpoint
- generate a `manifest.csv` / `manifest.json` for the `Outputs/` folder

## Requirements

- Python 3.9+ (no third-party dependencies)
- A FloodWaive API token with appropriate permissions

## Setup

```bash
export FLOODWAIVE_API_KEY="fw_example_token"
export SIMULATION_ID="simulation-550e8400-e29b-41d4-a716-446655440000"
```

## Example: export max water levels for a simulation

```bash
python3 floodwaive_exporter.py export-simulation \\
  --simulation-id "$SIMULATION_ID" \\
  --out-dir "../../Outputs/CREXDATA_Dortmund" \\
  --timesteps "-1" \\
  --repo-layout
```

## Example: build dataset manifest (for already-downloaded outputs)

```bash
python3 floodwaive_exporter.py build-manifest \\
  --outputs-root "../../Outputs" \\
  --manifest-csv "../../Outputs/manifest.csv" \\
  --manifest-json "../../Outputs/manifest.json"
```

## Authentication

The tool reads the API key from `FLOODWAIVE_API_KEY` (or `--api-key`).
