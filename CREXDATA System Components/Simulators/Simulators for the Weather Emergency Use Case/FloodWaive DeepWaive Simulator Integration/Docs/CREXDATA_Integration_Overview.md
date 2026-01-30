# FloodWaive DeepWaive Simulator Integration (CREXDATA) — Overview

This folder documents the **FloodWaive DeepWaive** integration for the **CREXDATA Weather Emergency** use case.

Contents:

- **Example outputs** (GeoTIFFs + metadata) in `Outputs/`
- **Public interface documentation** (OpenAPI + JSON Schemas) under `Schema/`
- **Export tooling** to reproduce exports under `Tools/`

FloodWaive is accessed through its public API; this repository focuses on published outputs plus the API subset used by the integration.

## Repository scope

This repository contains:

- Output datasets (GeoTIFFs) and their metadata
- Interface documentation (schemas + docs)
- Exporter/validator code

## Data model (high level)

FloodWaive Cloud Platform exposes an API for:

- **Areas**: geographic simulation domains
- **Rainfall Events**: precipitation forcing (1D intensity series or 2D GeoTIFF stacks)
- **Measures**: mitigation interventions / DEM modifications (e.g., barriers, terrain raise/lower)
- **Simulations**: configured runs for an Area (async execution)
- **Exports**: asynchronous export jobs (e.g., Simulation GeoTIFF export)

## Typical workflow

### 1) Create inputs

- Create or select an **Area**
- Create a **Rainfall Event** for that Area
- (Optional) Create **Measures** for that Area

### 2) Run a simulation

- Create a Simulation for the Area referencing:
  - `rainfall_event_id`
  - optional `measure_assignments` / `measures`

### 3) Export results

- Request a GeoTIFF export for the Simulation
  - For “maximum water levels”, use `timesteps: [-1]`
- Poll the Export until it reaches `status: completed`
- Download the GeoTIFF file(s) via the filesystem download endpoint (short-lived download URL)

See `Outputs/README.md` for the dataset format and conventions.