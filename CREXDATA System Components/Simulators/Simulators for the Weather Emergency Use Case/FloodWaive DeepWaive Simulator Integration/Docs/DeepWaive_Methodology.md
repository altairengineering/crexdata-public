# DeepWaive — scientific/methodological summary

This document provides a concise description of the DeepWaive approach as used in CREXDATA for integration documentation.

## 1) Model overview

DeepWaive is a **physics-informed deep learning surrogate model** for classical 2D hydrodynamic flood simulations (e.g., shallow-water-equation solvers). The intent is to approximate hydrodynamic simulation outputs with **drastically reduced inference time** while preserving physically plausible behavior for surface runoff.

## 2) Inputs and outputs (conceptual)

Typical inputs include:

- terrain / surface representation (DEM, relevant surface parameters)
- infrastructure and land-use information (where available/used)
- rainfall forcing (time series and/or spatial rainfall raster stacks)
- optional interventions/measures (terrain modifications, barriers, etc.)

Typical outputs include:

- raster time series and/or maxima for water depth
- optionally: velocity magnitude and direction fields (where configured)

## 3) Performance and validation (example KPIs)

DeepWaive outputs can be evaluated against hydrodynamic reference solutions using spatial overlap metrics such as the **Critical Success Index (CSI)** at chosen depth thresholds, and error statistics on water depth.

Reported values depend on the evaluation setup (area, rainfall forcing, resolution, and thresholds) and should be interpreted in that context.

## 4) System integration

For CREXDATA integration, DeepWaive is exposed as a **cloud service** with a **web-service API**. The API supports:

- creation and management of simulations
- parameterization via rainfall events and measures
- exports of results in standard formats (e.g., GeoTIFF)
