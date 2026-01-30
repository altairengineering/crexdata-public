# Output format (GeoTIFF) — FloodWaive → CREXDATA

This document describes the conventions used for the **published FloodWaive simulation outputs** in this repository.

For the canonical dataset description and folder structure, see `Outputs/README.md`.

## File types

### `max_water_levels.tif`

GeoTIFF raster representing **maximum water depth** per grid cell over the full simulation duration.

Typical conventions (see per-simulation README for exact metadata):

- **Data type**: Float32
- **Unit**: meters (water depth above terrain)
- **NoData**: `-9999.0`
- **Compression**: DEFLATE
- **CRS**: varies by area (often local projected CRS; see per-area README)

## Repository structure

```
Outputs/
  README.md
  CREXDATA_Dortmund/
    README.md
    simulation-550e8400-e29b-41d4-a716-446655440000/
      README.md
      max_water_levels.tif
```

## Visualization (QGIS)

- Open the GeoTIFF in QGIS.
- Use a sequential color ramp (e.g., light→dark blue).

A generic QGIS style is provided at `Outputs/qgis/max_water_levels.qml`.

