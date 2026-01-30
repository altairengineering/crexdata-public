# Measures & DEM adjustments (DGM-Anpassungen)

CREXDATA requires the ability to parameterize flood simulations with **interventions** and **terrain adjustments** (DGM/DEM modifications). In FloodWaive, these are represented as **Measures** that can be created for an Area and then applied to Simulations.

This document describes the **interface-level concept**.

## Concepts

### Measure

A **Measure** is a geospatial intervention that modifies simulation inputs, typically by:

- changing terrain elevation within a polygon (raise/lower)
- line-based “cut-through / clearcut” interventions with a width

In the integration API, geometries are provided as EPSG:4326 coordinates (\([lon, lat]\)) and associated with an Area.

### Applying measures to simulations

Simulations can reference Measures in two common ways:

- `measures`: list of measure IDs (backward compatibility)

## Example payloads

### Create a measure (polygon elevation change)

```json
{
  "name": "Barrier A (raise +0.5 m)",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[6.80, 51.50], [6.81, 51.50], [6.81, 51.51], [6.80, 51.51], [6.80, 51.50]]]
  },
  "elevation_change": 0.5
}
```

### Create a simulation referencing measures

```json
{
  "name": "CREXDATA scenario with measures",
  "model_id": "deepwaive-5.1",
  "resolution": 4,
  "rainfall_event_id": "rainfall_event-123e4567-e89b-12d3-a456-426614174000",
  "measure_assignments": [
    {
      "measure_id": "measure-6fa459ea-ee8a-3ca4-894e-db77e160355e",
      "parameter_overrides": {
        "elevation_change": 0.3
      }
    }
  ]
}
```
