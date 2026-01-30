# JSON Schemas (CREXDATA integration)

These JSON Schemas describe a subset of request/response payloads relevant for integrating FloodWaive into CREXDATA.

Notes:

- Canonical field naming is **snake_case**.
- The live API may accept additional aliases (camelCase) and may return additional fields.
- Clients should ignore unknown response fields for forward compatibility.

Included schemas:

- `simulation.create.request.schema.json`
- `simulation.measure_assignment.schema.json`
- `rainfall_event.create.request.schema.json`
- `measure.create.request.schema.json`
- `simulation.export_geotiff.request.schema.json`
- `export.response.schema.json`
- `filesystem.download_url.response.schema.json`
