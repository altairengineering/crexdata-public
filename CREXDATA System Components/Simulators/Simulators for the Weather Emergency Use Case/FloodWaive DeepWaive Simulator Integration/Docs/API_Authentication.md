# FloodWaive API authentication (CREXDATA integration)

FloodWaive API access uses **bearer token authentication**.

## Supplying the token

Requests include an `Authorization` header:

```
Authorization: Bearer $FLOODWAIVE_API_KEY
```

## Setup

Set the API token in your environment:

```bash
export FLOODWAIVE_API_KEY="fw_example_token"
```

## Example

```bash
export SIMULATION_ID="simulation-550e8400-e29b-41d4-a716-446655440000"

curl -sS \\
  -H "Authorization: Bearer $FLOODWAIVE_API_KEY" \\
  "https://api.floodwaive.de/v1/simulations/$SIMULATION_ID"
```

## Organization scoping

Tokens are issued for a specific organization. Access control and permissions are enforced by the service.
