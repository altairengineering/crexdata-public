#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_API_BASE_URL = "https://api.floodwaive.de/v1"


class ApiError(RuntimeError):
    pass


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.strip()
    env = os.environ.get("FLOODWAIVE_API_KEY", "").strip()
    if env:
        return env
    raise ApiError("Missing API key. Set FLOODWAIVE_API_KEY or pass --api-key.")


def _http_json(
    *,
    method: str,
    url: str,
    api_key: str,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    payload = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "floodwaive-exporter/crexdata",
    }
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, method=method.upper(), data=payload, headers=headers)
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise ApiError(f"HTTP {e.code} for {method} {url}: {msg}") from e
    except URLError as e:
        raise ApiError(f"Network error for {method} {url}: {e}") from e

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ApiError(f"Expected JSON response from {url}, got: {data[:200]!r}") from e


def _download_to_file(url: str, dst: Path, timeout_s: int = 300) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url=url, method="GET", headers={"User-Agent": "floodwaive-exporter/crexdata"})
    with urlopen(req, timeout=timeout_s) as resp, dst.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _api_url(api_base_url: str, path: str) -> str:
    return api_base_url.rstrip("/") + "/" + path.lstrip("/")


def _get_download_url(api_base_url: str, api_key: str, file_path: str) -> str:
    trimmed = file_path.lstrip("/")
    encoded = quote(trimmed, safe="/")
    url = _api_url(api_base_url, f"/filesystem/download/{encoded}")
    data = _http_json(method="GET", url=url, api_key=api_key)
    if "download_url" not in data:
        raise ApiError(f"Unexpected download-url response: {data}")
    return str(data["download_url"])


def _poll_export(api_base_url: str, api_key: str, export_id: str, *, interval_s: float, timeout_s: int) -> Dict[str, Any]:
    url = _api_url(api_base_url, f"/exports/{export_id}")
    deadline = time.time() + timeout_s
    last_status = None
    while True:
        export = _http_json(method="GET", url=url, api_key=api_key)
        status = str(export.get("status", "")).lower()
        if status != last_status:
            print(f"[export {export_id}] status={status} progress={export.get('progress_percent')}", file=sys.stderr)
            last_status = status

        if status in {"completed", "failed", "cancelled"}:
            return export

        if time.time() >= deadline:
            raise ApiError(f"Timed out waiting for export {export_id} after {timeout_s}s")

        time.sleep(interval_s)


def cmd_export_simulation(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.api_key)
    api_base_url = args.api_base_url

    timesteps = [int(t.strip()) for t in args.timesteps.split(",") if t.strip() != ""]
    body: Dict[str, Any] = {
        "timesteps": timesteps,
        "data_type": args.data_type,
        "scope": args.scope,
    }
    if args.bounds:
        body["bounds"] = [float(x) for x in args.bounds.split(",")]
    if args.target_crs:
        body["target_crs"] = args.target_crs
    if args.resolution:
        body["resolution"] = int(args.resolution)
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]

    export_url = _api_url(api_base_url, f"/simulations/{args.simulation_id}/export-geotiff")
    export = _http_json(method="POST", url=export_url, api_key=api_key, body=body)
    export_id = export.get("export_id")
    if not export_id:
        raise ApiError(f"Missing export_id in response: {export}")

    export = _poll_export(
        api_base_url,
        api_key,
        str(export_id),
        interval_s=args.poll_interval_s,
        timeout_s=args.timeout_s,
    )
    status = str(export.get("status", "")).lower()
    if status != "completed":
        raise ApiError(f"Export did not complete successfully: status={status} error={export.get('error_message')}")

    filesystem_path = str(export.get("filesystem_path", "")).rstrip("/")
    if not filesystem_path:
        raise ApiError(f"Export missing filesystem_path: {export}")

    out_dir = Path(args.out_dir)
    if args.repo_layout:
        sim_dir = out_dir / args.simulation_id
    else:
        sim_dir = out_dir
    sim_dir.mkdir(parents=True, exist_ok=True)

    file_progress = export.get("file_progress") or []
    downloaded = 0
    for item in file_progress:
        if str(item.get("status", "")).lower() != "completed":
            continue
        filename = item.get("filename")
        if not filename:
            continue
        src_path = f"{filesystem_path}/{filename}"
        dl_url = _get_download_url(api_base_url, api_key, src_path)

        dst_name = filename
        if args.repo_layout and timesteps == [-1] and len(file_progress) == 1:
            dst_name = "max_water_levels.tif"

        dst = sim_dir / dst_name
        print(f"Downloading {src_path} -> {dst}", file=sys.stderr)
        _download_to_file(dl_url, dst, timeout_s=args.download_timeout_s)
        downloaded += 1

    if downloaded == 0:
        raise ApiError("No completed files found in export file_progress.")

    if args.write_readme:
        sim = _http_json(
            method="GET",
            url=_api_url(api_base_url, f"/simulations/{args.simulation_id}"),
            api_key=api_key,
        )
        _write_minimal_sim_readme(sim_dir / "README.md", sim)

    print(f"Done. Downloaded {downloaded} file(s).", file=sys.stderr)
    return 0


def _write_minimal_sim_readme(dst: Path, sim: Dict[str, Any]) -> None:
    sim_id = str(sim.get("simulation_id") or sim.get("id") or "unknown")
    area_id = str(sim.get("area_id") or "unknown")
    created_at = sim.get("created_at") or ""
    resolution = sim.get("resolution")
    model_id = sim.get("model_id") or ""
    rainfall_event_id = sim.get("rainfall_event_id") or ""

    dst.write_text(
        "\n".join(
            [
                f"# {sim_id}",
                "",
                "**Flood Simulation Results**",
                "",
                f"`{sim_id}`",
                "",
                "## Overview",
                "",
                "| Property | Value |",
                "|:--|:--|",
                f"| **Simulation ID** | `{sim_id}` |",
                f"| **Area ID** | `{area_id}` |",
                f"| **Created At** | {created_at} |",
                f"| **Model** | {model_id} |",
                f"| **Resolution** | {resolution} m |",
                f"| **Rainfall Event ID** | `{rainfall_event_id}` |",
                "",
                "Source: FloodWaive API export.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass
class ManifestRow:
    area: str
    simulation_id: str
    resolution_m: Optional[int]
    model: str
    peak_intensity_mmh: Optional[float]
    total_precip_mm: Optional[float]
    rel_path: str
    size_bytes: int
    sha256: str


_MD_FIELD_RE = re.compile(r"^\|\s*\*\*(?P<key>[^*]+)\*\*\s*\|\s*(?P<value>.*?)\s*\|\s*$")


def _parse_sim_readme(readme_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = readme_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    lines = [ln.strip() for ln in text.splitlines()]
    if lines and lines[0].startswith("# "):
        out["__title__"] = lines[0][2:].strip()

    for ln in lines:
        m = _MD_FIELD_RE.match(ln)
        if not m:
            continue
        key = m.group("key").strip()
        value = m.group("value").strip()
        out[key] = value

    return out


def _to_float_maybe(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    cleaned = s.replace(",", "").strip()
    cleaned = cleaned.replace("mm/h", "").replace("mm", "").strip()
    try:
        return float(cleaned)
    except Exception:
        return None


def _to_int_maybe(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    cleaned = str(s).replace("m", "").strip()
    try:
        return int(float(cleaned))
    except Exception:
        return None


def _iter_simulation_dirs(outputs_root: Path) -> Iterable[Path]:
    for area_dir in sorted(outputs_root.iterdir()):
        if not area_dir.is_dir():
            continue
        if area_dir.name in {"qgis"}:
            continue
        for sim_dir in sorted(area_dir.iterdir()):
            if sim_dir.is_dir() and sim_dir.name.startswith("simulation-"):
                yield sim_dir


def cmd_build_manifest(args: argparse.Namespace) -> int:
    outputs_root = Path(args.outputs_root).resolve()
    if not outputs_root.exists():
        raise ApiError(f"Outputs root does not exist: {outputs_root}")

    rows: List[ManifestRow] = []
    for sim_dir in _iter_simulation_dirs(outputs_root):
        area = sim_dir.parent.name
        sim_id = sim_dir.name
        tif = sim_dir / "max_water_levels.tif"
        readme = sim_dir / "README.md"
        if not tif.exists():
            continue

        meta = _parse_sim_readme(readme) if readme.exists() else {}
        resolution_m = _to_int_maybe(meta.get("Resolution"))
        model = meta.get("Model") or ""
        peak = _to_float_maybe(meta.get("Peak Intensity"))
        total = _to_float_maybe(meta.get("Total Precipitation"))

        rel_path = str(tif.relative_to(outputs_root))
        size_bytes = tif.stat().st_size
        sha256 = _sha256_file(tif)

        rows.append(
            ManifestRow(
                area=area,
                simulation_id=sim_id,
                resolution_m=resolution_m,
                model=model,
                peak_intensity_mmh=peak,
                total_precip_mm=total,
                rel_path=rel_path,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )

    if not rows:
        raise ApiError("No outputs found (expected simulation-*/max_water_levels.tif).")

    generated_at = _now_utc_iso()
    if args.manifest_csv:
        _write_manifest_csv(Path(args.manifest_csv), rows, generated_at)
    if args.manifest_json:
        _write_manifest_json(Path(args.manifest_json), rows, generated_at)

    print(f"Done. Manifest entries: {len(rows)}", file=sys.stderr)
    return 0


def _write_manifest_csv(path: Path, rows: List[ManifestRow], generated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generated_at", generated_at])
        writer.writerow(
            [
                "area",
                "simulation_id",
                "resolution_m",
                "model",
                "peak_intensity_mmh",
                "total_precip_mm",
                "file_rel_path",
                "file_size_bytes",
                "sha256",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.area,
                    r.simulation_id,
                    r.resolution_m if r.resolution_m is not None else "",
                    r.model,
                    r.peak_intensity_mmh if r.peak_intensity_mmh is not None else "",
                    r.total_precip_mm if r.total_precip_mm is not None else "",
                    r.rel_path,
                    r.size_bytes,
                    r.sha256,
                ]
            )


def _write_manifest_json(path: Path, rows: List[ManifestRow], generated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at,
        "entries": [
            {
                "area": r.area,
                "simulation_id": r.simulation_id,
                "resolution_m": r.resolution_m,
                "model": r.model,
                "peak_intensity_mmh": r.peak_intensity_mmh,
                "total_precip_mm": r.total_precip_mm,
                "file_rel_path": r.rel_path,
                "file_size_bytes": r.size_bytes,
                "sha256": r.sha256,
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floodwaive_exporter.py")
    p.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    p.add_argument("--api-key", default=None, help="API key (or set FLOODWAIVE_API_KEY)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("export-simulation", help="Request GeoTIFF export and download results")
    s.add_argument("--simulation-id", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--timesteps", default="-1", help="Comma-separated timesteps. Use -1 for max water levels.")
    s.add_argument("--data-type", default="depth", choices=["depth", "velocity", "direction"])
    s.add_argument("--scope", default="full_area", choices=["full_area", "window"])
    s.add_argument("--bounds", default=None, help="minX,minY,maxX,maxY (required when scope=window)")
    s.add_argument("--target-crs", default=None, help="Optional target CRS (e.g., EPSG:25832)")
    s.add_argument("--resolution", default=None, help="Optional export resolution override (meters)")
    s.add_argument("--tags", default=None, help="Optional comma-separated tags")
    s.add_argument("--poll-interval-s", type=float, default=3.0)
    s.add_argument("--timeout-s", type=int, default=900)
    s.add_argument("--download-timeout-s", type=int, default=600)
    s.add_argument("--repo-layout", action="store_true", help="Store files under OUT_DIR/SIMULATION_ID/")
    s.add_argument("--write-readme", action="store_true", help="Write a minimal README.md using GET /simulations/{id}")
    s.set_defaults(func=cmd_export_simulation)

    m = sub.add_parser("build-manifest", help="Generate manifest files for Outputs/")
    m.add_argument("--outputs-root", required=True, help="Path to Outputs/ directory")
    m.add_argument("--manifest-csv", default=None, help="Write manifest CSV to this path")
    m.add_argument("--manifest-json", default=None, help="Write manifest JSON to this path")
    m.set_defaults(func=cmd_build_manifest)

    return p


def main(argv: List[str]) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return int(args.func(args))
    except ApiError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

