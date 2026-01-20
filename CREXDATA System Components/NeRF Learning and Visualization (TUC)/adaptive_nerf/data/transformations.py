import numpy as np
import pymap3d as pm
from argparse import Namespace

# Basis transforms
# COLMAP camera basis: RDF (Right, Down, Forward)
# Target world basis:  DRB (Down, Right, Back)
RDF_TO_DRB = np.array(
    [
        [0.0, 1.0, 0.0],  # Down   <- RDF_y
        [1.0, 0.0, 0.0],  # Right  <- RDF_x
        [0.0, 0.0, -1.0],  # Back   <- -RDF_z
    ],
    dtype=np.float64,
)

# ENU → DRB (Down = -Up, Right = East, Back = North)
ENU_TO_DRB = np.array(
    [
        [0.0, 0.0, -1.0],  # Down  <- -Up
        [1.0, 0.0, 0.0],  # Right <- East
        [0.0, -1.0, 0.0],  # Back  <- -North
    ],
    dtype=np.float64,
)

# Camera basis change: RDF → RUB (Right, Up, Back)
RDF_TO_RUB = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def ellipsoid_wgs84():
    """Return WGS84 ellipsoid for pymap3d."""
    try:
        return pm.Ellipsoid.from_name("wgs84")
    except AttributeError:
        return pm.Ellipsoid(6378137.0, 6356752.314245179)


def enu_span_meters(lat_min, lat_max, lon_min, lon_max, lat_ref, lon_ref, h_ref, ell):
    """Compute N/E span in meters of a lat/lon box around a reference ENU origin."""
    n1 = pm.ecef2enu(
        *pm.geodetic2ecef(lat_min, lon_ref, h_ref, ell=ell),
        lat_ref,
        lon_ref,
        h_ref,
        ell=ell,
    )[1]
    n2 = pm.ecef2enu(
        *pm.geodetic2ecef(lat_max, lon_ref, h_ref, ell=ell),
        lat_ref,
        lon_ref,
        h_ref,
        ell=ell,
    )[1]
    e1 = pm.ecef2enu(
        *pm.geodetic2ecef(lat_ref, lon_min, h_ref, ell=ell),
        lat_ref,
        lon_ref,
        h_ref,
        ell=ell,
    )[0]
    e2 = pm.ecef2enu(
        *pm.geodetic2ecef(lat_ref, lon_max, h_ref, ell=ell),
        lat_ref,
        lon_ref,
        h_ref,
        ell=ell,
    )[0]
    return abs(n2 - n1), abs(e2 - e1)


def choose_enu_origin(
    policy: str,
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    ordered_indices: np.ndarray,
    hparams: Namespace,
):
    """Choose ENU origin (lat, lon, alt, desc) using a simple policy."""
    policy = policy.lower()
    if policy == "first":
        idx0 = ordered_indices[0]
        lat0, lon0, h0 = float(lats[idx0]), float(lons[idx0]), float(alts[idx0])
        desc = "first camera (id-sorted)"
    elif policy == "mean":
        lat0, lon0, h0 = float(lats.mean()), float(lons.mean()), float(alts.mean())
        desc = "mean of all cameras"
    elif policy == "median":
        lat0, lon0, h0 = (
            float(np.median(lats)),
            float(np.median(lons)),
            float(np.median(alts)),
        )
        desc = "median of all cameras"
    elif policy == "custom":
        if None in (hparams.enu_ref_lat, hparams.enu_ref_lon, hparams.enu_ref_alt):
            raise ValueError(
                "--enu_ref=custom requires --enu_ref_lat, --enu_ref_lon, --enu_ref_alt"
            )
        lat0, lon0, h0 = (
            float(hparams.enu_ref_lat),
            float(hparams.enu_ref_lon),
            float(hparams.enu_ref_alt),
        )
        desc = "custom user-provided coordinates"
    else:
        raise ValueError(f"Unknown --enu_ref: {policy}")
    return lat0, lon0, h0, desc


def ecef_to_enu_rot(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation matrix mapping ECEF vectors to ENU components at (lat, lon)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sL, cL = np.sin(lon), np.cos(lon)
    sB, cB = np.sin(lat), np.cos(lat)
    # Rows are unit ENU axes expressed in ECEF; v_enu = Q * v_ecef
    Q = np.array(
        [
            [-sL, cL, 0.0],  # East
            [-sB * cL, -sB * sL, cB],  # North
            [cB * cL, cB * sL, sB],  # Up
        ],
        dtype=np.float64,
    )
    return Q


def is_likely_ecef(C: np.ndarray) -> bool:
    """Heuristic check whether coordinates are likely ECEF (Earth-centered)."""
    r = np.linalg.norm(C, axis=1)
    return r.mean() > 1e6 and r.std() < 5e5
