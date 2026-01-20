# Synthetic Vessel Proximity Events Generator

This repository contains a Jupyter notebook that assembles a **synthetic dataset of vessel‑proximity events** from raw AIS position data.  Given trajectories of many vessels, the notebook detects spatial intersections between pairs of vessels, aligns their timestamps so that the vessels meet at the same instant, and outputs a labelled dataset suitable for training and evaluating collision‑detection or trajectory‑prediction algorithms.

The generated data follow the schema used in the *Synthetic AIS Dataset of Vessel Proximity Events* (DOI 10.5281/zenodo.8358665).  Each synthetic event contains two vessels whose paths come within close proximity.  For each event the notebook produces a time‑ordered sequence of AIS messages including the vessel identifier (`mmsi`), time (milliseconds since epoch), geographic coordinates (`lon`, `lat`), heading/course over ground, and speed in knots.

## Overview

The notebook reads AIS positions from a CSV file named `coords.csv`.  Each row of this file represents a single AIS sample with the following columns:

| Column       | Description                                                 |
|-------------|-------------------------------------------------------------|
| `id`        | Unique identifier of the vessel (integer or string).        |
| `time`      | Timestamp in **milliseconds since epoch**.                  |
| `lat`       | Latitude of the AIS position.                               |
| `lon`       | Longitude of the AIS position.                              |
| `preds`     | Unused field from the original dataset (dropped).          |
| `c1`, `c2`  | Unused fields from the original dataset (dropped).          |
| `linestring`| Unused geometry field from the original dataset (dropped).  |

Only the first four columns (`id`, `time`, `lat` and `lon`) are required.  The notebook swaps the latitude and longitude columns when constructing `LineString` objects because Shapely expects coordinate pairs in `(x, y)` order.

For each vessel the samples are sorted by time and grouped by `id` to form trajectories.  The notebook then compares every pair of vessel trajectories to determine if their paths intersect.  A Shapely `LineString` represents each trajectory, and the `intersects()` method returns **`True` if the boundary or interior of the two geometries share any point**:contentReference[oaicite:0]{index=0}.  When two trajectories intersect the exact intersection point is obtained via `line1.intersection(line2)`.  If there is no intersection, Shapely returns an empty geometry rather than `None`:contentReference[oaicite:1]{index=1}.

For intersecting trajectories the notebook computes the speed and heading of each vessel using geodesic distances.  The speed calculation uses GeographicLib’s `Geodesic.WGS84.Inverse()` method, which solves the **inverse geodesic problem**: given two latitude/longitude pairs it returns the geodesic distance (`s12`) and the forward azimuths (`azi1`, `azi2`) at both points:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}.  Dividing the distance between successive positions by the time delta yields the vessel’s speed (in knots), and the forward azimuth is used as the course/heading.

To align the trajectories temporally, the script locates the recorded points nearest to the intersection for both vessels using a k‑dimensional tree (KD tree).  A KD tree partitions the data space into nested regions and allows the nearest neighbour of a query point to be found with only **O(log n) distance computations**, a significant improvement over the O(n) brute‑force approach:contentReference[oaicite:4]{index=4}.  Knowing the geodesic distances and speeds from the nearest recorded points to the collision point, the notebook computes how long each vessel would take to reach the intersection and adjusts the timestamps so that the vessels collide at the same instant.

After aligning times, the script iterates through the detected intersections a second time to refine the offsets and assigns new identifiers to each synthetic event (`mmsi` values are mapped to a `coll_id*10+1` / `coll_id*10+2` numbering scheme).  It then filters out events with too few AIS samples (less than ten points per trajectory), discards events where the collision occurs within the first three points, and removes events whose inferred time alignment differs too much from the expected difference computed from the nearest recorded points.  Finally, it exports two CSV files:

* **`collision_sample.csv`** – the synthetic AIS dataset containing the following columns:
  
  | Column      | Description                                                                                              |
  |------------|----------------------------------------------------------------------------------------------------------|
  | `station`  | AIS Station identifier.                                                                            |
  | `mmsi`     | MMSI or vessel identifier (integer).  Each collision event produces two consecutive identifiers.         |
  | `t`        | Timestamp in milliseconds since epoch (integer).                                                          |
  | `lon`      | Longitude of the AIS message (degrees).                                                                   |
  | `lat`      | Latitude of the AIS message (degrees).                                                                    |
  | `heading`  | Vessel heading in degrees (from 0° north, increasing clockwise).  Derived from the forward azimuth.      |
  | `course`   | Course over ground in degrees.  Identical to `heading` in this synthetic dataset.                         |
  | `speed`    | Speed in **knots** (integer) calculated from geodesic distance and time difference.                       |
  | `status`   | Navigation status (set to 1 for all messages).                                                            |

* **`collision_true.csv`** – results for the synthetic collisions, including the IDs of the two vessels involved, the coordinates of the collision point, the aligned collision time, indices of the nearest recorded points and other diagnostic fields.  Note that the `x` and `y` columns are swapped before export so that `x` corresponds to longitude and `y` to latitude.

An example of the resulting output may be found in *Synthetic AIS Dataset of Vessel Proximity Events* (DOI 10.5281/zenodo.8358665)

## Requirements

This project requires **Python 3.8 or newer**.  The following Python packages must be installed:

 **`pip install pandas numpy shapely geographiclib haversine scikit-learn`**

## Contact
For support and further inquires: ggrigoropoulos@kpler.com / [ggrigor89](https://github.com/ggrigor89)

## Contributors from Kpler Research Labs
Giannis Spiliopoulos [gsplpls](https://github.com/gsplpls) 
Ilias Chamatidis [ichamatidis-kpler](https://github.com/ichamatidis-kpler)

## Terms and Conditions 
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright (c) 2026 Kpler
