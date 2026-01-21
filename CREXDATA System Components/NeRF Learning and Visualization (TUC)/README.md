
# NeRF Learning and Visualization [(TUC)](https://www.tuc.gr/en/home)

The most up-to-date version of this repository can be found at: https://github.com/psklavos1/NeRF-Sys

This repository provides the real-system integration, developed within the [**CREXDATA**](https://crexdata.eu/) Horizon project, of the Adaptive NeRF framework introduced in the paper:

```bibtex
@inproceedings{sklavos2026adaptivenerf,
  title     = {An Adaptive NeRF Framework for City-Scale Emergency Awareness},
  author    = {Panagiotis Sklavos and Georgios Anestis and Antonios Deligiannakis},
  booktitle = {Proceedings of the 14th International Conference on Emerging Internet, Data \& Web Technologies (EIDWT 2026)},
  year      = {2026},
  note      = {Accepted for publication}
}
```

The aim of this component is to provide a reproducible and scalable system for
training, adapting, and visualizing city-scale Neural Radiance Fields (NeRFs)
under time-crtitical emergency conditions. By decomposing large scenes into
spatially localized NeRF experts and enabling incremental updates as new aerial
imagery becomes available, the system supports rapid adaptation, preserves
previously learned structure, and allows reliable deployment of NeRF-based
scene understanding in large, real-world environments.

---

## Architecture

![Overview of the adaptive NeRF system architecture](nerf_arch.jpg)

### Mediator

The system is centered around a persistent orchestrator, called the **Mediator**,
which runs as a long-lived process and coordinates **NeRF workloads** based on
incoming job configurations.

The Mediator continuously listens to a predefined **Kafka** topic for structured
**JSON** job descriptions and launches **NeRF jobs** accordingly. Jobs are executed
in isolation and in a non-blocking manner, allowing multiple jobs to run
concurrently or sequentially in the background.

### NeRF Runner

The NeRF Runner acts as an isolated job executor responsible for running NeRF
workloads based on structured JSON configurations. For each submitted job, the
configuration fully defines the execution parameters, including data sources,
model settings, and runtime behavior.

Depending on the requested operation, the runner handles training, adaptation,
evaluation, or visualization tasks (NeRF rendering) within an isolated execution context. All artifacts and logs produced during execution are stored under the job-specific
output directory.

---
## Structure

```text
.
├── adaptive_nerf/                  # Core Adaptive NeRF framework
│   ├── common/                     # Shared utilities (logging, helpers)
│   ├── data/                       # Dataset storage and parsing
│   ├── models/                     # NeRF models and architectures
│   ├── nerfs/                      # NeRF-specific helpers
│   ├── pipelines/                  # Meta-training and runtime adaptation pipelines
│   ├── scripts/                    # Internal execution scripts for data preparation or logging statistics
│   ├── viewer/                     # Interactive NeRF viewer
│   ├── nerf_runner.py              # Entry point for executing NeRF operations
│   ├── utils.py                    # Entry point utilities
│   └── __init__.py
│
├── configs/                        # Job and experiment configuration files
├── kafka_utils/                    # Kafka helpers for the server communication
├── logs/                           # Job outputs, logs, and checkpoints
├── scripts/                        # System-level utility scripts
│
├── mediator.py                     # Long-running orchestrator (system entry point)
├── README.md                       # System documentation
└── requirements.txt                # Python dependencies

```

---

## Environment Setup
Python 3.11 and CUDA 11.8 are verified for compatibility so we provide a stable installation guide:

### 1. Clone the repository
```bash
git clone http://github.com/altairengineering/crexdata-public/tree/main
cd WP4/NeRF-Learning-TUC
```

> Note: The most up-to-date version of this repository can be found at: [repo](https://github.com/psklavos1/NeRF-Sys.git).

### 2) Create the environmet & install requirements.
We provide an example setup using conda.
Install the correct version of PyTorch and dependencies:
```bash
conda create -n nerfenv python=3.11 -y
conda activate nerfenv
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 3) Install tiny-cuda-nn.
Efficient Instant-NGP utilization requires `tiny-cuda-nn`.

Install using the official [NVLabs](https://github.com/NVlabs/tiny-cuda-nn?utm_source) instructions. A compatible example is presented next:
```bash
conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
nvcc --version   # should say 11.8
pip install --no-build-isolation -v "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

In case of failure assert all the provided [requirements](https://github.com/NVlabs/tiny-cuda-nn?utm_source) are satisfied.

---

## Data Preparation
The framework operates on image-based datasets with known or recoverable camera poses.
Camera intrinsics and poses must be estimated prior to use. [COLMAP](https://colmap.github.io/) is the
recommended tool for this step. If GPS coordinates are not available, COLMAP’s model alignment utilities
(e.g. Manhattan-world alignment) can be used as a fallback as the data need to be geo-referenced.

> COLMAP models can be exported in either **ECEF** or **ENU** coordinates. The recommended
workflow is to export in **ECEF** and perform the ECEF→ENU conversion internally using
the provided scripts. Direct ENU exports are also supported.

### Dataset Preparation Scripts

The following scripts are provided for dataset preparation and clustering.

- `adaptive_nerfs/scripts/prepare_dataset.py` 
converts a COLMAP reconstruction into the framework’s internal dataset format. The
script normalizes scene scale for stable training and optionally converts poses to a
common ENU reference frame, storing all outputs using the framework’s coordinate
conventions.
#### Input Data
```text
data_path/
  ├── model/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  └── images/   # All registered images used by the COLMAP model
```
#### Example
```bash
./scripts/prepare_dataset.py --data_path data/drz --output_path data/drz/out/prepared --val_split 0.3 --scale_strategy camera_max --ecef_to_enu --enu_ref median
```

- `scripts/create_clusters.py` generates spatial training partitions for NeRF experts using distance-based routing.
Rays are assigned to one or more spatial regions, with optional boundary overlap.
2D clustering is recommended to reduce computation without affecting results.

#### Example

```bash
 ./scripts/create_clusters.py --data_path data/drz/out/prepared --grid_dim 2 2 --cluster_2d --boundary_margin 1.05 --ray_samples 256 --center_pixels --scene_scale 1.1 --output g22_grid_bm105_ss11 --resume
```

---

## Usage
This system supports three types of requests, each corresponding to a distinct
NeRF operation:

1. **train**: Performs offline meta-learning to obtain a coarse initialization
   of the scene that enables rapid future adaptation.
    > **Output**: Contains training execution logs, tensorboard logs for structured training monitoring (e.g. live training plots), and model checkpoints for future use. 

2. **eval**: Adapts a pre-trained model to newly acquired data for specified test-time-optimization steps (TTO) and evaluates
   reconstruction quality using metrics such as PSNR, SSIM, and LPIPS.
    > **Output**: Contains evaluation execution logs, a structured metrics dataframe, and the redenering results at `logs/<job_id>/gt and pred/<tto>` containing the evaluation images and the produced reconstruction by the NeRF respsectively.

3. **view**: Launches the NeRF viewer for interactive navigation or live
   monitoring of the adaptation process.
    > **Output**: Contains viewer execution logs and user-captured screenshots and checkpoints generated through the viewer’s API controls. The viewer is web-based; when a `view` request is executed, the executor launches a viewer instance at `0.0.0.0:7070`, which users can access through a browser. The viewer’s lifecycle is managed via the provided interactive controls.

The logs for all job executions are stored at: `logs/<job_id>` while the mediator's logs are appended at `logs/`.

### Demo
To experiment with the mediator 3 demo configurations are provided under `/configs` for training, evaluation and viewer operation respectively. Prepared data provided by: [DRZ](https://rettungsrobotik.de/) are located at `data/drz/out/example` while a demo checkpoint with 4 expert NeRFs is provided for testing: [`checkpoint`](https://github.com/psklavos1/adaptive-city-nerf/releases/tag/v1.0/4_experts.zip)

After downloading the model checkpoint, extract into: `logs/example`

1. **Run Mediator**  
    ```bash
    python mediator.py --topic nerfConfigs (--cleanup)
    ```
    > **Note:** If cleanup is enabled `/logs` is emptied prior of execution. `logs/example` is preserved which contains the demo checkpoint.
2. **Send Demo config**  
    ```bash
    ./scripts/send_config.py --topic nerfConfigs --configPath <configs/train.json | configs/eval.json | configs/view.json>
    ```
3.  **Monitor outputs** 
> Outputs provided at `logs/mediator.txt` for the mediator and `logs/<job_id>` for the executed job.
    
---

## Configuration

Each job is defined by a JSON configuration file. The configuration fully specifies the dataset, model, optimization, and runtime
behavior for a single job.
All jobs are submitted through a single entry point and are distinguished by the
`op` field.


### Train (`op: "train"`)

Offline training meta-learns a good model initialization that allows rapid adaptation.

| Field | Description |
|------|------------|
| `op` | Operation type (`"train"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `near`, `far` | Ray bounds |
| `ray_samples` | Samples per ray |
| `batch_size` | Number of tasks per expert |
| `support_rays` | Rays per task (support set) |
| `query_rays` | Rays per task (query set) |
| `num_submodules` | Number of NeRF experts |
| `log2_hashmap_size` | Hash grid size |
| `max_resolution` | Max encoding resolution |
| `optimizer` | Optimizer type |
| `lr` | Base learning rate |
| `outer_steps` | Meta-optimization steps |
| `inner_iter` | Inner-loop iterations |
| `inner_lr` | Inner-loop learning rate |
| `checkpoint_path` | (Optional) chekpoint to continue training from |
| `prefix` | Checkpoint prefix |

---

### Eval (`op: "eval"`)

Runtime evaluation of a pre-trained model using newly available images to produce reconstruction statistics.

| Field | Description |
|------|------------|
| `op` | Operation type (`"eval"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `near`, `far` | Ray bounds |
| `ray_samples` | Samples per ray |
| `support_rays` | Rays used for adaptation |
| `optimizer` | Optimizer type |
| `lr` | Adaptation learning rate |
| `tto` | Number of adaptation steps |
| `checkpoint_path` | Input checkpoint directory |
| `prefix` | Checkpoint prefix |

---

### View (`op: "view"`)

Launches the interactive NeRF viewer with entry-point at: `0.0.0.0:7070`

| Field | Description |
|------|------------|
| `op` | Operation type (`"view"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `ray_samples` | Samples per ray for rendering |
| `checkpoint_path` | Input checkpoint directory |
| `prefix` | Checkpoint prefix |

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
This research is supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101092749, project [**CREXDATA**](https://crexdata.eu/). This contribution is provided on behalf of the [**Technical University of Crete (TUC)**](https://www.tuc.gr/en/home).

We sincerely thank the members of the [Deutsches Rettungsrobotik Zentrum (DRZ)](https://rettungsrobotik.de/), for supporting the data acquisition and providing the aerial dataset used in this work.
