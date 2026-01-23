# AI Agent Instructions: Multi-Scale COVID-19 Lung Infection Simulation

## Project Overview

This is a **coupled multi-scale simulation** combining:
- **ALYA** (CFD/particle deposition): Lung-scale and alveoli-scale airflow/particle transport
- **PhysiBoSS** (agent-based cell modeling): Cell-scale COVID-19 infection dynamics
- **Bidirectional coupling**: ALYA outputs drive PhysiBoSS initial conditions; PhysiBoSS outputs modify ALYA boundary conditions in subsequent rounds

The simulation runs in **iterative rounds** where each round represents disease progression, with infected lung regions being "closed" in subsequent ALYA runs.

## Architecture & Data Flow

### Round-Based Pipeline (orchestrator.sh)

```
Round N:
  1. run_alya_lung       → Simulate lung-scale airflow/deposition
  2. run_alya_alveoli    → Simulate alveoli-scale deposition
  3. run_project2plane   → Project 3D alveoli data to 2D plane
  4. prepare_physiboss_input → Parse ALYA outputs, generate PhysiBoSS params
  5. run_physiboss_all   → Run parallel PhysiBoSS simulations (5 lobes)
  6. combine outputs     → Evaluate tissue damage, flag infected lobes
  Loop to Round N+1 with modified lung geometry
```

**Key insight**: Round 0 uses surrogate lung model; subsequent rounds use standard model with closed lobes based on infection severity.

### Lobe Mapping & Simulation IDs

Five lung lobes are simulated independently:
- `LU` (Left Upper, code 31), `LL` (Left Lower, 32)
- `RU` (Right Upper, 33), `RM` (Right Middle, 34), `RL` (Right Lower, 35)

Each PhysiBoSS instance runs in `experiments/{EXPID}/sim_{1-5}/` corresponding to one lobe.

### Critical File Structures

**Per-simulation instance** (`sim_N/`):
- `settings.xml` - PhysiBoSS config (generated from base config + JSON params)
- `variables.txt` - Stores `particle_type alveoli_id MOI` (parsed by `get_sim_id()`)
- `moi.txt` - Multiplicity of infection value from ALYA lung output
- `round_{R}/` subdirectories for each round:
  - `input_virion_alya.txt` - ALYA deposition file (copied from alveoli run)
  - `round.txt` - Tag (0=healthy, 1=infected/closed)
  - `final.csv` - PhysiBoSS cell timeseries output

**Experiment root** (`experiments/{EXPID}/`):
- `round_{R}.csv` - Summary CSV mapping `sim_id,tag` for all lobes
- `alya/round_{R}/lung/` - Lung simulation outputs including `lung_depo.out`
- `alya/round_{R}/alveoli/` - Alveoli outputs including `depo.pts-depo.csv`

## Key Scripts & Workflows

### Python Integration Scripts (pb/scripts/)

**`combined_alveoli.py`**:
- `parse_alveoli_file()`: Extract particle deposition MOI from ALYA text output
- `get_lung_data()`: Parse lung-scale deposition per lobe
- `detect_dead_tissue()`: Evaluate PhysiBoSS .mat files for tissue death
- `wait_for_final_csvs()`: Monitor simulation completion

**`input_alya.py`**:
- `modify_alya_input()`: Modify ALYA `fensap.nsi.dat` to close infected lobes (tag=1)
- `read_fensap_boundary_last_time()`: Parse boundary flow rates from previous round
- `combine_physiboss_outputs()`: Aggregate PhysiBoSS results into `round_{R}.csv`
- Modes: `"pb"` (combine outputs), `"lung"` (modify input), `"input_pb"` (prepare next round)

**`params.py`** (pb/python/):
- `params_to_xml()`: Update PhysiBoSS XML config from JSON parameters
- Also writes `variables.txt` for simulation tracking

### Bash Orchestration

**`orchestrator.sh`**: Main SLURM job script
- Load modules: `cmake oneapi fftw dlb ucx`
- Run 4 rounds by default (`ROUNDS=4`)
- Parallel PhysiBoSS execution via background processes
- Uses `srun` for MPI-parallelized ALYA

**Environment variables**:
- `TURBINE_OUTPUT`: Experiment output directory
- `EMEWS`: Points to `pb/` directory
- `BASE_CONFIG`: Template XML (`pb/PhysiBoSSv2/config/debug.xml`)
- `EXE`: PhysiBoSS executable (`pb/PhysiBoSSv2/COVID19`)

## PhysiBoSS (PhysiCell + MaBoSS)

### Building & Execution

```bash
cd pb/PhysiBoSSv2
make          # Compiles COVID19 executable
./COVID19 config/debug.xml  # Run with config
```

**Custom modules** (`custom_modules/`): COVID-specific viral dynamics, immune response
**Intracellular models**: MaBoSS boolean networks for cell signaling (`boolean_network/*.bnd/*.cfg`)

### Configuration Patterns

XML configs define:
- `<domain>`: 2D grid (880×880 μm typical, 20 μm voxels)
- `<user_parameters>`: Custom fields like `simulation_id`, `alveoli_id`, `particle_type`, `multiplicity_of_infection`, `input_virion_alya`, `rounds`
- `<cell_definitions>`: 8 cell types (epithelium, immune cells, fibroblasts)

**Critical**: `save.folder` must match instance directory; `turbine` parameter stores experiment root.

### Round Continuation Logic (main.cpp)

PhysiBoSS checks for `round.txt` in subdirectories:
- If `round_{R}/round.txt` exists and contains "0" → continue simulation
- If contains "1" → lobe is closed, no further simulation
- Reads `input_virion_alya.txt` to set initial virion concentrations

## Development Workflows

### Running a New Experiment

```bash
sbatch orchestrator.sh my_experiment_id
# Monitor: tail -f logs/agent_job_*.err
```

### Debugging Individual Components

**Test ALYA alone**:
```bash
cd alya/lung_surrogate/
srun ../alya fensap > lung.out
./alldepo.x fensap > lung_depo.out
```

**Test single PhysiBoSS instance**:
```bash
cd pb/PhysiBoSSv2
./COVID19 config/debug.xml
# Check output/ folder for results
```

**Validate data flow**:
```bash
# Check ALYA→PhysiBoSS params generation
python3 pb/scripts/combined_alveoli.py prepare \
  experiments/test/alya/round_0/lung/lung_depo.out \
  experiments/test/alya/round_0/alveoli/depo.pts-depo.csv \
  0 pb/data/variables.txt

# Check PhysiBoSS→ALYA feedback
python3 pb/scripts/input_alya.py pb experiments/test 0
```

## Common Patterns & Conventions

### Error Handling
- Pipeline uses `set -euo pipefail` for immediate failure detection
- SLURM job logs in `logs/agent_job_%j.{out,err}`

### Parallel Execution
- PhysiBoSS instances run in background with `&` in bash loop
- Wait for completion before post-processing
- ALYA uses MPI via `srun`

### File Naming
- ALYA outputs: `fensap*.{out,dat,csv,nsi}`
- PhysiBoSS: `output_*_{8-digit-index}.xml`, `states_*.csv`, `final.csv`
- Projection tool: `depo.pts-depo.csv` (3D→2D mapped coordinates)

### XML Parsing
- Use `xml.etree.ElementTree` (Python) or `pugixml` (C++)
- PhysiBoSS configs follow nested `<PhysiCell_settings>` structure

## External Dependencies

- **ALYA**: Closed-source CFD solver (executable at `alya/alya`)
- **MaBoSS**: Boolean network simulator (bundled in `PhysiBoSSv2/addons/PhysiBoSS/`)
- **Python packages**: `numpy`, `pandas`, `scipy` (for `.mat` file reading)
- **System**: OpenMP (g++ with `-fopenmp`), MPI, SLURM

## HPC Environment

- **Supercomputer context**: BSC MareNostrum 4
- **Job parameters**: 22 nodes, 2464 tasks, 24h walltime
- **Account**: `cns119`, QoS `gp_resa`
- **Modules**: Load via `module load` before compilation/execution

## Gotchas & Anti-Patterns

❌ **Don't** modify `pb/PhysiBoSSv2/core/` directly (upstream PhysiCell code)  
✅ **Do** customize via `custom_modules/` and XML configs

❌ **Don't** assume simulation IDs are sequential integers  
✅ **Do** parse `variables.txt` for lobe ID (LU/LL/RU/RM/RL)

❌ **Don't** run all rounds in one PhysiBoSS execution  
✅ **Do** use round-based restarts with updated initial conditions

❌ **Don't** manually edit `fensap.nsi.dat` boundary codes  
✅ **Do** use `input_alya.py modify_alya_input()` to programmatically close lobes

## Quick Reference: Key Directories

- `pb/PhysiBoSSv2/` - PhysiBoSS source & configs
- `pb/scripts/` - Python coupling/evaluation scripts
- `alya/` - ALYA input templates & executable
- `experiments/` - Simulation outputs (gitignored, generated at runtime)
- `logs/` - SLURM job logs
- `project_2_plane/` - 3D→2D projection utility

## When Modifying Code

**Adding new cell types**: Update `custom.cpp`, XML configs, and `celltype_dict` in `combined_alveoli.py`

**Changing coupling logic**: Focus on `input_alya.py` and PhysiBoSS `main.cpp` round continuation

**New ALYA scenarios**: Duplicate `lung_surrogate/` or `alveoli/`, update `orchestrator.sh` paths

**Parameter sweeps**: Generate multiple JSON lines in `data/variables.txt`, PhysiBoSS loops over them
