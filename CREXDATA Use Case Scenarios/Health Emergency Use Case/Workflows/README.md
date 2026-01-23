# Health Emergency Use Case - Workflows

This folder contains various computational workflows and simulations for the Health Emergency Use Case scenarios.

## Folder Structure

### COVID19-emews-physiboss/
Multi-scale simulation of COVID-19 progression using PhysiBoSS and EMEWS framework.
- **code/PhysiBoSSv2/**: PhysiBoSS modeling engine with cellular behavior models
- **alya_files/**: Domain and particle configuration files for different deposition scenarios (depo1-depo4)
- **python/**: Python utilities including genetic algorithms (deap_ga.py), parameter conversion (json2xml.py)
- **R/**: R scripts and utilities
- **scripts/**: Analysis and execution scripts including growth models, simulation verification, and data processing
- **swift/**: Swift-T scripts for distributed execution and parameter sweeping
- **data/**: Simulation data files including parameter combinations
- **ext/**: External tools including EQ-Py framework for workflow integration

### episim-emews/
EMEWS-based workflow integration for epidemic simulation.

### episim-models/
Epidemic simulation models and configurations.

### episim-rl/
Reinforcement learning approaches for epidemic simulation and intervention strategies.

### multiscale-rapidminer-workflows/
RapidMiner workflows for multi-scale data analysis and modeling.
- **pb_intervention.rmp**: RapidMiner process for intervening a PB simulation
- **pb_launch.rmp**: RapidMiner process for workflow initialization
- **pb_monitor.rmp**: RapidMiner process for monitoring PB simulation with kafka
- **VirtualMachine.rmp**: RapidMiner virtual machine configuration and initialization of PB-ALYA

## Usage

Each subfolder contains independent workflow implementations that can be executed based on the specific analysis or simulation needs. Refer to the individual folder READMEs and documentation files for detailed instructions on running each workflow.
