# Multi-Scale RapidMiner Workflows

This folder contains RapidMiner workflows for multi-scale data analysis, modeling, and simulation control in the Health Emergency Use Case.

## Workflows Overview

### pb_launch.rmp
RapidMiner process for workflow initialization and PhysiBoSS simulation launch.

**Purpose:** Initializes the multi-scale simulation environment and launches the PhysiBoSS (Physiboss) cellular simulation engine.

**Key Functions:**
- Setup and configuration of simulation parameters
- Initialization of the PhysiBoSS framework
- Launching the cellular-level simulation
- Preparation of input data for downstream analysis

### pb_monitor.rmp
RapidMiner process for monitoring PhysiBoSS simulation with Kafka integration.

**Purpose:** Real-time monitoring and data streaming of active PhysiBoSS simulations using Apache Kafka.

**Key Functions:**
- Continuous monitoring of simulation progress
- Real-time data collection from running simulations
- Kafka producer/consumer integration for distributed data streaming
- Performance metrics tracking and logging
- Data buffering and aggregation

### pb_intervention.rmp
RapidMiner process for intervening in and controlling active PhysiBoSS simulations.

**Purpose:** Dynamically modify simulation parameters and apply interventions during runtime.

**Key Functions:**
- Intervention parameter application to running simulations
- Dynamic modification of cellular behavior rules
- Simulation control and state management
- Response to external triggers or conditions

### VirtualMachine.rmp
RapidMiner virtual machine configuration and PhysiBoSS-ALYA initialization.

**Purpose:** Setup and configuration of the virtual machine environment for the coupled PhysiBoSS-ALYA simulation framework.

**Key Functions:**
- Virtual machine initialization
- Environment configuration and resource allocation
- Integration setup for PhysiBoSS and ALYA (fluid dynamics) coupling
- System dependency verification and installation

## Usage

1. **Initialize Simulation:**
   - Open `pb_launch.rmp` in RapidMiner
   - Configure simulation parameters as needed
   - Execute to initialize the PhysiBoSS simulation

2. **Monitor Simulation:**
   - Open `pb_monitor.rmp` in RapidMiner
   - Connect to the running simulation via Kafka
   - Monitor real-time data streams and performance metrics

3. **Apply Interventions:**
   - Open `pb_intervention.rmp` in RapidMiner
   - Define intervention parameters
   - Apply interventions to the active simulation

4. **System Setup:**
   - Run `VirtualMachine.rmp` to configure the computing environment
   - Ensure all dependencies are properly installed

## Requirements

- RapidMiner Studio (compatible version)
- Apache Kafka (for monitoring workflows)
- PhysiBoSS simulation framework
- Python environment (for supporting utilities)
- ALYA fluid dynamics solver (for coupled simulations)

## Integration

These workflows are part of the larger Health Emergency Use Case multi-scale modeling framework and are designed to work in conjunction with:
- PhysiBoSS cellular-level simulations
- ALYA computational fluid dynamics
- Kafka-based data streaming infrastructure
- External analysis and visualization tools

## Documentation

For detailed information about individual workflow components and parameters, refer to the inline documentation within each RapidMiner process file.

## Related Resources

- Parent folder: [../README.md](../README.md)
- COVID19-EMEWS-PhysiBoSS multi-scale simulation framework
- CREXDATA project documentation
