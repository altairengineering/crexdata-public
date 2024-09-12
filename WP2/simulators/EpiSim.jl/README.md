# EpiSim.jl
A julia-based simulator for simulating epidemics. 

Currently, it implements MMCA for simulating an extended SEIR in a metapopulation. 

The package is a stand-alone simulator that allow to run simulations using different engines/models. Currently EpiSim enable the used of two engines:
* MMCAcovid19 Julia package [https://docs.juliahub.com/MMCAcovid19/]([url](https://docs.juliahub.com/MMCAcovid19/))
* 
# Brief history

## MMCAcovid19
This package [MMCAcovid19](https://github.com/jtmatamalas/MMCAcovid19), written in the [Julia](https://julialang.org) language, implements the epidemic model for COVID-19 developed by a group of researchers from [Universitat Rovira i Virgili](https://www.urv.cat) and [Universidad de Zaragoza](http://unizar.es) [[1](#References-1)]. The model makes use of a Microscopic Markov Chain Approach (MMCA) to describe mathematically the dynamics of a so-called metapopulation model of epidemic spreading [[2-4](#References-1)]. 

## MMCA_with_vaccination
The original was later extended by Piergiorgio Castioni to allow modelling of the effect of vaccines, herd immunity and reinfections. Each of the previous compartments has been triplicated to account for the fact that there are vaccinated and unvaccinated individuals. Having received the vaccine changes some of the transition probabilities related to the most negative aspects of the disease, such as transmission, hospitalization and death.

MMCA_with_vaccination [https://github.com/PGcastioni/MMCA_with_vaccination/]([url](https://github.com/PGcastioni/MMCA_with_vaccination/))



## EpiSim.jl

This project is an attempt to extend the functionalities of the previous version. Currently, this version provides a standard configuration format for defining a model and provides a simple set of command line scripts to generate configuration templates and run the model.


### Installing

To install EpiSim.jl, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/EpiSim.jl.git
   cd EpiSim.jl
   ```

2. Run the installation script:
   ```
   ./install.sh [HPC_ACCOUNT_NAME]
   ```
   
   Note: The `HPC_ACCOUNT_NAME` argument is required only when running on BSC HPC systems.

   This is because the compile time is long enough that we need to schedule a slurm job for it.

The compilation process (`install.jl`) creates a single precompiled executable:

- Builds the application in a `build` folder
- Creates a symlink named `episim` in the target directory (default is the current directory)

After installation, you can run EpiSim using the `episim` command.


### Using EpiSim.jl

EpiSim works as a command line frontend to launch simulations. It provides a simple JSON based config format to define an instance of a model. The config format included a common or core sets of parameter and specific parameters required by the different engines supported by EpiSim.jl. An example is given at `models/mitma/config.json`


### The `epiconfig.json` format
```
{
	"simulation": {
		"start_date": "2020-03-10",
		"end_date": "2020-04-15",
		"save_full_output": true,
		"save_time_step": null,
		"output_folder": "output",
		"output_format": "netcdf"
	},
	"data": {
		# Engine specifc
        # path to files with input data, e.g. metapopulation, mobility matrix, etc.
	},
	"epidemic_params": {
		# Engine specifc
        # Epidemic parameters e.g. infection and recovery rates.
    },
	"population_params": {
        # Engine specifc
        # Population parameter, e.g. average contacts, age speciffic contact matrix.
    }
}
```

```bash
episim -e MMCACovid19Vac run -c models/mitma/config.json -d models/mitma -i runs
```

### Using the Python interface

Install dependencies:

```bash
pip install -r py_interface/requirements.txt
```

Run the model in steps:

```python
from epi_sim import EpiSim

executable_path = "./episim"
config_path = "models/mitma/config.json"
data_folder = "models/mitma"
instance_folder = "runs"
initial_conditions = "models/mitma/initial_conditions.nc"

model = MMCACovid19(
	executable_path, config_path, data_folder, instance_folder, initial_conditions
)

new_state, next_date = model.step("2020-03-10", 5)

new_state, next_date = model.step(next_date, 5)
```

See `py_interface/EpiSim.py` for more examples.

<!-- 

## run_simulations.jl

Main script to run model simulations, with multiple configuration options.

Examples:

	julia --project=scripts/ scripts/run_simulations.jl

	# start and end date can be provided in the config.json
	julia --project=scripts/ scripts/run_simulations.jl -i experiments/test20 -d data -c data/config.json

	# or can be provided as command line arguments
	julia --project=scripts/ scripts/run_simulations.jl -d data -c data/config.json -i experiments/test20 --start-date 2020-02-09 --end-date 2020-05-01

	# to export the compartments at a given time t use the option --export-compartments-time-t
	julia --project=scripts/ scripts/run_simulations.jl -d data -c data/config.json -i experiments/test20 --start-date 2020-02-09 --end-date 2020-05-01 --export-compartments-time-t 5

	# to export the full compartments time series (for all dates), use the option --export-compartments
	julia --project=scripts/ scripts/run_simulations.jl -d data -c data/config.json -i experiments/test20 --start-date 2020-02-09 --end-date 2020-05-01 --export-compartments

	# to use the compartments from other simulation as initial conditions, use the option --initial-compartments
	julia --project=scripts/ scripts/run_simulations.jl -d data -c data/config.json -i experiments/test20 --start-date 2020-02-13 --end-date 2020-05-01 --initial-compartments experiments/test20/output/compartments_2020-02-13_10.h5


Usage:
	
	usage: run_simulations.jl -c CONFIG -d DATA-FOLDER -i INSTANCE-FOLDER
	                        [--export-compartments-full]
	                        [--export-compartments-time-t EXPORT-COMPARTMENTS-TIME-T]
	                        [--initial-compartments INITIAL-COMPARTMENTS]
	                        [--start-date START-DATE]
	                        [--end-date END-DATE]


## run_parallel_simulation.jl

Basic script that runs multiple simulation in parallel. 


Note: the <instance_folder> must contain a `params.csv` file, with one set of parameters per row.


Usage:
	
	julia --project=scripts/. scripts/run_parallel_simulation.jl <data_folder> <instance_folder>



## run_parallel_simulation_daily_mobility.jl

Script that runs multiple simulation in parallel. It uses a modified version of the original package, that uses a different mobility matrix (Rij) for each day.

The modified package is in `$ROOT_FOLDER/julia/MMCACovid19custom`

The script looks at `data/config.json` for the keyword `daily_mobility_matrices`, which defines a folder that contains the mobility matrices for each day.

If the mobility matrix for one day is missing, it can be configured whether to use a default matrix, or previous day matrix. 


Usage:
	
	julia --project=scripts/. scripts/run_parallel_simulation_daily_mobility.jl <data_folder> <instance_folder>


Note: the <instance_folder> must contain a `params.csv` file, with one set of parameters per row.



## run_parallel_simulation_seeds.jl

This script is a modified version of `run_parallel_simulation.jl` which uses a different set of seeds for each simulation.


It requires a `seeds.csv` file to be present in the instance/ folder (appart from `params.csv`), with one row per simulation, defining the set of seeds to be used in each simulation.


Usage:
	
	julia --project=scripts/. scripts/run_parallel_simulation_seeds.jl <data_folder> <instance_folder>



## sample_params.py

Sample parameters from a `priors.json` file:

Usage:

	python python/sample_params.py --priors data/priors.json --size 1000 --output experiments/test1000/instance_1/params.csv



## evaluate.py

Evaluates one instance, and outputs a `scores.csv` file, wich is the same as the `params.csv` with an additional `score` column.


Example:
	
	python python/evaluate.py -i <instance_folder> -d <data_folder>

	python python/evaluate.py -i experiments/test1000/instance_1/ -d data/

	# fit with incidence
	python python/evaluate.py -i experiments/test1000/instance_1/ -d data/ --fit incidence

	# fit with deaths
	python python/evaluate.py -i experiments/test1000/instance_1/ -d data/ --fit deaths


	
Usage:

	usage: evaluate.py [-h] --instance-folder INSTANCE_PATH --data-folder DATA_PATH [--config CONFIG_PATH] [--first-day-train FIRST_DAY_TRAIN]
                   [--last-day-train LAST_DAY_TRAIN] [--metric METRIC] [--weights WEIGHTS] [--fit FIT]


## summarize.py

Script that summarizes an instance, and outputs several plots.

Examples:

	python python/summarize.py -i <instance_folder> -d <data_folder>

	# draw main plots
	python python/summarize.py -i experiments/test1000/instance_1/ -d data/
	
	# draw all plots (takes time)
	python python/summarize.py --all -i experiments/test1000/instance_1/ -d data/

	# fit with incidence
	python python/summarize.py -i experiments/test1000/instance_1/ -d data/ --fit incidence

	# fit with deaths
	python python/summarize.py -i experiments/test1000/instance_1/ -d data/ --fit deaths

Usage:

	usage: summarize.py [-h] --instance-folder INSTANCE_PATH --data-folder DATA_PATH [--config CONFIG_PATH] [--output-folder OUTPUT_PATH]
	                    [--simulation SIMULATION] [--first-day-train FIRST_DAY_TRAIN] [--last-day-train LAST_DAY_TRAIN] [--metric METRIC] [--weights WEIGHTS]
	                    [--fit FIT] [--all] [--run RUN]

-->


## References

1. Alex Arenas, Wesley Cota, Jesús Gómez-Gardeñes, Sergio Gómez, Clara Granell, Joan T. Matamalas, David Soriano-Paños and Benjamin Steinegger: Modeling the spatiotemporal epidemic spreading of COVID-19 and the impact of mobility and social distancing interventions, _Physical Review X_ **10** (2020) 041055 ([doi](https://doi.org/10.1103/PhysRevX.10.041055))

2. Sergio Gómez, Alex Arenas, Javier Borge-Holthoefer, Sandro Meloni and Yamir Moreno: Discrete-time Markov chain approach to contact-based disease spreading in complex networks, _Europhysics Letters_ **89** (2010) 38009 ([doi](https://doi.org/10.1209/0295-5075/89/38009))

3. Jesús Gómez-Gardeñes, David Soriano-Paños and Alex Arenas: Critical regimes driven by recurrent mobility patterns of reaction-diffusion processes in networks, _Nature Physics_ **14** (2018) 391–395 ([doi](https://doi.org/10.1101/2020.03.21.20040022))

4. David Soriano-Paños, L. Lotero, Alex Arenas and Jesús Gómez-Gardeñes: Spreading processes in multiplex metapopulations containing different mobility networks, _Physical Review X_ **8** (2018) 031039 ([doi](https://doi.org/10.1103/PhysRevX.8.031039))

