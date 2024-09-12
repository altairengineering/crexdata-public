""""
This is a wrapper for EpiSim.jl

It handles the file writing and reading for executing the model: config files and model state.
We do it like this to get the precompile performance benefits but it has some downsides, mainly:

- we cannot run two instances of the model at the same time
- we need to write the config json file every time we change the params
- we need to write the model state .nc file for every step !!

The alternative is to call functions from EpiSim.jl directly but this also has downsides:

- startup cost every single time we run a step
- we need to change EpiSim.jl to enable this direct access
- we need to marshall data types between python and julia, particularly the model state (big arrays!)
"""

import os, sys
import json
import subprocess
import logging
import pandas as pd
import uuid
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler(sys.stdout))

logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

class EpiSim:
    """
    A wrapper for EpiSim.jl that handles file writing and reading for executing the model.

    This class manages the configuration files and model state, allowing for step-by-step
    execution of the EpiSim model.

    Attributes:
        executable_path (str): Path to the compiled EpiSim executable or Julia script.
        executable_type (str): Either 'compiled' or 'interpreter'.
        instance_folder (str): Folder to store instance-specific data.
        uuid (str): Unique identifier for this model instance.
        model_state_folder (str): Folder to store model state files.
        config_path (str): Path to the current configuration file.
        data_folder (str): Folder containing input data for the model.
        model_state (str): Path to the current model state file.
        backend_engine (str): The backend engine to use for EpiSim ('MMCACovid19Vac' or 'MMCACovid19').

    """

    # location of the compiled EpiSim.jl
    DEFAULT_EXECUTABLE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "episim")
    # entrypoint for running EpiSim.jl by the Julia interpreter. Slower startup time, faster to debug code changes
    DEFAULT_INTERPRETER_PATH = ["julia", os.path.join(os.path.dirname(__file__), "run.jl")]

    DEFAULT_BACKEND_ENGINE = 'MMCACovid19Vac'
    BACKEND_ENGINES = [
        { 'name': 'MMCACovid19Vac', 'description': 'Model with vaccination' },
        { 'name': 'MMCACovid19', 'description': 'Model without vaccination' }
    ]

    def __init__(self, config, data_folder, instance_folder, initial_conditions=None):
        """
        Initialize the MMCACovid19 model wrapper.

        Args:
            config (dict or str): Model configuration as a dictionary or path to a JSON file.
            data_folder (str): Folder containing input data for the model.
            instance_folder (str): Folder to store instance-specific data.
            initial_conditions (str, optional): Path to initial conditions file.

        Raises:
            AssertionError: If required paths do not exist or are invalid.
        """
        assert os.path.exists(data_folder)
        assert os.path.exists(instance_folder)

        self.setup_complete = False
        self.executable_path = None
        self.executable_type = None

        self.instance_folder = instance_folder
        self.uuid = str(uuid.uuid4())
        self.model_state_folder = os.path.join(instance_folder, self.uuid)
        os.makedirs(self.model_state_folder, exist_ok=False)

        config_path = EpiSim.handle_config_input(self.model_state_folder, config)

        self.config_path = config_path
        self.data_folder = data_folder

        self.model_state = initial_conditions
        if initial_conditions:
            # Copy initial conditions to the unique folder
            new_initial_conditions = os.path.join(self.model_state_folder, os.path.basename(initial_conditions))
            shutil.copy(initial_conditions, new_initial_conditions)
            self.model_state = new_initial_conditions

        self.backend_engine = EpiSim.DEFAULT_BACKEND_ENGINE

        logger.info(f"Model wrapper init complete. UUID: {self.uuid}")

    def setup(self, executable_type='interpreter', executable_path=None):
        """
        Set up the execution environment for EpiSim.

        Args:
            executable_type (str): Either 'compiled' or 'interpreter'.
            executable_path (str): Path to the compiled executable or Julia script.

        Raises:
            ValueError: If invalid executable_type or missing executable_path.
        """
        if executable_type not in ['compiled', 'interpreter']:
            raise ValueError("executable_type must be 'compiled' or 'interpreter'")

        if executable_type == 'compiled':
            executable_path = executable_path or EpiSim.DEFAULT_EXECUTABLE_PATH
            if not executable_path:
                raise ValueError("cannot find a valid executable_path for the compiled model")
            assert os.path.exists(executable_path)
            assert os.path.isfile(executable_path)
            assert os.access(executable_path, os.X_OK)
            self.executable_path = [executable_path]
        else:
            # assert that julia interpreter is available
            assert shutil.which("julia"), "Julia interpreter not found"
            self.executable_path = EpiSim.DEFAULT_INTERPRETER_PATH

        self.executable_type = executable_type
        self.setup_complete = True
        logger.info(f"EpiSim setup complete. Type: {self.executable_type}")
        return self

    def _check_setup(self):
        if not self.setup_complete:
            raise RuntimeError("EpiSim not set up. Call setup() first.")

    def step(self, start_date, length_days):
        """
        Run the model for a given number of days starting from a given start date.

        This method updates the config and model state, then calls the simulator.

        Args:
            start_date (str): Start date for the simulation step (format: 'YYYY-MM-DD').
            length_days (int): Number of days to simulate.

        Returns:
            tuple: A tuple containing:
                - str: Path to the updated model state file.
                - str: The next start date after this step.
        """
        self._check_setup()
        end_date = date_addition(start_date, length_days - 1)

        logger.debug(f"Running model from {start_date} to {end_date}")
        self.run_model(
            length_days=length_days,
            start_date=start_date,
            end_date=end_date,
            model_state=self.model_state,
        )

        self.model_state = self.model_state_filename(end_date)

        logger.debug(f"Step complete")
        return self.model_state, date_addition(end_date, 1)

    def update_config(self, config):
        self.config_path = self.handle_config_input(self.model_state_folder, config)

    def run_model(self, override_config=None, override_model_state=None):
        """
        Run the compiled model for a specific time period.

        Args:
            length_days (int): Number of days to simulate.
            start_date (str): Start date for the simulation (format: 'YYYY-MM-DD').
            end_date (str): End date for the simulation (format: 'YYYY-MM-DD').
            model_state (str, optional): Path to the initial model state file.

        Returns:
            str: Output from the model execution.

        Raises:
            RuntimeError: If the model execution fails.
        """
        self._check_setup()

        cmd = list(self.executable_path)
        cmd.extend(["-e", self.backend_engine, "run"])  # Use the selected backend engine
        cmd.extend(["--config", self.config_path])
        cmd.extend(["--data-folder", self.data_folder])
        cmd.extend(["--instance-folder", self.model_state_folder])

        if override_model_state:
            cmd.extend(["--initial-conditions", override_model_state])

        if override_config and isinstance(override_config, dict):
            if override_config['save_time_step']:
                # save the model state at a specific time step
                cmd.extend(["--export-compartments-time-t", str(override_config['save_time_step'])])
            if override_config['start_date']:
                cmd.extend(["--start-date", override_config['start_date']])
            if override_config['end_date']:
                cmd.extend(["--end-date", override_config['end_date']])

        cmdstr = " ".join(cmd)
        logger.debug(f"Running command:\n{cmdstr}")
        with open('episimlogs.txt', 'w') as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Model execution failed: {result.stdout}")

        return self.uuid, result.stdout

    def model_state_filename(self, end_date):
        return os.path.join(self.model_state_folder, "output", f"compartments_t_{end_date}.h5")

    def update_model_state(self, end_date):
        self.model_state = self.model_state_filename(end_date)
        return self

    def set_backend_engine(self, engine):
        """
        Set the backend engine for EpiSim.

        Args:
            engine (str): The backend engine to use ('MMCACovid19Vac' or 'MMCACovid19').

        Raises:
            ValueError: If an invalid engine is provided.
        """
        if engine not in [e['name'] for e in EpiSim.BACKEND_ENGINES]:
            raise ValueError(f"Invalid backend engine {engine}. Choose 'MMCACovid19Vac' or 'MMCACovid19'.")
        self.backend_engine = engine
        logger.info(f"Backend engine set to: {self.backend_engine}")
        return self

    @staticmethod
    def handle_config_input(model_state_folder, config):
        """
        Process the configuration input and save it to a file.

        Args:
            model_state_folder (str): Folder to save the configuration file.
            config (dict or str): Configuration as a dictionary or path to a JSON file.

        Returns:
            str: Path to the processed configuration file.

        Raises:
            ValueError: If the config input is invalid.
        """
        if isinstance(config, dict):
            # write our own config file for the model to use
            config_path = os.path.join(model_state_folder, 'config_auto_py.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        elif isinstance(config, str) and os.path.exists(config):
            config_path = os.path.join(model_state_folder, os.path.basename(config))
            shutil.copy(config, config_path)
        else:
            raise ValueError(f"Invalid config: {config}")

        logger.debug(f"Using config at: {config_path}")
        return config_path

# MMCACovid19 class end


######################
# utils
######################

def date_addition(start_date, length_days):
    """Calculate the end date given a start date and number of days."""
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=length_days)
    return end.strftime('%Y-%m-%d')

def pardir(): return os.path.join(os.path.dirname(__file__), "..")


######################
# example usage
######################

def run_model_example():
    executable_path = os.path.join(pardir(), "episim")

    initial_conditions = os.path.join(pardir(), "models/mitma/initial_conditions.nc")

    # read the config file sample to dict
    with open(os.path.join(pardir(), "models/mitma/config.json"), 'r') as f:
        config = json.load(f)
    
    data_folder = os.path.join(pardir(), "models/mitma")
    instance_folder = os.path.join(pardir(), "runs")

    model = EpiSim(
        config, data_folder, instance_folder, initial_conditions
    )
    
    # Set up with compiled executable
    # model.setup(executable_type='compiled', executable_path=os.path.join(pardir(), "episim"))
    
    # Or set up with Julia interpreter
    model.setup(executable_type='interpreter', executable_path=os.path.join(pardir(), "run.jl"))

    logger.info("Running")
    output = model.run_model(length_days=1, start_date="2023-01-01", end_date="2023-01-02")
    logger.info(output)
    logger.info("Example done")

def agent_flow_example():
    "Run steps and update the policy"
    executable_path = os.path.join(pardir(), "episim")

    initial_conditions = os.path.join(pardir(), "models/mitma/initial_conditions.nc")

    # read the config file sample to dict
    with open(os.path.join(pardir(), "models/mitma/config.json"), 'r') as f:
        config = json.load(f)

    data_folder = os.path.join(pardir(), "models/mitma")
    instance_folder = os.path.join(pardir(), "runs")

    model = EpiSim(
        config, data_folder, instance_folder, initial_conditions
    )
    
    # Set up with compiled executable
    # model.setup(executable_type='compiled', executable_path=os.path.join(pardir(), "episim"))
    
    # Or set up with Julia interpreter
    model.setup(executable_type='interpreter', executable_path=os.path.join(pardir(), "run.jl"))

    logger.debug("debug Model wrapper init complete")

    start_date="2023-01-01"
    logger.info(f"First date: {start_date}")
    current_date = start_date
    for i in range(10):
        new_state, next_date = model.step(start_date=current_date, length_days=10)

        # update the policy
        # increase the level of lockdown by 5% at each iteration
        config["NPI"]["κ₀s"] = [ config["NPI"]["κ₀s"][0] * (1 - 0.05) ]
        model.update_config(config)

        logger.debug(f"Iteration {i+1} - Model state: {new_state}")
        logger.info(f"Iteration {i+1} - Next date: {next_date}")
        current_date = next_date

    logger.info("Example done")


if __name__ == "__main__":
    agent_flow_example()