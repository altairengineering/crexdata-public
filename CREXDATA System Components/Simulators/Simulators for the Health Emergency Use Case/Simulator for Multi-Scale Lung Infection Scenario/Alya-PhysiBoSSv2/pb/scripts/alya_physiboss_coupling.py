#!/usr/bin/env python3
"""
alya_physiboss_coupling.py

Unified script for ALYA-PhysiBoSS bidirectional coupling in multi-round simulations.

This module handles:
- Parsing ALYA outputs (lung and alveoli deposition)
- Generating PhysiBoSS input parameters
- Evaluating PhysiBoSS simulation results (tissue damage/infection)
- Modifying ALYA inputs based on infected lobes
- Coordinating round-based simulation workflow

Usage:
    # Round 0: Parse ALYA outputs and generate PhysiBoSS parameters
    python alya_physiboss_coupling.py prepare <lung_depo_file> <alveoli_depo_file> <round> <output_json>
    
    # After PhysiBoSS runs: Combine outputs and evaluate infection status
    python alya_physiboss_coupling.py evaluate <turbine_output> <round>
    
    # Before next ALYA lung run: Modify input to close infected lobes
    python alya_physiboss_coupling.py modify_lung <turbine_output> <round> <alya_lung_dir>
    
    # Before next PhysiBoSS runs: Prepare inputs for next round
    python alya_physiboss_coupling.py prepare_next <turbine_output> <round> <lung_depo_file> <alveoli_depo_file>
"""

import sys
import os
import json
import re
import time
from glob import glob
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import numpy as np
import pandas as pd


# =============================================================================
# ALYA Output Parsing
# =============================================================================

def parse_alveoli_file(file_path, positions):
    """
    Parse ALYA lung deposition output to extract MOI per lobe.
    
    Args:
        file_path: Path to lung_depo.out file
        positions: Path to alveoli deposition CSV file
        
    Returns:
        List of dictionaries with parameters for each simulation instance
    """
    region_map = {
        "left sup": "LU",
        "left inf": "LL",
        "right sup": "RU",
        "right mid": "RM",
        "right inf": "RL"
    }
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            match = re.search(r'\(FREEZE\) type partice=\s*(\d+)', line)
            if match:
                particle_value = int(match.group(1))
            id = 1
            for region, id_value in region_map.items():
                if f"number of deposited particle in {region}" in line:
                    next_line = lines[i + 1].strip()
                    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", next_line)
                    if match:
                        moi_value = float(match.group())
                        data.append({
                            "user_parameters.simulation_id": id,
                            "user_parameters.particle_type": particle_value,
                            "user_parameters.alveoli_id": id_value,
                            "user_parameters.multiplicity_of_infection": round(moi_value, 2),
                            "user_parameters.input_virion_alya": positions
                        })
                id += 1
    return data


def get_lung_data(lung_path):
    """
    Parse lung-scale deposition file to get MOI per lobe.
    
    Args:
        lung_path: Path to lung_depo.out file
        
    Returns:
        Dictionary mapping lobe IDs to MOI values
    """
    region_map = {
        "left sup": "LU",
        "left inf": "LL",
        "right sup": "RU",
        "right mid": "RM",
        "right inf": "RL",
        "wall": "wall"
    }
    data = {}
    with open(lung_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            match = re.search(r'\(FREEZE\) type partice=\s*(\d+)', line)
            if match:
                particle_value = int(match.group(1))
            id = 1
            for region, id_value in region_map.items():
                if f"number of deposited particle in {region}" in line:
                    next_line = lines[i + 1].strip()
                    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", next_line)
                    if match:
                        data[id_value] = float(match.group())
                id += 1
    print(f"Lung MOI data: {data}")
    return data


# =============================================================================
# PhysiBoSS Output Evaluation
# =============================================================================

def convert_mat_to_csv(instance_folder):
    """
    Convert PhysiBoSS .mat outputs to CSV timeseries.
    
    Args:
        instance_folder: Path to simulation instance directory
    """
    celltype_dict = {
        0: "default",
        1: "lung_epithelium",
        2: "immune",
        3: "CD8_Tcell",
        4: "macrophage",
        5: "neutrophil",
        6: "DC",
        7: "CD4_Tcell",
        8: "fibroblast"
    }
    cm_dic = {4: "alive", 100: "apoptotic", 101: "necrotic"}
    output_data = instance_folder 
    data = []

    index = 0
    for filename in sorted(os.listdir(output_data)):
        if filename.endswith("cells_physicell.mat") and filename.startswith("output"):
            file_path = os.path.join(output_data, filename)
        else:
            continue
        instance = {}

        mcds = loadmat(file_path)
        ct = mcds['cells'][5]
        cp = mcds['cells'][6]
        vir_arr = mcds['cells'][90]
        all_cells = len(ct)
        instance["timepoint"] = index * 30  # Time in minutes (assuming 30-minute intervals)
        instance['num_all_cells'] = all_cells

        unique_types = np.unique(ct)
        for cell_type in unique_types:
            type_indices = [i for i, t in enumerate(ct) if t == cell_type]
            celltype_name = celltype_dict.get(cell_type, f"type_{cell_type}")
            instance[f'num_total_{celltype_name}'] = len(type_indices)
            type_phases = [cp[i] for i in type_indices]
            unique_phases, counts = np.unique(type_phases, return_counts=True)
            count_dict = dict(zip(unique_phases, counts))
            instance[f'num_alive_{celltype_name}'] = count_dict.get(6, 0)
            instance[f'num_apoptotic_{celltype_name}'] = count_dict.get(100, 0)
            instance[f'num_necrotic_{celltype_name}'] = count_dict.get(101, 0)
            infected = [vir_arr[i] for i in type_indices if vir_arr[i] >= 1]
            instance[f'num_infected_{celltype_name}'] = len(infected)
        data.append(instance)
        index += 1
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_data, "timeseries_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Converted to CSV: {csv_path}")


def classify_alveolus_from_counts(row_init, row_final, row_median, immune_threshold=150):
    """
    Classify alveolus health based on epithelial cell counts.
    
    Args:
        row_init: Initial timepoint row from CSV
        row_final: Final timepoint row from CSV
        row_median: Median timepoint row from CSV
        immune_threshold: Threshold for immune cell count
        
    Returns:
        tuple: (tag, metrics_dict) where tag is 0=healthy, 1=dysfunctional
    """
    init_alive = row_init['num_alive_lung_epithelium']
    final_alive = row_final['num_alive_lung_epithelium']
    total_final = row_final['num_total_lung_epithelium']
    median_alive = row_median['num_alive_lung_epithelium']
    
    # Basic derived values
    alive_ratio = median_alive / init_alive if init_alive > 0 else 0
    final_apop = row_final['num_apoptotic_lung_epithelium']
    final_necro = row_final['num_necrotic_lung_epithelium']
    death_ratio = (final_apop + final_necro) / total_final if total_final > 0 else 1
    alive_fraction = final_alive / total_final if total_final > 0 else 0
    infected_ratio = row_final['num_infected_lung_epithelium'] / init_alive if init_alive > 0 else 1
    alive_median = median_alive / init_alive if init_alive > 0 else 0
    immune_alive = sum([row_final[col] for col in row_final.index if "num_alive_" in col and "lung_epithelium" not in col])
    
    print(f"Alive ratio: {alive_ratio}")
    
    # Rule-based tagging
    dysfunctional = (
        alive_ratio < 0.5 or
        death_ratio > 0.4 or
        alive_fraction < 0.4 or
        immune_alive > immune_threshold or
        infected_ratio > 0.2
    )

    return int(dysfunctional), {
        "alive_ratio": alive_ratio,
        "death_ratio": death_ratio,
        "alive_fraction": alive_fraction,
        "infected_ratio": infected_ratio,
        "immune_alive": immune_alive,
        "alive_median": alive_median,
    }


def evaluate_epithelial_cells(csv_path):
    """
    Evaluate epithelial cell health from timeseries CSV.
    
    Args:
        csv_path: Path to timeseries_data.csv
        
    Returns:
        tuple: (tag, metrics)
    """
    df = pd.read_csv(csv_path)
    print(f"Evaluating CSV: {csv_path}")
    if df.empty:
        print("No data found in the CSV file.")
        return 1, {}
    
    median_idx = int(len(df) / 2)
    tag, metrics = classify_alveolus_from_counts(df.iloc[0], df.iloc[-1], df.iloc[median_idx])
    return tag, metrics


def wait_for_round_end(round_path, timeout=3600, interval=10):
    """
    Wait for a simulation round to complete and evaluate results.
    
    Args:
        round_path: Path to round directory (e.g., sim_1/round_0/)
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds
        
    Returns:
        bool: True when round completes successfully
    """
    waited = 0
    while True:
        if os.path.exists(os.path.join(round_path, "final.csv")):
            print(f"Round ended: {round_path}")
            convert_mat_to_csv(round_path)
            tag, metrics = evaluate_epithelial_cells(os.path.join(round_path, "timeseries_data.csv"))
            with open(os.path.join(round_path, "tag.txt"), "w") as f:
                f.write(f"{tag},,{metrics}")
            return True
        
        if waited >= timeout:
            raise TimeoutError(f"Timeout waiting for final.csv in {round_path}")
        time.sleep(interval)
        waited += interval


# =============================================================================
# ALYA Input Modification
# =============================================================================

# Lobe code mappings
CODE_MAP = {
    31: "0.77219341E-004",
    32: "0.99817803E-004",
    33: "0.88628529E-004",
    34: "0.39321860E-004",
    35: "0.10301267E-003"     
}

CODE_TO_LOBE = {
    31: "LU",
    32: "LL",
    33: "RU",
    34: "RM",
    35: "RL"
}


def read_fensap_boundary_last_time(file_path, filter_codes=None):
    """
    Read the last Time block from a fensap-boundary.nsi.dat file.
    
    Args:
        file_path: Path to fensap-boundary.nsi.dat file
        filter_codes: Optional list of codes to filter
        
    Returns:
        Dictionary mapping code -> last column float value
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find indices of lines that mark a Time block
    time_indices = [i for i, L in enumerate(lines) if L.strip().startswith('#') and 'Time' in L]
    if not time_indices:
        raise ValueError(f"No Time block found in {file_path}")

    # Start parsing immediately after the last Time header
    start_idx = time_indices[-1] + 1

    # Regex to capture: code, col2, col3, col4 (we need col4)
    data_re = re.compile(r'^\s*(\d+)\s+([\d\.+\-EeDd]+)\s+([\d\.+\-EeDd]+)\s+([\d\.+\-EeDd]+)')
    result = {}
    for L in lines[start_idx:]:
        if not L.strip():
            break
        m = data_re.match(L)
        if not m:
            break
        code = int(m.group(1))
        last_col_str = m.group(4).replace('D', 'E').replace('d', 'e')
        try:
            last_col = float(last_col_str)
        except ValueError:
            continue
        if filter_codes is None or code in filter_codes:
            result[code] = last_col

    return result


def modify_alya_input(turbine_output, round_num, alya_folder):
    """
    Modify ALYA input file to close infected lobes.
    
    Args:
        turbine_output: Experiment output directory
        round_num: Current round number
        alya_folder: ALYA lung directory for this round
    """
    prev_round = int(round_num) - 1
    print(f"Modifying ALYA input for round {round_num} in {alya_folder}")
    
    pb_input = os.path.join(turbine_output, f"round_{prev_round}.csv")
    print(f"Reading PhysiBoSS results from {pb_input}")
    lobes = pd.read_csv(pb_input, header=0)
    print(lobes)
    
    # Get codes for lobes to close (tag == 1)
    codes = lobes[lobes["tag"] == 1]["sim_id"].values
    print(f"Lobes to close (tag==1): {codes}")
    
    fensap_path = os.path.join(alya_folder, "fensap.nsi.dat")
    with open(fensap_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        print(f"Processing line: {line.strip()}")
        
        if len(parts) >= 2:
            try:
                code = parts[-1]
                print(f"Processing line for code {code}: {line.strip()}")
                if code in codes and parts[1] == "000":
                    print(f"Modifying line for code {code}: {line.strip()}")
                    # Replace boundary code from 000 to 111 (closed)
                    line = line.replace("000", "111", 1)
            except ValueError:
                pass
        
        # Adjust flow rate based on closed lobes
        if "FLOW_RATE" in parts:
            lobe_to_code = {v: k for k, v in CODE_TO_LOBE.items()}
            active_lobes = lobes[lobes['tag'] == 1]['sim_id'].tolist()
            
            if int(round_num) == 1:
                # First round: use predefined code map
                subtract_sum = sum(float(CODE_MAP[lobe_to_code[lobe]]) for lobe in active_lobes)
                original_flow_rate = float(parts[2].strip(","))
                new_flow_rate = original_flow_rate - subtract_sum
                line = line.replace(parts[2], f"{new_flow_rate:.8f}", 1)
                print(f"Updated FLOW_RATE: {new_flow_rate}")
                
            elif int(round_num) > 1:
                # Subsequent rounds: read from previous boundary file
                boundary_file = f"/fensap-boundary.nsi.set"
                fensap_boundary_path = os.path.join(turbine_output, f"alya/round_{prev_round}/lung{boundary_file}")
                print(f"Reading boundary data from: {fensap_boundary_path}")
                res = read_fensap_boundary_last_time(fensap_boundary_path, filter_codes=None)
                print(f"Boundary data: {res}")
                
                subtract_sum = sum(float(res[lobe_to_code[lobe]]) for lobe in active_lobes)
                original_flow_rate = float(parts[2].strip(","))
                new_flow_rate = original_flow_rate - subtract_sum
                line = line.replace(parts[2], f"{new_flow_rate:.8f}", 1)
                print(f"Updated FLOW_RATE: {new_flow_rate}")

        new_lines.append(line)
    
    with open(fensap_path, "w") as f:
        f.writelines(new_lines)
    print(f"Modified fensap.nsi.dat in {alya_folder} for round {round_num}")


# =============================================================================
# Workflow Coordination
# =============================================================================

def get_sim_id(sim_folder):
    """
    Extract simulation/lobe ID from variables.txt.
    
    Args:
        sim_folder: Path to sim_N directory
        
    Returns:
        Lobe ID string (e.g., "LU", "RL")
    """
    p = os.path.join(sim_folder, "variables.txt")
    print(f"Getting simulation ID from: {sim_folder}")
    with open(p, "r") as f:
        line = f.readline().strip()
        parts = line.split(" ")
        return parts[1] if len(parts) > 1 else "unknown"


def combine_physiboss_outputs(turbine_output, round_num):
    """
    Combine PhysiBoSS outputs from all lobes and generate round summary.
    
    Args:
        turbine_output: Experiment output directory
        round_num: Current round number
    """
    print(f"Combining outputs for round {round_num} in {turbine_output}")
    tags = {}
    
    for folder in sorted(os.listdir(turbine_output)):
        if folder.startswith("sim"):
            sim_folder = os.path.join(turbine_output, folder)
            sim_round = os.path.join(sim_folder, f"round_{round_num}/")
            print(f"Processing simulation round: {sim_round}")

            wait_for_round_end(sim_round)
            tag_file = os.path.join(sim_round, "tag.txt")
            if os.path.exists(tag_file):
                with open(tag_file, "r") as f:
                    tag = f.read().strip().split(",")[0]
                    sim_id = get_sim_id(sim_folder)
                    print(f"Simulation {sim_id}: tag = {tag}")
                    tags[sim_id] = tag

    # Write round summary CSV
    output_csv = os.path.join(turbine_output, f"round_{round_num}.csv")
    with open(output_csv, "w") as f:
        f.write("sim_id,tag\n")
        for sim_id, tag in tags.items():
            f.write(f"{sim_id},{tag}\n")
    print(f"Written round summary: {output_csv}")


def prepare_next_round_inputs(turbine_output, round_num, lung_depo_file, alveoli_depo_file):
    """
    Prepare PhysiBoSS inputs for the next round based on ALYA outputs.
    
    Args:
        turbine_output: Experiment output directory
        round_num: Current round number
        lung_depo_file: Path to lung_depo.out
        alveoli_depo_file: Path to depo.pts-depo.csv
    """
    dist = get_lung_data(lung_depo_file)
    print(f"Lung distribution: {dist}")
    
    prev_round = int(round_num) - 1
    tags = pd.read_csv(os.path.join(turbine_output, f"round_{prev_round}.csv"), header=0)

    for folder in sorted(os.listdir(turbine_output)):
        if folder.startswith("sim"):
            sim_folder = os.path.join(turbine_output, folder)
            sim_round = os.path.join(sim_folder, f"round_{round_num}/")
            lobe_id = get_sim_id(sim_folder)
            print(f"Preparing inputs for lobe {lobe_id}")
            
            # Copy alveoli deposition file
            with open(os.path.join(sim_round, "input_virion_alya.txt"), "w") as f:
                with open(alveoli_depo_file, "r") as depo_file:
                    for line in depo_file:
                        f.write(line)
            
            # Write MOI value
            with open(os.path.join(sim_folder, "moi.txt"), "w") as f:
                f.write(str(dist[lobe_id]))
            
            # Create round tag file
            with open(os.path.join(sim_round, "round.txt"), "w") as f:
                print(f"Writing round.txt for {lobe_id} in {sim_round}")
                tag = tags[tags['sim_id'] == lobe_id]['tag'].values[0]
                f.write(str(tag))
    
    print(f"Prepared inputs for round {round_num}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "prepare":
        # Usage: prepare <lung_path> <positions_path> <round_num> <out_file>
        lung_path = sys.argv[2]
        positions_path = sys.argv[3]
        round_num = int(sys.argv[4])
        
        if round_num == 0:
            out_file = sys.argv[5]
            output_data = parse_alveoli_file(lung_path, positions_path)
            with open(out_file, "w") as json_file:
                for el in output_data:
                    json_data = json.dumps(el)
                    json_file.write(json_data + '\n')
            print(f"Generated PhysiBoSS parameters: {out_file}")
        else:
            print(f"For round > 0, use 'prepare_next' mode")
    
    elif mode == "evaluate":
        # Usage: evaluate <turbine_output> <round>
        turbine_output = sys.argv[2]
        round_num = int(sys.argv[3])
        combine_physiboss_outputs(turbine_output, round_num)
    
    elif mode == "modify_lung":
        # Usage: modify_lung <turbine_output> <round> <alya_folder>
        turbine_output = sys.argv[2]
        round_num = int(sys.argv[3])
        alya_folder = sys.argv[4]
        modify_alya_input(turbine_output, round_num, alya_folder)
    
    elif mode == "prepare_next":
        # Usage: prepare_next <turbine_output> <round> <lung_depo_file> <alveoli_depo_file>
        turbine_output = sys.argv[2]
        round_num = int(sys.argv[3])
        lung_depo_file = sys.argv[4]
        alveoli_depo_file = sys.argv[5]
        prepare_next_round_inputs(turbine_output, round_num, lung_depo_file, alveoli_depo_file)
    
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
