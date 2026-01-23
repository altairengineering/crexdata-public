import sys
import os
import json
import re
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import numpy as np
import pandas as pd

def parse_alveoli_file(file_path, positions):
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
    print(data)
    return data

def get_mat_file_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for filename in root.findall(".//filename"):
        fname = filename.text
        if fname and fname.endswith("_cells_physicell.mat"):
            return os.path.join(os.path.dirname(xml_path), fname)
    raise FileNotFoundError("No _cells_physicell.mat file found in XML.")

def load_cell_data(mat_path):
    mat = loadmat(mat_path)
    for key in mat:
        if not key.startswith("cells") and isinstance(mat[key], np.ndarray):
            arr = mat[key]
            if arr.ndim == 2 and arr.shape[1] > 10:
                return arr
    raise ValueError("Could not find cell data array in .mat file.")

def detect_dead_tissue(cell_data, label_map, dead_phase_indices=[100,101], dead_flags_indices=None, dead_fraction_threshold=0.5):
    n_cells = cell_data.shape[0]
    if n_cells == 0:
        return True
    phase_col = label_map.get("current_phase")
    dead_cells = np.zeros(n_cells, dtype=bool)
    if phase_col is not None:
        dead_cells |= np.isin(cell_data[:, phase_col], dead_phase_indices)
    if dead_flags_indices:
        for idx in dead_flags_indices:
            dead_cells |= (cell_data[:, idx] > 0.5)
    dead_fraction = np.sum(dead_cells) / n_cells
    return dead_fraction > dead_fraction_threshold

def get_label_map_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    label_map = {}
    for label in root.findall(".//label"):
        name = label.text.strip()
        idx = int(label.attrib["index"])
        size = int(label.attrib.get("size", 1))
        if size == 1:
            label_map[name] = idx
        else:
            for i in range(size):
                label_map[f"{name}_{i}"] = idx + i
    return label_map

def wait_for_final_csvs(output_dir, n_simulations, timeout=36000000000000000000, interval=10):
    import time
    waited = 0
    while True:
        count = 0
        for i in range(1, n_simulations+1):
            if os.path.isfile(f"{output_dir}/sim_{i}/final.csv"):
                count += 1
        if count == n_simulations:
            break
        if waited >= timeout:
            raise TimeoutError("Timeout waiting for final.csv files.")
        time.sleep(interval)
        waited += interval

def convert_mat_to_csv(instance_folder):
    # Map cell type codes to names from central.xml
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

    index=0
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
    df.to_csv(output_data + "/timeseries_data.csv", index=False)
    print(f"Converted {output_data + '/timeseries_data.csv'} to CSV format.")
def classify_alveolus_from_counts(row_init, row_final,row_median, immune_threshold=150):
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
    print(alive_ratio)
    # Rule-based tagging
    # dysfunctional = (
    #     alive_ratio < 0.5 or
    #     death_ratio > 0.4 or
    #     alive_fraction < 0.4 or
    #     immune_alive > immune_threshold or
    #     infected_ratio > 0.2
    # )
    dysfunctional = (alive_ratio <= 0.38643)
    return int(dysfunctional), {
        "alive_ratio": alive_ratio,
        "death_ratio": death_ratio,
        "alive_fraction": alive_fraction,
        "infected_ratio": infected_ratio,
        "immune_alive": immune_alive,
        "alive_median": alive_median,
    }

def evaluate_epithelial_cells(csv_path):
    df = pd.read_csv(csv_path)
    print(f"csv path {csv_path}")
    if df.empty:
        print("No data found in the CSV file.")
        return
    df = pd.read_csv(csv_path)
    a = int(len(df)/2)
    print(a)
    tag,metrics = classify_alveolus_from_counts(df.iloc[0], df.iloc[-1],df.iloc[a])
    return tag,metrics

# function for monitoring simulation round ended
def wait_for_round_end(round_path, timeout=3600, interval=10):
    while True:
        if os.path.exists(os.path.join(round_path, "final.csv")):
            print("Round ended.")
            convert_mat_to_csv(round_path)
            tag, metrics = evaluate_epithelial_cells(os.path.join(round_path, "timeseries_data.csv"))
            # alveoli_id = get_alveoli_id_fromxml(os.path.join(round_path, "settings.xml"))
            with open(os.path.join(round_path, "tag.txt"), "w") as f:
                f.write(f"{tag},,{metrics}")
            return True
    

def main():
    if len(sys.argv) < 2:
        print("Usage: python combined_alveoli.py <mode> [args...]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "prepare":
        # Usage: prepare <lung_path> <positions_path> <out_file> <round_num>
        lung_path = sys.argv[2]
        positions_path = sys.argv[3]
        
        round_num = int(sys.argv[4])
        output_data = parse_alveoli_file(lung_path, positions_path)
        if round_num == 0:
            out_file = sys.argv[5]
            with open(out_file, "w") as json_file:
                for el in output_data:
                    json_data = json.dumps(el)
                    json_file.write(json_data + '\n')
        else:
            # this is the case where alya is finished but the round isnt 0
            # prepare <lung_path> <positions_path> <out_file> <round_num> <path_for_flags>


            pass    
            # path_for_flags = sys.argv[6]
            # with open(path_for_flags+"/round.txt", "a") as file:
            #     # positions_path is a file it needs to be read
            #     # copy positions_path contents to round.txt
            #     with open(positions_path, "r") as pos_file:
            #         positions = pos_file.read().strip()
            #     file.write(positions + '\n')
            # moi_value = output_data["user_parameters.multiplicity_of_infection"]
            # with open(path_for_flags+"/moi.txt", "a") as file:
            #     file.write(f"{moi_value}\n")
    elif mode == "wait_and_evaluate":
        # Usage: wait_and_evaluate <turbine_output> <n_simulations> <round>
        turbine_output = sys.argv[2]
        n_simulations = int(sys.argv[3])
        round = int(sys.argv[4])
        wait_for_round_end(turbine_output + f"/round_{round}", timeout=3600, interval=10)
        # simulation number go to setting and get code
        # sims = []
        # for i in range(1, n_simulations+1):
        #     # i is the simulation id
        #     convert_mat_to_csv(f"{turbine_output}/sim_{i}")
        #     tag,metrics = evaluate_epithelial_cells(f"{turbine_output}/sim_{i}/round_{round}/timeseries_data.csv")
        #     if tag == 1:
        #         with open(f"{turbine_output}/sim_{i}/round_{round}/kill.txt", "w") as f:
        #             f.write(metrics)
        #         sims[i] = {
        #             "tag": tag,
        #             "metrics": metrics
        #             # maybe it need to retrieve the alveoli_id from the setting.xml
        #         }
        #         # also need to inform here alya which lobe should close.
        # with open(f"{turbine_output}/round_{round}.json", "w") as f:
        #     json.dump(sims, f)

    else:
        print("Unknown mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
