"""
helper_functions.py

Object-oriented helpers used by input_alya.py and combined_alveoli.py.

Classes:
 - AlveoliParser: parse ALYA alveoli/lung output for deposition and MOI values
 - LungDataReader: convenience wrapper around lung parsing
 - MatConverter: convert _cells_physicell.mat outputs into CSV timeseries
 - Evaluator: classify alveolus health from timeseries CSVs
 - RoundMonitor: wait for a round to finish and run conversion+evaluation

The implementations closely follow the behavior used in the existing scripts,
but are packaged as small classes so other scripts can import and reuse them.
"""
import os
import re
import time
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat


class AlveoliParser:
    """Parse ALYA text output to extract depositional MOI and positions.

    parse(file_path, positions) -> list of dict (for writing to params/JSON)
    """
    region_map = {
        "left sup": "LU",
        "left inf": "LL",
        "right sup": "RU",
        "right mid": "RM",
        "right inf": "RL"
    }

    @classmethod
    def parse(cls, file_path: str, positions: str) -> List[Dict]:
        data = []
        particle_value = None
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, 'r') as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines):
            line = raw.strip()
            m = re.search(r'\(FREEZE\) type partice=\s*(\d+)', line)
            if m:
                particle_value = int(m.group(1))
            for region, id_value in cls.region_map.items():
                if f"number of deposited particle in {region}" in line:
                    # next line contains numeric MOI
                    if i + 1 < len(lines):
                        next_line = lines[i+1].strip()
                        m2 = re.search(r"[-+]?[0-9]*\.?[0-9]+", next_line)
                        if m2:
                            moi_value = float(m2.group())
                            data.append({
                                "user_parameters.simulation_id": 1,
                                "user_parameters.particle_type": particle_value,
                                "user_parameters.alveoli_id": id_value,
                                "user_parameters.multiplicity_of_infection": round(moi_value, 2),
                                "user_parameters.input_virion_alya": positions
                            })
        return data


class LungDataReader:
    """Read lung output and map regions to short codes (LU, LL, RU, RM, RL).

    get_lung_data(file_path) -> dict{ code: float }
    """
    region_map = {
        "left sup": "LU",
        "left inf": "LL",
        "right sup": "RU",
        "right mid": "RM",
        "right inf": "RL",
        "wall": "wall"
    }

    @classmethod
    def get_lung_data(cls, lung_path: str) -> Dict[str, float]:
        data: Dict[str, float] = {}
        if not os.path.isfile(lung_path):
            raise FileNotFoundError(lung_path)
        with open(lung_path, 'r') as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines):
            line = raw.strip()
            m = re.search(r'\(FREEZE\) type partice=\s*(\d+)', line)
            if m:
                particle_value = int(m.group(1))  # currently unused but preserved
            for region, id_value in cls.region_map.items():
                if f"number of deposited particle in {region}" in line:
                    if i + 1 < len(lines):
                        next_line = lines[i+1].strip()
                        m2 = re.search(r"[-+]?[0-9]*\.?[0-9]+", next_line)
                        if m2:
                            data[id_value] = float(m2.group())
        return data


class MatConverter:
    """Convert Physicell .mat cell files into a simple timeseries CSV used by evaluation.

    convert_mat_to_csv(folder) -> writes `timeseries_data.csv` into folder
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

    @classmethod
    def convert_mat_to_csv(cls, instance_folder: str) -> str:
        output_data = instance_folder
        data = []
        index = 0
        if not os.path.isdir(output_data):
            raise FileNotFoundError(output_data)
        for filename in sorted(os.listdir(output_data)):
            if filename.endswith("cells_physicell.mat") and filename.startswith("output"):
                file_path = os.path.join(output_data, filename)
            else:
                continue
            mcds = loadmat(file_path)
            # Heuristic indexing used in previous script
            ct = mcds['cells'][5]
            cp = mcds['cells'][6]
            vir_arr = mcds['cells'][90]
            all_cells = len(ct)
            instance = {}
            instance["timepoint"] = index * 30
            instance['num_all_cells'] = all_cells
            unique_types = np.unique(ct)
            for cell_type in unique_types:
                type_indices = [i for i, t in enumerate(ct) if t == cell_type]
                celltype_name = cls.celltype_dict.get(cell_type, f"type_{cell_type}")
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
        out_csv = os.path.join(output_data, "timeseries_data.csv")
        df.to_csv(out_csv, index=False)
        return out_csv


class Evaluator:
    """Evaluate epithelial health and provide a tag and metrics.

    classify_alveolus_from_counts(row_init, row_final, row_median) -> (tag, metrics)
    evaluate_epithelial_cells(csv_path) -> (tag, metrics)
    """

    @staticmethod
    def classify_alveolus_from_counts(row_init, row_final, row_median, immune_threshold=150):
        init_alive = row_init.get('num_alive_lung_epithelium', 0)
        final_alive = row_final.get('num_alive_lung_epithelium', 0)
        total_final = row_final.get('num_total_lung_epithelium', 0)
        median_alive = row_median.get('num_alive_lung_epithelium', 0)
        alive_ratio = median_alive / init_alive if init_alive > 0 else 0
        final_apop = row_final.get('num_apoptotic_lung_epithelium', 0)
        final_necro = row_final.get('num_necrotic_lung_epithelium', 0)
        death_ratio = (final_apop + final_necro) / total_final if total_final > 0 else 1
        alive_fraction = final_alive / total_final if total_final > 0 else 0
        infected_ratio = row_final.get('num_infected_lung_epithelium', 0) / init_alive if init_alive > 0 else 1
        alive_median = median_alive / init_alive if init_alive > 0 else 0
        immune_alive = sum([row_final[col] for col in row_final.keys() if "num_alive_" in col and "lung_epithelium" not in col])
        dysfunctional = (
            alive_ratio < 0.4
        )
        return int(dysfunctional), {
            "alive_ratio": alive_ratio,
            "death_ratio": death_ratio,
            "alive_fraction": alive_fraction,
            "infected_ratio": infected_ratio,
            "immune_alive": immune_alive,
            "alive_median": alive_median,
        }

    @classmethod
    def evaluate_epithelial_cells(cls, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("No data found in the CSV file.")
        a = int(len(df) / 2)
        tag, metrics = cls.classify_alveolus_from_counts(df.iloc[0].to_dict(), df.iloc[-1].to_dict(), df.iloc[a].to_dict())
        return tag, metrics


class RoundMonitor:
    """Monitor a round folder for completion and run conversion + evaluation.

    wait_for_round_end(round_path, timeout=3600, interval=10) -> writes tag.txt and returns True
    """

    def __init__(self, converter: Optional[MatConverter] = None, evaluator: Optional[Evaluator] = None):
        self.converter = converter or MatConverter
        self.evaluator = evaluator or Evaluator

    def wait_for_round_end(self, round_path: str, timeout: int = 3600, interval: int = 10) -> bool:
        waited = 0
        while True:
            final_csv = os.path.join(round_path, "final.csv")
            if os.path.exists(final_csv):
                # convert mat outputs to timeseries CSV
                try:
                    self.converter.convert_mat_to_csv(round_path)
                except Exception:
                    # conversion failure should not crash monitor; still attempt evaluation
                    pass
                try:
                    tag, metrics = self.evaluator.evaluate_epithelial_cells(os.path.join(round_path, "timeseries_data.csv"))
                except Exception:
                    tag, metrics = 0, {}
                with open(os.path.join(round_path, "tag.txt"), "w") as f:
                    f.write(f"{tag},,{metrics}")
                return True
            if waited >= timeout:
                raise TimeoutError("Timeout waiting for final.csv files.")
            time.sleep(interval)
            waited += interval


def get_mat_file_from_xml(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for filename in root.findall('.//filename'):
        fname = filename.text
        if fname and fname.endswith('_cells_physicell.mat'):
            return os.path.join(os.path.dirname(xml_path), fname)
    raise FileNotFoundError('No _cells_physicell.mat file found in XML.')


def get_label_map_from_xml(xml_path: str) -> Dict[str, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    label_map = {}
    for label in root.findall('.//label'):
        name = label.text.strip()
        idx = int(label.attrib['index'])
        size = int(label.attrib.get('size', 1))
        if size == 1:
            label_map[name] = idx
        else:
            for i in range(size):
                label_map[f"{name}_{i}"] = idx + i
    return label_map
