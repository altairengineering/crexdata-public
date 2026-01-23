import sys
import os
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import numpy as np

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
    # Try to find the main cell data array
    # This may be named 'cells' or similar; adjust as needed
    for key in mat:
        if not key.startswith("cells") and isinstance(mat[key], np.ndarray):
            arr = mat[key]
            # Heuristic: look for a 2D array with many columns (cell features)
            if arr.ndim == 2 and arr.shape[1] > 10:
                return arr
    raise ValueError("Could not find cell data array in .mat file.")

def detect_dead_tissue(cell_data, label_map, dead_phase_indices=[1,2], dead_flags_indices=None, dead_fraction_threshold=0.5):
    # dead_phase_indices: indices in 'current_phase' that correspond to dead states (apoptosis, necrosis, etc.)
    # dead_flags_indices: list of indices for custom death flags (e.g., pyroptosis)
    # label_map: dict mapping label names to column indices
    n_cells = cell_data.shape[0]
    if n_cells == 0:
        return True  # No cells left = dead tissue

    # Check current_phase
    phase_col = label_map.get("current_phase")
    dead_cells = np.zeros(n_cells, dtype=bool)
    if phase_col is not None:
        # Typical: phase 1 or 2 = dead (apoptosis/necrosis)
        dead_cells |= np.isin(cell_data[:, phase_col], dead_phase_indices)

    # Check custom death flags
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
            # For vector-valued labels, store start index
            for i in range(size):
                label_map[f"{name}_{i}"] = idx + i
    return label_map

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_alveoli.py <output_xml>")
        sys.exit(1)
    xml_path = sys.argv[1]
    mat_path = get_mat_file_from_xml(xml_path)
    label_map = get_label_map_from_xml(xml_path)
    cell_data = load_cell_data(mat_path)

    # Indices for custom death flags (adjust as needed)
    dead_flags = []
    for flag in ["cell_pyroptosis_flag", "cell_virus_induced_apoptosis_flag"]:
        if flag in label_map:
            dead_flags.append(label_map[flag])

    is_dead = detect_dead_tissue(
        cell_data,
        label_map,
        dead_phase_indices=[100,101],  
        dead_flags_indices=dead_flags,
        dead_fraction_threshold=0.5  # 50% of cells dead = dead tissue
    )
    with open("kill.txt" if is_dead else "round.txt", "w") as f:
        f.write("1" if is_dead else "0")
    print("dead" if is_dead else "not dead")