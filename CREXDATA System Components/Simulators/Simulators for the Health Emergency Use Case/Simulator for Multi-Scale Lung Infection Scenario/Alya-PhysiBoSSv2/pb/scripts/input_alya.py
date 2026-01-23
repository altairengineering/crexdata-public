# script that processes physibosses round and generates input for ALYA
from glob import glob
import os,sys,re
from combined_alveoli import wait_for_round_end,get_lung_data
import pandas as pd

code_map = {
31  : "0.77219341E-004",
32  : "0.99817803E-004",
33  : "0.88628529E-004",
34  : "0.39321860E-004",
35  : "0.10301267E-003"     
}
code_to_lp ={
        31 : "LU",
        32 : "LL",
        33 : "RU",
        34 : "RM",
        35 : "RL"
}
def read_fensap_boundary_last_time(file_path, filter_codes=None):
    """
    Read the last Time block from a fensap-boundary.nsi.dat style file and
    return a dict mapping the first-column integer code -> last-column float value.

    - file_path: path to the fensap-boundary.nsi.dat file
    - filter_codes: iterable of int codes to include (if None, include all found)

    The function looks for the last line that starts with '#' and contains 'Time'
    and parses the following data lines until a blank or non-matching line.
    Scientific notation with 'E' (or Fortran 'D') is supported.
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
            # stop if we hit a line that doesn't match the data pattern
            break
        code = int(m.group(1))
        last_col_str = m.group(4).replace('D', 'E').replace('d', 'e')
        try:
            last_col = float(last_col_str)
        except ValueError:
            # skip unparsable values
            continue
        if filter_codes is None or code in filter_codes:
            result[code] = last_col

    return result
def get_sim_id(sim_folder):
    p = os.path.join(sim_folder, "variables.txt")
    print("Getting simulation ID from:", sim_folder)
    with open(p, "r") as f:
        # 1 ls 34.27
        line = f.readline().strip()
        parts = line.split(" ")
        return parts[1] if len(parts) > 1 else "unknown"
def create_round_flag_file(sim_folder, tag, round):
    print("Creating round flag file in:", sim_folder)
    next_round_folder = sim_folder + "/round_{}/".format(int(round)+1)
    with open(os.path.join(next_round_folder, "round.txt"), "w") as f:
        f.write(tag)
def modify_alya_input(pb_sim_folder,round,alya_folder):
    prev_round = int(round)-1
    print(f"Modifying ALYA input for round {round} in {alya_folder}")
    pb_input = os.path.join(pb_sim_folder, "round_{}".format(prev_round)+".csv")
    print(f"Reading Physiboss input from {pb_input}")
    lobes = pd.read_csv(pb_input,header=0)
    print(lobes)
    # get codes from lobes that tag == 1
    codes = lobes[lobes["tag"] == 1]["sim_id"].values
    print(f"Lobes to close (tag==1): {codes}")
    fensap_path = os.path.join(alya_folder, "fensap.nsi.dat")
    # now put 11 in fensap.nsi.dat file where the tag in the lobes dataframe is 1
    with open(fensap_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Match lines like " 31            000  0.000000 0.000000 0.000000 $ LU"
        parts = line.strip().split()
        print(f"Processing line: {line.strip()}")
        if len(parts) >= 2:
            try:
                code = parts[-1]
                print(f"Processing line for code {code}: {line.strip()}")
                if code in codes and parts[1] == "000":
                    print(f"Modifying line for code {code}: {line.strip()}")
                    # Replace only the code part with 111
                    line = line.replace("000", "111", 1)
            except ValueError:
                pass
        if ("FLOW_RATE" in parts):
            if int(round) == 1:
                # Reverse mapping: lobe name to code
                lp_to_code = {v: k for k, v in code_to_lp.items()}
                active_lobes = lobes[lobes['tag'] == 1]['sim_id'].tolist()
                subtract_sum = sum(float(code_map[lp_to_code[lobe]]) for lobe in active_lobes)
                original_flow_rate = float(parts[2].strip(","))

                # Calculate new flow rate
                new_flow_rate = original_flow_rate - subtract_sum
                line = line.replace(parts[2], f"{new_flow_rate:.8f}", 1)
                print("Updated FLOW_RATE:", new_flow_rate)

            elif int(round) >1:
                # here you need to go and read the fensap-boundary.nsi.dat
                #  and get the last flow rates
                # and then deduct from
                file="/fensap-boundary.nsi.set"
                # but in previous folder?
                print("experiments/r2/alya/round_1/lung/fensap-boundary.nsi.set")
                fensap_boundary_path = pb_input = os.path.join(pb_sim_folder, "alya/round_{}".format(prev_round)+"/lung/"+file)
                print(fensap_boundary_path)
                res = read_fensap_boundary_last_time(fensap_boundary_path, filter_codes=None)
                print(res)
                # exit()
                lp_to_code = {v: k for k, v in code_to_lp.items()}
                print(lp_to_code)
                active_lobes = lobes[lobes['tag'] == 1]['sim_id'].tolist()
                subtract_sum = sum(float(res[lp_to_code[lobe]]) for lobe in active_lobes)
                original_flow_rate = float(parts[2].strip(","))

                # Calculate new flow rate
                new_flow_rate = original_flow_rate - subtract_sum
                line = line.replace(parts[2], f"{new_flow_rate:.8f}", 1)
                print("Updated FLOW_RATE:", new_flow_rate)
                # pass

        new_lines.append(line)
    with open(fensap_path, "w") as f:
        f.writelines(new_lines)  
    print(f"Modified fensap.nsi.dat in {alya_folder} for round {round} with codes: {codes}")
def combine_physiboss_outputs(turbine_output,round):
    print(f"Combining outputs for round {round} in {turbine_output}")
    tags = {}
    for folder in sorted(os.listdir(turbine_output)):
        
        if folder.startswith("sim"):
            sim_folder = os.path.join(turbine_output, folder)
            sim_round = os.path.join(turbine_output, folder, "round_{}/".format(round))
            print(f"sim round {sim_round}")
            print(f"Gamimeno print sim folder in {sim_folder}")

            wait_for_round_end(sim_round)
            tag_file = sim_round + "/tag.txt"
            if os.path.exists(tag_file):
                with open(tag_file, "r") as f:
                    tag = f.read().strip()
                    tag = tag.split(",")[0]
                    sim_id = get_sim_id(sim_folder)
                    print(tag)
                    tags[sim_id] = tag
                # create_round_flag_file(sim_folder, tag, round)
                
                # now create round txt with 1 inside or 0 depending on the tag.

    with open(os.path.join(turbine_output, "round_{}.csv".format(round)), "w") as f:
        f.write("sim_id,tag\n")
        for sim_id, tag in tags.items():
            f.write(f"{sim_id},{tag}\n")
def main():
    mode = sys.argv[1]
    turbine_output = sys.argv[2]
    round = sys.argv[3]
    if mode == "pb":
        combine_physiboss_outputs(turbine_output, round)
    if mode == "lung":
        alya_folder = sys.argv[4]
        modify_alya_input(turbine_output, round, alya_folder)
    if mode == "input_pb":
        alya_dist_file = sys.argv[4]
        alya_depo_file = sys.argv[5]
        dist = get_lung_data(alya_dist_file)
        print(f"dist {dist}" )
        prev_round = str(int(round)-1)
        tags = pd.read_csv(os.path.join(turbine_output, "round_{}.csv".format(prev_round)), header=0)

        for folder in sorted(os.listdir(turbine_output)):
            if folder.startswith("sim"):
                sim_folder = os.path.join(turbine_output, folder)
                sim_round = os.path.join(turbine_output, folder, "round_{}/".format(round))
                lobe_id = get_sim_id(sim_folder)
                print(f"lobe id {lobe_id}")
                # now create in the sim_round folder a file called round.txt that contains the contents of the deposition file
                with open(os.path.join(sim_round, "input_virion_alya.txt"), "w") as f:
                   with open(alya_depo_file, "r") as depo_file:
                        for line in depo_file:
                            f.write(line)
                with open(os.path.join(sim_folder, "moi.txt"), "w") as f:
                    f.write(str(dist[lobe_id]))
                # create the round file with the tag
                with open(os.path.join(sim_round, "round.txt"), "w") as f:
                    print(f"Writing round.txt for {lobe_id} in {sim_round}")
                    # get the tag for this lobe id
                    tag = tags[tags['sim_id'] == lobe_id]['tag'].values[0]
                    # write something
                    f.write(str(tag))

if __name__ == "__main__":
    main()
