#!/bin/bash
#SBATCH --job-name=physiboss_agent
#SBATCH --output=logs/agent_job_%j.out
#SBATCH --error=logs/agent_job_%j.err
#SBATCH --nodes=22
#SBATCH --ntasks=2464
#SBATCH --time=24:00:00
#SBATCH --qos=gp_resa
#SBATCH --account=cns119
set -euo pipefail

# === Load Modules ===
module purge
module load cmake oneapi fftw/3.3.10 dlb ucx

# === Check Args ===
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") EXPERIMENT_ID"
  exit 1
fi

# === Environment Setup ===
PROJ_DIR=$(pwd)
EXPID=$1
export TURBINE_OUTPUT="$PROJ_DIR/experiments/$EXPID"
EMEWS="$PROJ_DIR/pb/"
BASE_CONFIG="$PROJ_DIR/pb/PhysiBoSSv2/config/debug.xml"
EXE="$EMEWS/PhysiBoSSv2/COVID19"
PARAM_FILE="$EMEWS/data/variables.txt"
COUPLING_SCRIPT="$EMEWS/scripts/alya_physiboss_coupling.py"
mkdir -p "$TURBINE_OUTPUT"
echo "=== Running $EXPID in $TURBINE_OUTPUT ==="
# === Function to Run Alya Lung ===
run_alya_lung() {
    local ROUND=$1
    local LUNG_DIR="$TURBINE_OUTPUT/alya/round_${ROUND}/lung/"
    mkdir -p "$LUNG_DIR"
    cd "$LUNG_DIR"
    echo "=== Running Lung ==="

    if [[ $ROUND -eq 0 ]]; then
        cp -r $PROJ_DIR/alya/lung_surrogate/* $LUNG_DIR
        time srun $PROJ_DIR/alya/alya fensap > lung.out
        $PROJ_DIR/alya/lung_surrogate/alldepo.x fensap > lung_depo.out
    else
        cp -r $PROJ_DIR/alya/lung_standard/* $LUNG_DIR
        # Modify ALYA input to close infected lobes
        echo "=== Preparing Lung Input for Round $ROUND in the Lung Directory $LUNG_DIR ==="
        python3 "$COUPLING_SCRIPT" "modify_lung" "$TURBINE_OUTPUT" "$ROUND" "$LUNG_DIR"
        time srun $PROJ_DIR/alya/alya fensap > lung.out
        $PROJ_DIR/alya/lung_standard/alldepo.x fensap > lung_depo.out
    fi
    echo "=== Completed Lung ==="
}

# === Function to Run Alya Alveoli ===
run_alya_alveoli() {
    echo "=== Running Alveoli ==="
    local ROUND=$1
    local ALV_DIR="$TURBINE_OUTPUT/alya/round_${ROUND}/alveoli"
    mkdir -p "$ALV_DIR"
    cd "$ALV_DIR"
    cp -r $PROJ_DIR/alya/alveoli/* $ALV_DIR
    time srun $PROJ_DIR/alya/alya fensap > alveoli.out
    ./alya-deposition.x fensap > alveoli_depo.out
    echo "=== Completed Alveoli ==="
}

# === Function to Run Project2Plane ===
run_project2plane() {
    echo "  === Running Projection ==="
    local ROUND=$1
    local ALV_DIR="$TURBINE_OUTPUT/alya/round_${ROUND}/alveoli"
    local PROJ2DIR_DIR="$PROJ_DIR/project_2_plane"
    cd $PROJ2DIR_DIR
    echo "eftasa"
    cp "$ALV_DIR/fensap-deposition.pts.csv" "depo.pts.csv"
    ./project_2_plane "depo.pts" > "out.txt"
    # output is depo-depo.csv
    cp "depo.pts-depo.csv" "$ALV_DIR/"
    echo "  === Completed Projection ==="
}

# === Function to Generate PhysiBoSS Input Parameters ===
prepare_physiboss_input() {
    local ROUND=$1
    local LUNG_DIR="$TURBINE_OUTPUT/alya/round_${ROUND}/lung/lung_depo.out"
    local ALV_DIR="$TURBINE_OUTPUT/alya/round_${ROUND}/alveoli/"
    
    cd $EMEWS
    if [[ $ROUND -eq 0 ]]; then
        # Round 0: Parse ALYA outputs and generate initial PhysiBoSS parameters
        python3 "$COUPLING_SCRIPT" "prepare" "$LUNG_DIR" "$ALV_DIR/depo.pts-depo.csv" "$ROUND" "$PARAM_FILE"
    else
        # Subsequent rounds: Prepare inputs based on previous round results
        python3 "$COUPLING_SCRIPT" "prepare_next" "$TURBINE_OUTPUT" "$ROUND" "$LUNG_DIR" "$ALV_DIR/depo.pts-depo.csv"
    fi
    echo "PhysiBoSS input for round $ROUND prepared!"
}


run_physiboss_all() {
    echo $PARAM_FILE
    mapfile -t PARAMS < "$PARAM_FILE"
    export N_SIMULATIONS=${#PARAMS[@]}
    for i in "${!PARAMS[@]}"; do
        IDX=$((i+1))
        DIR="$TURBINE_OUTPUT/sim_${IDX}/"
        mkdir -p "$DIR"
        PARAM_JSON="${PARAMS[$i]}"
        python3 "$EMEWS/python/params.py" "$PARAM_JSON" "$BASE_CONFIG" "$DIR/settings.xml" "$DIR/" "$TURBINE_OUTPUT"
        "$EXE" "$DIR/settings.xml" > "$DIR/stdout.log" 2> "$DIR/stderr.log" &
        
    done
    # Launch the check_and_evaluate script in the background

}

# === Main Pipeline Loop ===
ROUNDS=4  # Set number of rounds
for ((ROUND=0; ROUND<$ROUNDS; ROUND++)); do
    echo "\n=== Starting round $ROUND ==="
    run_alya_lung $ROUND
    run_alya_alveoli $ROUND
    run_project2plane $ROUND
    prepare_physiboss_input $ROUND
    if [[ $ROUND -eq 0 ]]; then
        run_physiboss_all $ROUND
    fi

    # Evaluate PhysiBoSS outputs and generate round summary
    python3 "$COUPLING_SCRIPT" "evaluate" "$TURBINE_OUTPUT" "$ROUND"
    
    # Check if all lobes are infected (all tags == 1) and exit early if so
    if [[ -f "$TURBINE_OUTPUT/round_${ROUND}.csv" ]]; then
        HEALTHY_COUNT=$(grep -c ",0" "$TURBINE_OUTPUT/round_${ROUND}.csv" || true)
        if [[ $HEALTHY_COUNT -eq 0 ]]; then
            echo "=== All lobes infected. Stopping simulation early. ==="
            break
        fi
    fi
    
    echo "=== Round $ROUND finished ===\n"
done
wait
echo "All rounds completed."
