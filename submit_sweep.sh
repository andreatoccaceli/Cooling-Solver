#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_cuda_cooling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

source scripts/env.leonardo.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ACC_NUM_CORES="${SLURM_CPUS_PER_TASK}"

DEVICE="${1:-}"

case "${DEVICE}" in
    cpp)
        EXECUTABLE="install/bin/cooling_cpp_bench"
        ;;
    omp)
        EXECUTABLE="install/bin/cooling_omp_bench"
        ;;
    acc_cpu)
        EXECUTABLE="install/bin/cooling_acc_cpu_bench"
        ;;
    acc_gpu)
        EXECUTABLE="install/bin/cooling_acc_gpu_bench"
        ;;
    cuda)
        EXECUTABLE="install/bin/cooling_cuda_bench"
        ;;
    *)
        echo "[ERROR] Unsupported device: ${DEVICE:-<missing>}" >&2
        echo "Usage: sbatch submit_sweep.sh {cpp|omp|acc_cpu|acc_gpu|cuda}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${EXECUTABLE}" ]]; then
    echo "[ERROR] Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi

if [[ ! -d input/sweeps ]]; then
    echo "[ERROR] Sweep input directory not found: input/sweeps" >&2
    exit 1
fi

mkdir -p logs

INPUT_FILES=()
while IFS= read -r input_file; do
    INPUT_FILES+=("${input_file}")
done < <(find input/sweeps -type f -name '*.in' -print | sort -V)

if ((${#INPUT_FILES[@]} == 0)); then
    echo "[ERROR] No input files found under input/sweeps" >&2
    exit 1
fi

echo "[INFO] Device:       ${DEVICE}"
echo "[INFO] Executable:   ${EXECUTABLE}"
echo "[INFO] Input files:  ${#INPUT_FILES[@]}"
echo "[INFO] CPU threads:  ${SLURM_CPUS_PER_TASK}"

completed=0

for input_file in "${INPUT_FILES[@]}"; do
    category="$(basename "$(dirname "${input_file}")")"
    configuration="$(basename "${input_file}" .in)"
    run_stem="sweep_${DEVICE}_${category}_${configuration}_j${SLURM_JOB_ID}"

    echo "[RUN] $((completed + 1))/${#INPUT_FILES[@]} input=${input_file}"

    "${EXECUTABLE}" \
        "${input_file}" \
        none \
        /dev/null \
        0 \
        > "logs/${run_stem}.out" \
        2> "logs/${run_stem}.err"

    ((completed += 1))
done

echo "[DONE] ${completed} configurations completed for ${DEVICE}"
