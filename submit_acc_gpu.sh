#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_acc_gpu_cooling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive


# Use benchmark as default
MODE="${1:-benchmark}"

source scripts/env.leonardo.sh

INPUT_FILE="./input/Cooling.in"

case "${MODE}" in
    compare)
        EXECUTABLE="install/bin/cooling_acc_gpu_compare"
        HDF5_OUTPUT="output/Cooling_acc_gpu.h5"
        CSV_OUTPUT="output/Cooling_acc_gpu.csv"
        OUTPUT_EVERY=20
        ;;

    benchmark)
        EXECUTABLE="install/bin/cooling_acc_gpu_bench"
        HDF5_OUTPUT="none"
        CSV_OUTPUT="output/Cooling_acc_gpu_bench.csv"
        OUTPUT_EVERY=0
        ;;

    *)
        echo "[ERROR] Mode not valid: ${MODE}" >&2
        echo "Use: sbatch submit_acc_gpu.sh {compare|benchmark}" >&2
        exit 2
        ;;
esac

"${EXECUTABLE}" \
    "${INPUT_FILE}" \
    "${HDF5_OUTPUT}" \
    "${CSV_OUTPUT}" \
    "${OUTPUT_EVERY}"
