#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_acc_cpu_cooling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

export ACC_NUM_CORES=${SLURM_CPUS_PER_TASK}
source scripts/env.leonardo.sh
./install/bin/cooling_openacc_cpu ./input/Cooling.in none ./output/Cooling_acc_cpu.csv

