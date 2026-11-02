#!/bin/bash
#SBATCH --job-name=hbdml_pnl
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-500
#SBATCH --output=logs/hbdml_pnl_%A_%a.log
#SBATCH --exclude=amd[021-060]

# run folder of the 2026-10-23 rerun (items F1, F8, F15): four replications per task, 2,000 in all
cd "${BCB_RUN_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

python3 -u sim_panel.py --rep $SLURM_ARRAY_TASK_ID --nrep 4
