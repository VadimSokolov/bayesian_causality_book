#!/bin/bash
#SBATCH --job-name=hbdml_cs
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-500
#SBATCH --output=logs/hbdml_cs_%A_%a.log
#SBATCH --exclude=amd[021-060]

# run folder of the 2026-10-23 rerun (sampler fix, review items F1)
cd "${BCB_RUN_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

python3 -u sim_cross_section.py --rep $SLURM_ARRAY_TASK_ID
