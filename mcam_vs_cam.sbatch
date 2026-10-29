#!/bin/bash
#SBATCH --job-name=mcam_vs_cam
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mcam_vs_cam-%j.log
#SBATCH --error=mcam_vs_cam-%j.log
#SBATCH --mail-type=BEGIN,END

# Usage:
#   sbatch mcam_vs_cam.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate tng311

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg

python3 -u mcam_vs_cam.py
