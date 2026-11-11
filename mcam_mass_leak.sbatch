#!/bin/bash
#SBATCH --job-name=mcam_mass_leak
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mcam_mass_leak-%j.log
#SBATCH --error=logs/mcam_mass_leak-%j.log
#SBATCH --mail-type=BEGIN,END

# Usage:
#   sbatch mcam_mass_leak.sbatch
# Outputs go to tests/mass_leak/

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate tng311

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg

python3 -u mcam_mass_leak.py
