#!/bin/bash
#SBATCH --job-name=mcam_hydro_only
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mcam_hydro_only-%j.log
#SBATCH --error=mcam_hydro_only-%j.log
#SBATCH --mail-type=BEGIN,END

# Usage:
#   sbatch mcam_hydro_only.sbatch                                                        # no sampling
#   sbatch --time=02:00:00 --export=ALL,RUN_SAMPLING=True mcam_hydro_only.sbatch       # MultiCamSampling
#   sbatch --export=ALL,FEATURES=mah mcam_hydro_only.sbatch                             # full-MAH features

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate tng311

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg
export RUN_SAMPLING=${RUN_SAMPLING:-False}
export FEATURES=${FEATURES:-z12}

echo "RUN_SAMPLING=$RUN_SAMPLING FEATURES=$FEATURES"

python3 -u mcam_hydro_only.py
