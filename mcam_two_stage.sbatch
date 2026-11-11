#!/bin/bash
#SBATCH --job-name=mcam_two_stage
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mcam_two_stage-%j.log
#SBATCH --error=logs/mcam_two_stage-%j.log
#SBATCH --mail-type=BEGIN,END

# Usage:
#   sbatch mcam_two_stage.sbatch                                                         # no sampling
#   sbatch --time=06:00:00 --export=ALL,RUN_SAMPLING=True mcam_two_stage.sbatch        # MultiCamSampling
#   sbatch --export=ALL,N_MSTAR_DRAWS=1 mcam_two_stage.sbatch                           # one sampled MultiCAM M* draw
#   sbatch --export=ALL,N_MSTAR_DRAWS=25 mcam_two_stage.sbatch                          # 25 sampled MultiCAM M* draws
# Outputs go to tests/mcam_two_stage/

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate tng311

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg
export RUN_SAMPLING=${RUN_SAMPLING:-False}
export N_MSTAR_DRAWS=${N_MSTAR_DRAWS:-0}

echo "RUN_SAMPLING=$RUN_SAMPLING N_MSTAR_DRAWS=$N_MSTAR_DRAWS"

python3 -u mcam_two_stage.py
