#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=sm3090el9_devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G

#SBATCH --output=logs_validation/electra_MM_%j.log

# Usage: sbatch validate_electra.sh [--config <yaml>] [--metrics_file <path>]
# Arguments are passed on to validate_mm.py (default config: hpc_conf_mol_mm_extra.yaml).

# Debugging flags (optional)

# KeOps needs CUDA; el8 nodes have 12.1.1, el9 nodes only 12.9.1.
module load CUDA/12.1.1 2>/dev/null || module load CUDA/12.9.1

source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate electra_env

python3 /home/energy/s234633/ELECTRA/validate_mm.py "$@"
