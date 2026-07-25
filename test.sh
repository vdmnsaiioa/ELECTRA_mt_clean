#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G

#SBATCH --output=logs5/electra_MM_%j.log

# Debugging flags (optional)

source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate electra_env

#python3 /home/energy/s234633/ELECTRA/train_frozen_mlp.py --config /home/energy/s234633/ELECTRA/hpc_conf_frozen.yaml --feature_cache /home/energy/s234633/ELECTRA/frozen_features_with_density.pt --epochs 20
python3 /home/energy/s234633/ELECTRA/train2.py
