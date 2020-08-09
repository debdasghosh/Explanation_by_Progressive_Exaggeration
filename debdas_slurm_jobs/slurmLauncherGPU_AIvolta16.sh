#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -N 1
#SBATCH -A ac5616p 
#SBATCH -t 48:00:00 # HH:MM:SS
#SBATCH --gres=gpu:volta16:1
#SBATCH --time-min=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=singla

# echo commands to stdout
echo "$@"
"$@"