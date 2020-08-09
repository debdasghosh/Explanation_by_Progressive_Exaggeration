#!/bin/bash
#SBATCH -p DBMI-GPU
#SBATCH -N 1
#SBATCH -A bi561ip 
#SBATCH --ntasks-per-node 10
#SBATCH -t 48:00:00 # HH:MM:SS
#SBATCH --gres=gpu:p100:1
#SBATCH --time-min=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=singla

# echo commands to stdout
echo "$@"
"$@"
