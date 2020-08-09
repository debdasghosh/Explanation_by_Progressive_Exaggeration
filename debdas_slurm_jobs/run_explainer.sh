#!/usr/bin/env bash
PREFIX="Explainer"
JOBNAME=$PREFIX"RUN"
OUTFN="/pylon5/ac5616p/debdas/Explanation/slurm_output/Output/"$PREFIX"RUN.log"
ERRFN="/pylon5/ac5616p/debdas/Explanation/slurm_output/Error/"$PREFIX"RUN.log"
# sbatch -o $OUTFN -e $ERRFN --job-name $JOBNAME /pylon5/ac5616p/debdas/Explanation/debdas_slurm_jobs/slurmLauncherDBMI_GPU_p100.sh  python /pylon5/ac5616p/debdas/Explanation/TrainExplainer.py

sbatch -o $OUTFN -e $ERRFN --job-name $JOBNAME /pylon5/ac5616p/debdas/Explanation/debdas_slurm_jobs/slurmLauncherGPU_AIvolta16.sh  python /pylon5/ac5616p/debdas/Explanation/TrainExplainerV2.py