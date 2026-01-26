#!/bin/bash
#SBATCH --account=project_2017263   # Choose the project to be billed. Change to own project, if used outside of the course.
#SBATCH --partition=gpusmall        # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job.             
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=32           # How many processors work on one task. Upper limit depends on number of CPUs per GPU. In Mahti there are 32 CPU cores per one GPU.
#SBATCH --time=03:00:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --mem=40G                   # Reserved memory.
#SBATCH --gres=gpu:a100:1           # Number of GPUs 

# Load Pytorch module
module load pytorch/2.9

# Activate virtual environment containing special packages
source /projappl/project_2017263/fmi_course/bin/activate

# Run the Python code
srun python3 train_model.py
