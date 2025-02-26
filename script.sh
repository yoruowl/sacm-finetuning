#!/bin/bash

#SBATCH --partition=week                        #Partition to submit to
#SBATCH --time=0-00:00:30                       #Time limit for this job
#SBATCH --nodes=1                               #Nodes to be used for this job during runtime. Use MPI jobs with multiple nodes.
#SBATCH --ntasks-per-node=1                     #Number of CPUs. Cannot be greater than number of CPUs on the node.
#SBATCH --mem=512                               #Total memory for this job
#SBATCH --job-name="LLM Finetuning"             #Name of this job in work queue
#SBATCH --output=llm_finetuning.out             #Output file name
#SBATCH --error=llm_finetuning.err              #Error file name
#SBATCH --mail-type=END                         #Email notification type (BEGIN, END, FAIL, ALL). To have multiple use a comma separated list. i.e END,FAIL.
#SBATCH --partition=GPU
#SBATCH --gpus=1

# Job Commands Below
#installing conda

bash Anaconda-latest-Linux-x86_64.sh
conda env create -f unsloth.yml
conda activate unsloth_env
conda env list
python3 finetuning.py