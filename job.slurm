#!/bin/bash

#SBATCH --job-name a100-ml
#SBATCH --partition high-gpu-mem
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ltupa001@odu.edu

module load container_env python3/2023.2-py310

cat >> output/job_logs << EOF

Code Executed @ $(date)
==============================================================

Task ID: $SLURM_JOB_ID

Parameters
Data: $1
Model: $2
Epochs: $3
Image Size: $4
Batch Size: $5
Project: $6
Name: $7

Description:
$8

==============================================================

EOF

crun -p ~/envs/ultralytics yolo segment train data=$1 model=$2 epochs=$3 imgsz=$4 batch=$5 device=0 project=$6 name=$7 seed=1337

# Example
# sbatch job.slurm datasets/clean/clean.yaml models/yolov8x-seg.pt 100 1280 1 ../trains clean "Baseline model for future references"