#!/bin/bash
#SBATCH --job-name=HMM
#SBATCH --output=/home-mscluster/djarvis/teacher_student_dynamics/run_out.txt
#SBATCH --ntasks=1
#SBATCH --partition=stampede
export CUDA_DIR=/usr/local/cuda-10.0-alternative
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.0-alternative
echo $CUDA_DIR
echo $XLA_FLAGS
cd /home-mscluster/djarvis/teacher_student_dynamics/teacher_student_dynamics/experiments/
python run.py
