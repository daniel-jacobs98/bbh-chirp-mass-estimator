#!/bin/bash
#SBATCH --job-name=train_nn
#SBATCH --output=res.txt
#SBATCH --mem=25G
#SBATCH --time=6:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:1
module load gcc/9.2.0 openmpi/4.0.2
module load python/3.7.4
module load cudnn/7.6.4-cuda-10.1.243
source /fred/oz016/djacobs/ml/bin/activate
python train_nn.py
