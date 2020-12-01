#!/bin/bash
#SBATCH --job-name=gen_sigs_o2
#SBATCH --output=./res_fullrange.txt
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1000
module load gcc/7.3.0
module load python/3.6.4
source /fred/oz016/djacobs/samplegen/bin/activate
python generate_dataset_multiprocessed_new.py -o /fred/oz016/djacobs/Datasets/new_snr/snr_20_30/snr_20_30_100k_5.hdf 
