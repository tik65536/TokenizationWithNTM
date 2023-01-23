#!/bin/bash
#SBATCH -J tokenBytoken
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:45:00
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
module load python/3.8.6
cd $HOME/Thesis/tokenization
python main.py -seqlen 5 -memsize 10 -controllerDim 100 -head 3
