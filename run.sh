#!/bin/bash
#SBATCH --mail-user=21422885@life.hkbu.edu.hk
#SBATCH --mail-type=end
#SBATCH --time=5:00:00
# std oupt
#SBATCH -o log.o
#SBATCH --partition=gpu
##SBATCH --partition=compute
#SBATCH --gres=gpu:0

#SBATCH --job-name="cp"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --gpus-per-task=1
##SBATCH --mem-per-cpu=8G
#SBATCH --account="research-ceg-tp"

module load 2022r2 cuda/11.7

module load miniconda3
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cp
cd ${HOME}/Devs/Crowd\-Prediction
python main.py --mode ood --file dcrnn

#export conda_env=${HOME}/anaconda3/envs/frl
