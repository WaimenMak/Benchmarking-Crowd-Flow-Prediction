#!/bin/bash
#SBATCH --mail-user=21422885@life.hkbu.edu.hk
#SBATCH --mail-type=end
#SBATCH --time=5:00:00
# std oupt
#SBATCH -o log.o
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --job-name="cp"
#SBATCH --ntasks=1
#SBATCH --account="research-ceg-tp"

module load 2023r1 cuda/11.7

module load miniconda3
#unset CONDA_SHLVL
#source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cp
cd ${HOME}/Devs/Crowd\-Prediction
echo "Current working directory: $(pwd)"
#python main.py --mode ood --file dcrnn

#python ./baselines/GATRNN.py
python ./baselines/STGCN.py
#python main.py --mode in-sample --file gatrnn
#python main.py --mode in-sample --file stgcn
#export conda_env=${HOME}/anaconda3/envs/frl
