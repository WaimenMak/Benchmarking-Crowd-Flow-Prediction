#!/bin/bash

#module load 2022r2 cuda/11.7
#
#module load miniconda3
#unset CONDA_SHLVL
#source "$(conda info --base)/etc/profile.d/conda.sh"

conda init bash
conda activate mps
cd ${HOME}/Devs/Crowd\-Prediction
for i in $(seq 1 5)
do
python LRegression_v2.py --mode in-sample;
python LRegression_v2.py --mode ood;
done



#python DCRNN.py --mode ood

#python GATRNN.py --mode in-sample
#python GATRNN.py --mode ood
#
#python RNN.py --mode in-sample
#python RNN.py --mode ood

#python XGBOOST.py --mode in-sample
#python XGBOOST.py --mode ood
#export conda_env=${HOME}/anaconda3/envs/frl
