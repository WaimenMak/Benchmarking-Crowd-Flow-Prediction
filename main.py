# -*- coding: utf-8 -*-
"""
Created on 03/11/2022 22:11

@Author: mmai
@FileName: main.py
@Software: PyCharm
"""

import argparse

def str_to_bool(value):
    """
    code from MTGNN train multi step
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

# mmai$ ssh mmai@login.delftblue.tudelft.nl
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='gatrnn', help='file name')
    parser.add_argument('--cl', type=str_to_bool, default=False,help='whether to do curriculum learning')
    parser.add_argument('--loss_func', type=str, default='none', help='loss func')
    parser.add_argument('--step', type=int, default=12, help='input seq len')
    args = parser.parse_args()
    # model = "Linear Regression"
    if args.filename == "lr":
        from baselines.LRegression import main
    elif args.filename == "lr v1":
        from baselines.LRegression_v2 import main
    elif args.filename == "mlp":
        from baselines.MLP import main
    elif args.filename == "mlp v1":
        from baselines.MLP_v2 import main
    elif args.filename == "rnn seq2seq":
        from baselines.RNN import main
    elif args.filename == "rnn":
        from baselines.RNN_keras import main
    elif args.filename == "dcrnn":
        from baselines.DCRNN import main         # python main.py --filename dcrnn --mode ood
    elif args.filename == "xgboost":
        from baselines.XGBOOST import main
    elif args.filename == "xgboost v1":
        from baselines.XGBOOST_v2 import main
    elif args.filename == "var v1":            # python main.py --filename var\ v1 --step=3 --mode ood
        from baselines.VAR_v2 import main
    elif args.filename == "var":              # python main.py --filename var --step=3 --mode ood
        from baselines.VAR import main
    elif args.filename == "gatrnn":
        from baselines.GATRNN import main
    else:
        raise Exception("model is not in storage.")

    main(args)