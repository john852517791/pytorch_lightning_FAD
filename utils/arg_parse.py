#!/usr/bin/env python
import argparse
"""
startup_config

Startup configuration utilities

"""
import os
import sys
import torch
import importlib
import random
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    return






""" Arg_parse"""
# using the common args,like batchsize, epoch...
def f_args_parsed(argument_input = None):
        
    parser = argparse.ArgumentParser(
        description='General argument parse'
        )
    mes=""
    # random seed
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    # for DDP
    parser.add_argument('--gpuid', type=str, default="0")
    
    # ⭐⭐⭐ training model filename and its data config filename
    # 'module of model definition (default model, model.py will be loaded)'
    parser.add_argument('--module_model', type=str, default="model")
    # module of torch lightning model train step file
    parser.add_argument('--tl_model', type=str, default="models.tl_model")
    # datamodule python file
    parser.add_argument('--data_module', type=str, default="utils.loadData.asvspoof_data_DA")

    # pretrained model config(hugging face config)
    # parser.add_argument('--pretrained-model-config', type=str, default="")
     ######
    # ⭐⭐Training settings    
    # inference or train
    parser.add_argument('--inference', action='store_true', default=False, help=mes) 
    # 'batch size for training/inference (default: 8)'
    parser.add_argument('--batch_size', type=int, default=8)
    # 'number of epochs to train (default: 50)'
    parser.add_argument('--epochs', type=int, default=100)
    # 'number of no-best epochs for early stopping (default: 5)'
    parser.add_argument('--no_best_epochs', type=int, default=5)
   
    
    ######
    # ⭐options to save model 
    # checkpoint dir
    parser.add_argument('--savedir', type=str, default="./a_train_log", help='save model to this direcotry (default ./)')
 
    #######
    # ⭐options to load model
    # 'a trained model for inference or resume training '
    parser.add_argument('--trained_model', type=str, default="", help=mes + "(default: '')")
    # infer dataset
    parser.add_argument('--testset', type=str, default="LA21", help=mes + "(default: 'LA21, DF21, ITW')")
    parser.add_argument('--truncate', type=int, default=64600)
    
    
    # ⭐for loss selection
    # (default is CE, WCE, AM, OC, SAM, ASAM; other str will be defaultly set to CE)
    parser.add_argument('--loss', type=str, default="WCE")
    # 1 for reduction, 0 for no reduce
    parser.add_argument('--reduce', type=int, default=0)
    parser.add_argument('--loss_lr', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--eta', type=str, default=0.0)


    
    # # ⭐optimizer setting
    # for optimizer selection, (adam, adamw, sgd )
    parser.add_argument('--optim', type=str, default="adam")
    # # learning rate 
    parser.add_argument('--optim_lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
    # # weight_decay / l2 penalty
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # for SGD
    parser.add_argument('--momentum', type=float, default=0.9)
    
    
    # (cosWarmup, cosAnneal, step)
    parser.add_argument('--scheduler', type=str, default="")
    # warm up settings,uppper stage, default 3
    parser.add_argument('--num_warmup_steps', type=int, default=3)
    # for cosAnneal, num of train samples // batchsize * epochs
    parser.add_argument('--total_step', type=int, default=1057)
    # scheduler
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # applying data augmentation
    parser.add_argument('--usingDA', action='store_true', default=False)
    parser.add_argument('--da_prob', type=float, default=2)
        
        
    args_main = parser.parse_args()  
        
    if not args_main.usingDA:
        return args_main
    else:
        ##===================================================Rawboost data augmentation ======================================================================#

        parser.add_argument('--algo', type=int, default=5, 
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                            5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

        # LnL_convolutive_noise parameters 
        parser.add_argument('--nBands', type=int, default=5, 
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
        parser.add_argument('--minF', type=int, default=20, 
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
        parser.add_argument('--maxF', type=int, default=8000, 
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
        parser.add_argument('--minBW', type=int, default=100, 
                        help='minimum width [Hz] of filter.[default=100] ')
        parser.add_argument('--maxBW', type=int, default=1000, 
                        help='maximum width [Hz] of filter.[default=1000] ')
        parser.add_argument('--minCoeff', type=int, default=10, 
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
        parser.add_argument('--maxCoeff', type=int, default=100, 
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
        parser.add_argument('--minG', type=int, default=0, 
                        help='minimum gain factor of linear component.[default=0]')
        parser.add_argument('--maxG', type=int, default=0, 
                        help='maximum gain factor of linear component.[default=0]')
        parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
        parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
        parser.add_argument('--N_f', type=int, default=5, 
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

        # ISD_additive_noise parameters
        parser.add_argument('--P', type=int, default=10, 
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
        parser.add_argument('--g_sd', type=int, default=2, 
                        help='gain parameters > 0. [default=2]')

        # SSI_additive_noise parameters
        parser.add_argument('--SNRmin', type=int, default=10, 
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
        parser.add_argument('--SNRmax', type=int, default=40, 
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
        
        ##===================================================Rawboost data augmentation ======================================================================#
        args_main = parser.parse_args()  
    
    return args_main    
    # return parser.parse_args()
    
if __name__ == "__main__":
    pass