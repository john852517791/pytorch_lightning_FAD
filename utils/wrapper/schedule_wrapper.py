from torch import optim, nn, utils, Tensor
import torch,transformers
import numpy as np


class schdule_conf():
    def __init__(self):
        self.scheduler = "cosWarmup"
        self.epochs = 100
        # for cosWarmup
        self.num_warmup_steps = 5
        # self.num_training_steps = self.epochs - self.num_warmup_steps
        # for cosanneal 
        self.total_step = 1057 # (25380//24) * 100
        # for step
        self.step_size = 5
        self.gamma = 0.1
        
        self.optim_lr = 1


class scheduler_wrap():
    """ Wrapper over different types of learning rate Scheduler
    
    """
    def __init__(self, optimizer, args:schdule_conf): 
        self.optimizer = optimizer
        self.args = args
        
    def get_scheduler(self):
        
        # other config or none
        scheduler = None   
         
        if  self.args.scheduler == "cosWarmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer = self.optimizer, 
                num_warmup_steps=self.args.num_warmup_steps,          
                num_training_steps = self.args.epochs
            )
        elif self.args.scheduler == "cosAnneal":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: cosine_annealing(
                        step,
                        105700,
                        1,  # since lr_lambda computes multiplicative factor
                        0.000005 / 0.0001))
        elif self.args.scheduler == "normal_cosAnneal":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                # eta_min=0.000005 / 0.0001
                eta_min=0.05*self.args.optim_lr
                )
        elif self.args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size= self.args.step_size, 
                gamma=self.args.gamma
                )
        else:
            print(f"no scheduler is used")
            
        
        return scheduler
    
    
def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

