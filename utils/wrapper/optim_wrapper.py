import torch


class opti_conf():
    def __init__(self):
        self.optim = "adam"
        
        self.optim_lr = 0.0001
        self.weight_decay = 0.5
        # for sgd
        self.momentum = 0.9

class optimizer_wrap():
    def __init__(self,cfg:opti_conf,model):
        super(optimizer_wrap).__init__()
        self.cfg = cfg
        self.model = model


    def get_optim(self):
        optim = None
        
        if self.cfg.optim == "adam":
            optim = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.cfg.optim_lr, 
                weight_decay=self.cfg.weight_decay,
                )
        elif self.cfg.optim == "adamw":
            optim = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.cfg.optim_lr, 
                weight_decay=self.cfg.weight_decay
                )
        elif self.cfg.optim == "sgd":
            optim = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.cfg.optim_lr, 
                momentum = self.cfg.momentum
                )

        else:
            raise Exception(f"no optim named {self.cfg.optim}")


        return optim

    