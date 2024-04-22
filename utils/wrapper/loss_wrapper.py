import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
from typing import Tuple

from argparse  import Namespace



class loss_config():
    def __init__(self):
        self.loss = "WCE"
        self.reduce = 1
        # for WCE
        # self.loss_weight = torch.FloatTensor([0.1, 0.9])
        self.loss_lr = 0.01
        # for SAM ASAM
        self.samloss_optim = None
        self.model = None
        self.rho = 0.5
        self.eta = 0.0

class loss_wrap():
    def __init__(self,cfg:loss_config):
        super(loss_wrap).__init__()
        self.cfg = cfg
        if cfg.reduce == 1:
            self.reduce = "mean"
        else:
            self.reduce = "none"
            
    
    def get_loss(self):
        final_Loss = None
        loss_optim = None
        minimizor = None
        if self.cfg.loss == "CE":
            final_Loss = nn.CrossEntropyLoss(reduction=self.reduce)
        elif self.cfg.loss == "FOCAL":
            final_Loss = FocalLoss(gamma=2 , alpha=2580/22800)
        elif self.cfg.loss == "WCE":
            final_Loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]), reduction=self.reduce)
        elif self.cfg.loss == "WCEsf":
            final_Loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.15, 0.85]), reduction=self.reduce)
        elif self.cfg.loss == "AM":
            final_Loss=AMSoftmax()
            loss_optim = torch.optim.SGD(final_Loss.parameters(), lr=self.cfg.loss_lr) # 0.01
        elif self.cfg.loss == "OC":
            final_Loss=OCSoftmax()
            loss_optim = torch.optim.SGD(final_Loss.parameters(), lr=self.cfg.loss_lr) #0.0003
            
        elif self.cfg.loss == "SAM":
            minimizor = SAM(self.cfg.samloss_optim, self.cfg.model, rho=self.cfg.rho, eta=self.cfg.eta)
                 
        elif self.cfg.loss == "ASAM":     
            minimizor = ASAM(self.cfg.samloss_optim, self.cfg.model, rho=self.cfg.rho, eta=self.cfg.eta)
            
        else:
            raise Exception(f"no loss named {self.cfg.loss}")
        
        return final_Loss, loss_optim, minimizor
    
    
    


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()    


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0,reduce = True):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        self.reduce = reduce
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        if self.reduce:
            loss = self.softplus(self.alpha * scores).mean()
        else:
            loss = self.softplus(self.alpha * scores)
        # print(output_scores.squeeze(1).shape)
        # return loss, -output_scores.squeeze(1)
        return loss

class AMSoftmax(nn.Module):
    def __init__(self, num_classes=2, enc_dim=2, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)
        # print(margin_logits.shape)

        # return logits, margin_logits
        return logits



class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


if __name__ == "__main__":

    cfg = loss_config()
    cfg.loss = "wzy"
    wrap = loss_wrap(cfg)
    wrap.get_loss()