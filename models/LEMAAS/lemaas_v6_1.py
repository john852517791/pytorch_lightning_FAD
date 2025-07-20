"""
lemaas
"""
import numpy as np
import torch,math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

def nearest_odd_number(c):
    # 计算表达式的值
    value = 0.5 * math.log2(c) + 0.5
    # 取整并转换为最接近的奇数
    nearest_odd = round(value)
    if nearest_odd % 2 == 0:
        # 如果最接近的奇数是偶数，则向上取整到下一个奇数
        nearest_odd += 1
    return nearest_odd


class eca_layer(nn.Module):
    def __init__(self, channel):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.k_size = nearest_odd_number(channel)
        self.conv = nn.Conv1d(channel, channel, kernel_size=self.k_size,padding=(self.k_size-1)//2, bias=True, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class modified_Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    planes: dimension of input tensor
    """
    def __init__(self,planes, stride=1, scales=4, groups=1, norm_layer=True,drop_path = 0):
        super().__init__()
        if planes % scales != 0: #输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  #BN层
            norm_layer = nn.BatchNorm1d
        
        self.scales = scales
        self.stride = stride
        self.relu = nn.ReLU()
        #3*3的卷积层，一共有3个卷积层和3个BN层
        self.res2net_conv1n = nn.ModuleList([nn.Conv1d(planes // scales, planes // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales-1)])
        self.res2net_bn = nn.ModuleList([norm_layer(planes // scales) for _ in range(scales-1)])
        
        self.pwconv1 = nn.Linear(planes, 4 * planes) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.SELU()
        self.pwconv2 = nn.Linear(4 * planes, planes)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cam1 = eca_layer(planes)

    def forward(self, x):
        input = x
        
        # scales个(1x3)的残差分层架构
        xs = torch.chunk(x, self.scales, 1) #将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.res2net_bn[s-1](self.res2net_conv1n[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.res2net_bn[s-1](self.res2net_conv1n[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = out.permute(0, 2, 1) # (N, C, W) -> (N, W, C)
        # linear layer in in*4
        x = self.pwconv1(out)
        # selu
        x = self.act(x)
        # linear layer in in
        x = self.pwconv2(x)
        # channel attention module
        x = x.permute(0, 2,1) # (N, W, C) -> (N, C, W)
        x = self.cam1(x)
        
        x = input + self.drop_path(x)
        return x

class Model(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        op1 = 16
        op2 = 32
        op3 = 64
        op4 = 128
        self.conv1 = nn.Conv1d(1, op1, kernel_size=(7,), stride=(1,),padding=3)
        self.bn1_1 = nn.BatchNorm1d(op1)
        self.bn1_2 = nn.BatchNorm1d(op1)
        self.block1_1 = modified_Block(op1,)
        
        self.conv2 = nn.Conv1d(op1, op2, kernel_size=(7,), stride=(1,),padding=3)
        self.bn2 = nn.BatchNorm1d(op2)
        self.block2_1 = modified_Block(op2,)
        self.block2_2 = modified_Block(op2,)
        
        self.conv3 = nn.Conv1d(op2, op3, kernel_size=(7,), stride=(1,),padding=3)
        self.bn3 = nn.BatchNorm1d(op3)
        self.block3_1 = modified_Block(op3,)
        self.block3_2 = modified_Block(op3,)
        self.block3_3 = modified_Block(op3,)

        self.conv4 = nn.Conv1d(op3, op4, kernel_size=(7,), stride=(1,),padding=3)
        # self.conv4 = nn.Conv1d(op3, op4, kernel_size=(7,), stride=(1,),padding=3)
        self.bn4 = nn.BatchNorm1d(op4)
        self.block4 = modified_Block(op4,)
        
        self.maxpool = nn.MaxPool1d(kernel_size=9)
        self.flatten = nn.Flatten()
        self.selu_act = nn.SELU()
        
        self.linear1 = nn.Linear(1792,64)
        self.linear2 = nn.Linear(64,16)
        self.linear3 = nn.Linear(16,2)
        self.softmax = nn.Softmax(1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1_1(x)
        x = self.block1_1(x)
        x = self.bn1_2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.block4(x)
        x = self.bn4(x)
        x = self.selu_act(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        
        x = self.linear1(x)
        x = self.selu_act(x)
        x = self.linear2(x)
        x = self.selu_act(x)
        x = self.linear3(x)
        x = self.softmax(x)
        
        return x,None
        
        
        
        
if __name__ == "__main__":
    md = Model()
    # print(summary(md, torch.randn((8,64600)), show_input=False))
    op,res = md( torch.randn((4,96000)))
    print(op.shape)
    # # print(res.shape)
    print(sum(i.numel() for i in md.parameters() if i.requires_grad)/1000)  # 0.97M
    
    # test = eca_layer(16)
    # z = test(torch.randn((8,16,95994)))
    # print(z.shape)
    # test = eca_layer(32)
    # z = test(torch.randn((8,32)))
    # print(z.shape)
    # test = eca_layer(64)
    # z = test(torch.randn((8,64)))
    # print(z.shape)
    # test = eca_layer(128)
    # z = test(torch.randn((8,128)))
    # print(z.shape)
    
    # mblock = modified_Block(16)
    # print(mblock(torch.randn((32, 16, 95994))).shape)
    # print(sum(i.numel() for i in mblock.parameters() if i.requires_grad)/1000)  # 0.97M
    