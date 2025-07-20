import sys
sys.path.append("./")
from transformers import Wav2Vec2Model,AutoConfig
import torch.nn as nn
import torch
from models.wav2vec.aasist import W2VAASIST
from models.moe_research.arch_w2v2.w2v2_MoEMLP import moe_Wav2Vec2Model,moe_Wav2Vec2FeedForward



class this_arg():
    stage = 1
    # if stage == 1, means that the pretrained model is freezed
    # if stage == 2, means that training will finetune the pretrained model

class Model(nn.Module):
    def __init__(self, args = this_arg()):
        super().__init__()
        self.pretrain_model = moe_Wav2Vec2Model.from_pretrained(
            "datasets/pretrained_model/facebook/wav2vec2-xls-r-300m" )
        self.classifier = W2VAASIST()
        # self.freeze_parameters()
        # self.unfreeze_moe_Wav2Vec2FeedForward()
        
    
    def freeze_parameters(self):
        print("freeze other")            
        for param in self.pretrain_model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        print("unfreeze")
        for param in self.pretrain_model.parameters():
            # print(param.requires_grad)
            param.requires_grad = True
        
    def unfreeze_moe_Wav2Vec2FeedForward(self):
        print("fine tune moe MLP")            
        for name, module in self.pretrain_model.named_modules():
            if isinstance(module, moe_Wav2Vec2FeedForward):
                for param in module.parameters():
                    param.requires_grad = True
    
    
    def forward(self, x):
        x = self.pretrain_model(
                x,
                output_hidden_states = False
                ).last_hidden_state
        pred , hidden_state = self.classifier(x)
        
        return pred, hidden_state
        
        
        
        
if __name__ == "__main__":
    md = Model().to("cuda:6")
    # md.freeze_parameters()
    # md.unfreeze_parameters()
    op, hd = md( torch.randn((8,64600)).to("cuda:6"))
    print(op.shape)