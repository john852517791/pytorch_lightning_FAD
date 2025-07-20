import sys
sys.path.append("./")
from transformers import Wav2Vec2Model,AutoConfig
import torch.nn as nn
import torch
from models.wav2vec.aasist import W2VAASIST




class this_arg():
    stage = 1
    # if stage == 1, means that the pretrained model is freezed
    # if stage == 2, means that training will finetune the pretrained model

class Model(nn.Module):
    def __init__(self, args = this_arg()):
        super().__init__()
        self.pretrain_model = Wav2Vec2Model.from_pretrained(
            "datasets/pretrained_model/facebook/wav2vec2-xls-r-300m" )
        self.classifier = W2VAASIST()
    
    def freeze_parameters(self):
        print("freeze")
        for param in self.pretrain_model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        print("unfreeze")
        for param in self.pretrain_model.parameters():
            # print(param.requires_grad)
            param.requires_grad = True
    
    def forward(self, x):
        # with torch.no_grad():
        if True:
            x = self.pretrain_model(
                x,
                output_hidden_states = True
                ).last_hidden_state
        pred , hidden_state = self.classifier(x)
        
        return pred, hidden_state
        
        
        
        
if __name__ == "__main__":
    md = Model()
    # md.freeze_parameters()
    # md.unfreeze_parameters()
    op, hd = md( torch.randn((8,64600)))
    print(op.shape)