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
        pretrain_cfg = AutoConfig.from_pretrained("datasets/pretrained_model/facebook/wav2vec2-xls-r-300m/config.json")
        pretrain_cfg.num_hidden_layers = 6
        self.pretrain_model = Wav2Vec2Model.from_pretrained(
            "datasets/pretrained_model/facebook/wav2vec2-xls-r-300m",
            config=pretrain_cfg)
        self.classifier = W2VAASIST()
        # self.freeze_parameters()
        # self.register_buffer('pre_features', torch.zeros(args.batch_size,160)) 
        self.register_buffer('pre_features', torch.zeros(32,160)) 
        # self.register_buffer('pre_weight1', torch.ones(args.batch_size, 1))
        self.register_buffer('pre_weight1', torch.ones(32, 1))
    
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
        with torch.no_grad():
            x = self.pretrain_model(
                x,
                output_hidden_states = True
                ).hidden_states[5]
        pred , hidden_state = self.classifier(x)
        
        return pred, hidden_state
        
        
        
        
if __name__ == "__main__":
    md = Model()
    # md.freeze_parameters()
    # md.unfreeze_parameters()
    op, hd = md( torch.randn((8,64600)))
    print(op.shape)