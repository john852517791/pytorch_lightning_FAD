import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
import sys
sys.path.append("./")
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2FeedForward
)
from utils.ideas import moe 

class moe_Wav2Vec2FeedForward(Wav2Vec2FeedForward):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        moe_conf = [2,4,512]
        self.MoENet = moe.MoE(input_size=1024, output_size=1024, num_experts=moe_conf[1], hidden_size=moe_conf[2],conf=config, noisy_gating=True, k=moe_conf[0])


    def forward(self, hidden_states):
        bs,t,sp = hidden_states.shape
        
        hid, loss = self.MoENet(hidden_states.view(bs*t, sp))        
        return hid.view(bs,t, sp)


class moe_Wav2Vec2EncoderLayerStableLayerNorm(Wav2Vec2EncoderLayerStableLayerNorm):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.feed_forward = moe_Wav2Vec2FeedForward(config)


class moe_Wav2Vec2EncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
                [moe_Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
            )
    
    
    
class moe_Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.encoder = moe_Wav2Vec2EncoderStableLayerNorm(config)
        
        
if __name__ == "__main__":
    pretrain_model = moe_Wav2Vec2Model.from_pretrained(
        "datasets/pretrained_model/facebook/wav2vec2-xls-r-300m"
        )
    op= pretrain_model( torch.randn((8,64600)))
    print(op.shape)