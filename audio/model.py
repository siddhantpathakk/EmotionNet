import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector
        return attn_pool, alpha


### MODEL FOR EMOTION RECOGNITION ###
class EmotionGRUCell(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(EmotionGRUCell, self).__init__(*args, **kwargs)
    
    def _select_parties(self, X, indexes):
        pass
    
    def forward(self, U):
        pass
    
class EmotionRNN(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(EmotionRNN, self).__init__(*args, **kwargs)
        
    def forward(self, U):
        pass
    

class EmoNet(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(EmoNet, self).__init__(*args, **kwargs)
        
    def _reverse_sequence(self, X, mask):
        pass
    
    def forward(self, U):
        pass