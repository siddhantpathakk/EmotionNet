from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

### FOR AUDIO EMBEDDING ###

class StatisticalUnit(nn.Module):
    def __init__(self, dim=0, keepdim=False):
        super(StatisticalUnit, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=self.keepdim)
        max = x.max(dim=self.dim, keepdim=self.keepdim)[0]
        min = x.min(dim=self.dim, keepdim=self.keepdim)[0]
        return torch.cat([mean, max, min], dim=self.dim).unsqueeze(0)


class AudioEmbeddingNet(nn.Module):
    def __init__(self):
        super(AudioEmbeddingNet, self).__init__()
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.stat_unit = StatisticalUnit()
        
    def forward(self, x):
        audio_embeddings = self.vggish.forward(x)
        utterance_wise_representation = self.stat_unit(audio_embeddings)
        return utterance_wise_representation



### MODELS FOR EMOTION RECOGNITION ###

class CumulativeContext(nn.Module):
    """
    ### State Update Equations:
    C_t = GRU_{C}(C_t-1, (S_t-1 CONCAT I_t-1 CONCAT u_t))
    where:
        C_t : cumulative context state
        I_t : intra-speaker state
        S_t : self-speaker state
        u_t : utterance embedding
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(CumulativeContext, self).__init__()

        self.gru_c = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, bias=bias,
                            bidrectional=True)
    
    
    def forward(self, x):
        output, _ = self.gru_c(x)
        output = output[:,-1, :]
        return output     
        
        
    def calculate_contextual_attention_vector(self, c):
        """             
        ### Soft attention:
        a_t = sum(alpha_i * C_i) ->> contextual attention vector
        alpha_i = softmax(u_i) 
        u_i = tanh(W * C_i + b)
        """
        u_i = torch.tanh(self.W * c + self.b)
        alpha_i = F.softmax(u_i)
        a_t = torch.sum(alpha_i * c)
        return a_t    
    
    
class IntraSpeakerState(nn.Module):
    """
    ### State Update Equations:
    I_t = GRU_I(I_t-1, (a_t CONCAT u_t))
    where:
        I_t : intra-speaker state
        a_t : contextual attention vector (obtained from cumulative context state)
        u_t : utterance embedding
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(IntraSpeakerState, self).__init__()
        
        self.gru_i = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            bidirectional=True)
        
    def forward(self, x):
        output, _ = self.gru_i(x)
        output = output[:,-1, :]
        return output
        


class SelfSpeakerState(nn.Module):
    """
    ### State Update Equations:
    S_KT = GRU_SK(S_KT-1, (C_KT CONCAT u_t))
    where:
        K belongs to {Speaker, Listener}
        It consists of two GRUs, one for speaker and one for listener
        S_KT : self-speaker state of the Kth speaker at time T
        C_KT : cumulative context state of the Kth speaker at time T
        u_t : utterance embedding
    """
    def __init__(self):
        super(SelfSpeakerState, self).__init__()

    def forward(self, x):
        pass
    
    
class EmotionState(nn.Module):
    """
    ### State Update Equations:
    E_t = GRU_E(E_t-1, (C_t CONCAT S_t CONCAT u_t))
    where:
        K belongs to {Speaker, Listener}
        It consists of two GRUs, one for speaker and one for listener
        S_KT : self-speaker state of the Kth speaker at time T
        C_KT : cumulative context state of the Kth speaker at time T
        u_t : utterance embedding
    """    
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(EmotionState, self).__init__()
        
        self.gru_e = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            bidirectional=True)

    def forward(self, x):
        output, _ = self.gru_e(x)
        output = output[:,-1, :]
        return output
        


class AudioEmotionNet(nn.Module):
    def __init__(self):
        super(AudioEmotionNet, self).__init__()
        
        self.attentivecontextual_state = CumulativeContext()
        self.intraspeaker_state = IntraSpeakerState()
        self.selfspeaker_state = SelfSpeakerState()
        
        self.emotion_state = EmotionState()
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pass