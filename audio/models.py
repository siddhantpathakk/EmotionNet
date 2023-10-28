from torch import nn
import torch.nn.functional as F
import torch

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


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha


### MODEL FOR EMOTION RECOGNITION ###
class NNCell(nn.Module):
    def __init__(self, D_u, D_c, D_s, D_i, D_e, D_a):
        """
        D_u: Input dimension u
        D_c: CumulativeContext dimension
        D_s: SelfSpeaker dimension
        D_i: IntraSpeaker dimension
        D_e: EmotionState dimension
        D_a: Attention vector dimension
        """
        super(NNCell, self).__init__()

        # Defining the GRU cells for different states
        self.c_cell = nn.GRUCell(D_s + D_i + D_u, D_c)
        self.ss_cell = nn.GRUCell(D_c + D_u, D_s)
        self.sl_cell = nn.GRUCell(D_c + D_u, D_s)
        self.i_cell = nn.GRUCell(D_a + D_u, D_i)
        self.e_cell = nn.GRUCell(D_c + D_s + D_u, D_e)
    
    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel
    
    def forward(self, U, qmask, g_hist, q0, e0):
        
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        # gru_g 
        # g_t = GRU_g (g_t-1, [U CONCAT q0_sel])


        c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
        alpha = None

        # U_c = [U CONCAT c]
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)

        # gru_p
        # q_s_t = GRU_p (q_s_t-1, [U_c])
        
        U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
        ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
        U_ss_ = torch.cat([U_,ss_],1)
        ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        # gru_l
        # q_l_t = GRU_l (q_l_t-1, [U_ss])
        # U_ss = [U CONCAT ss]
        
        
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        # gru_e
        # e_t = GRU_e (e_t-1, [q_sel])

        return g_,q_,e_,alpha


class EmotionNN(nn.Module):
    def __init__(self, dim_c, dim_s, dim_i, dim_u, dim_e, dim_a):
        super(EmotionNN, self).__init__()
        
        self.dim_c = dim_c
        self.dim_s = dim_s
        self.dim_i = dim_i
        self.dim_u = dim_u
        self.dim_e = dim_e
        self.dim_a = dim_a # can be calculated  beforehand using the above dimensions
                
        self.nn_cell = NNCell(dim_c, dim_s, dim_i, dim_u, dim_e, dim_a)
            
    def forward(self, u):
        pass
    

class EmotionClassifier(nn.Module):
    def __init__(self, dim_c, dim_s, dim_i, dim_u, dim_e, dim_a, dim_h, num_classes):
        super(EmotionClassifier, self).__init__()
        
        self.dim_c = dim_c
        self.dim_s = dim_s
        self.dim_i = dim_i
        self.dim_u = dim_u
        self.dim_e = dim_e
        self.dim_a = dim_a
        
        self.dim_h = dim_h
        
        self.num_classes = num_classes
        
        self.emotion_nn = EmotionNN(dim_c, dim_s, dim_i, dim_u, dim_e, dim_a, dim_h)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(dim_e, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, num_classes),
        )
        
    def forward(self, u):
        pass