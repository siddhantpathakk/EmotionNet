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
    def __init__(self, D_u, D_g, D_s, D_i, D_e, D_a=100, listener_state=False):
        super().__init__()
        
        self.D_u = D_u
        self.D_g = D_g
        self.D_s = D_s
        self.D_e = D_e
        
        self.g_cell = nn.GRUCell(D_s + D_i + D_u, D_g)
        self.p_cell = nn.GRUCell(D_g + D_u, D_s)
        self.i_cell = nn.GRUCell(D_a + D_u, D_i)
        self.e_cell = nn.GRUCell(D_g + D_s + D_u, D_e)

        self.attention = SimpleAttention(D_g)
        
        self.listener_state = listener_state
        
        if self.listener_state:
            self.l_cell = nn.GRUCell(D_s + D_u, D_s)
    
    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel
    
    def forward(self, U, qmask, g_hist, i0, e0, q0):
        # q0 is equivalent to s0
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)
        
        qs_i_U = torch.cat([q0_sel, i0, U], dim=1)
        g_ = self.g_cell(qs_i_U,
                         torch.zeros(U.size()[0], self.D_g).type(U.type()) if g_hist.size()[0]==0 
                         else g_hist[-1])

        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist)
        
        g_U = torch.cat([g_, U], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        qs_ = self.p_cell(g_U.contiguous().view(-1, self.D_g + self.D_u),
                            q0_sel.view(-1, self.D_s)).view(U.size()[0], -1, self.D_s)
        
        if self.listener_state:
            ql_ = self.l_cell(g_U.contiguous().view(-1, self.D_g + self.D_u),
                                q0_sel.view(-1, self.D_l)).view(U.size()[0], -1, self.D_l)
            
        else:
            ql_ = q0
            
        c_u = torch.cat([c_, U], dim=1)
        i_ = self.i_cell(c_u, i0)
            
        qmask = qmask.unsqueeze(2)
        q_ = qmask * qs_ + (1 - qmask) * ql_
        e0 = torch.zeros(U.size()[0], self.D_e).type(U.type()) if e0 is None else e0
        
        
        g_q_u = torch.cat([g_, self._select_parties(q_, qm_idx), U], dim=1)
        e_ = self.e_cell(g_q_u, e0)
        
        return e_, q_, i_, g_, alpha
            
            
class EmotionGRU(nn.Module):
    def __init__(self, D_u, D_g, D_s, D_i, D_e, D_a=100, listener_state=False):
        super().__init__()
        
        self.D_u = D_u
        self.D_g = D_g
        self.D_s = D_s
        self.D_i = D_i
        self.D_e = D_e
        self.D_a = D_a
        
        self.listener_state = listener_state
        
        self.cell = EmotionGRUCell(D_u, D_g, D_s, D_i, D_e, D_a, listener_state)
    
    def forward(self, U, qmask):
        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(U.type()) # batch, party, D_p
        e_ = torch.zeros(0).type(U.type()) # batch, D_e
        i_ = torch.zeros(0).type(U.type()) # batch, D_i
        e = e_
        alpha = []
        
        for u_, qmask_ in zip(U, qmask):
            e_, q_, i_, g_, alpha_ = self.cell(u_, qmask_, g_hist, i_, e_, q_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])
            
        return e,alpha # seq_len, batch, D_e

class BiModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', 
                 D_a=100, 
                 dropout_rec=0.5,
                 dropout=0.5):
        super(BiModel, self).__init__()

        self.D_m       = D_m
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        self.dialog_rnn_f = EmotionGRU(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = EmotionGRU(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear     = nn.Linear(2*D_e, 2*D_h)
        self.smax_fc    = nn.Linear(2*D_h, n_classes)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        
        hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        return log_prob, [], alpha_f, alpha_b
      