import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, C):
        scale = self.scalar(C) 
        alpha = F.softmax(scale, dim=0).permute(1,2,0) 
        attn_pool = torch.bmm(alpha, C.transpose(0,1))[:,0,:] 
        return attn_pool, alpha


class EmotionGRUCell(nn.Module):
    """
        This class acts as a unit of EmotionRNN.
    """
    def __init__(self, D_m, D_q, D_g, D_r, D_e, dropout = 0.5, **kwargs):
        super(EmotionGRUCell, self).__init__( **kwargs)
        
        self.D_m = D_m # dimension of utterance 
        self.D_q = D_q # dimension of self state
        self.D_g = D_g # dimension of global state
        self.D_r = D_r # dimension of intra speaker state
        self.D_e = D_e # dimension of emotion state
        self.D_a = D_g # dimension of attention vector
        
        self.g_cell = nn.GRUCell(D_q + D_m + D_r, D_g) # global cell
        
        self.p_cell = nn.GRUCell(D_g + D_m, D_q) # self speaker cell
        
        self.pl_cell = nn.GRUCell(D_g + D_m, D_q) # self listener cell
        
        self.r_cell = nn.GRUCell(D_g + D_m, D_r) # intra-speaker cell
        
        self.rl_cell = nn.GRUCell(D_g + D_m, D_r) # intra-listener cell
        
        self.e_cell = nn.GRUCell(D_m + D_q + D_g, D_e) # emotion cell

        # dropout unit
        self.dropout = nn.Dropout(dropout)

        # attention layer
        self.attention = SimpleAttention(D_g)
        
    def _select_parties(self, X, indexes):
        q0_sel = []
        for idx, j in zip(indexes, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, dim=0)
        return q0_sel
    
    def forward(self, U, qmask, g_hist, q0, r0, e0):

        
        qm_idx = torch.argmax(qmask, dim=1)
        
        q0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)
        
        
        ## global cumulative context state ##
        inp_g = torch.cat([q0_sel, r0_sel, U], dim=1)
        
        if g_hist.size()[0] == 0:
            g_ = self.g_cell(inp_g, torch.zeros(U.size()[0], self.D_g).type(U.type()))
        else:
            g_ = self.g_cell(inp_g, g_hist[-1])
        
        g_ = self.dropout(g_)
        
        ## context attention ##
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_a).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist)
        
        
        ## intra speaker state ##
        inp_r = torch.cat([c_, U], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        rs_ = self.r_cell(inp_r.contiguous().view(-1, self.D_m + self.D_a), r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        rs_ = self.dropout(rs_)
        
        ## Self speaker state ##
        inp_p = torch.cat([U, g_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(inp_p.contiguous().view(-1, self.D_m + self.D_g), q0.view(-1, self.D_q)).view(U.size()[0], -1, self.D_q)
        qs_ = self.dropout(qs_)
        
        ## Intra listener state ##
        U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
        rss_ = self._select_parties(rs_, qm_idx).unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_r)
        inp_rl = torch.cat([rss_, U_], dim=1)
        rl_ = self.rl_cell(inp_rl, r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        rl_ = self.dropout(rl_)
        
        ## Self listener state ##
        ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_q)
        inp_pl = torch.cat([U_,ss_],1)
        ql_ = self.pl_cell(inp_pl, q0.view(-1, self.D_q)).view(U.size()[0], -1, self.D_q)
        ql_ = self.dropout(ql_)
        
        
        ## Final self state ##
        qmask = qmask.unsqueeze(2)
        q_ = ql_ * qmask + qs_ * (1 - qmask)
        r_ = rs_ * qmask + rl_ * (1 - qmask)
        
        
        ## Emotion state ##
        inp_e = torch.cat([U, self._select_parties(q_, qm_idx),  g_], dim=1)
        if e0.size()[0]==0:
            e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type())
            
        e_ = self.e_cell(inp_e, e0)
        e_ = self.dropout(e_)
        
        return g_, q_, r_, e_, alpha
        
        
class EmotionRNN(nn.Module):
    """
    This acts as a wrapper for EmotionGRUCell.
    Use this class to implement the EmotionRNN in the EmoNet.
    """
    def __init__(self, D_m, D_q, D_g, D_r, D_e, dropout = 0.5, **kwargs):
        super(EmotionRNN, self).__init__(**kwargs)
        
        self.D_m = D_m
        self.D_q = D_q
        self.D_g = D_g
        self.D_r = D_r
        self.D_e = D_e
        
        self.dropout = nn.Dropout(dropout)
        
        self.cell = EmotionGRUCell(D_m, D_q, D_g, D_r, D_e, dropout)
        
    def forward(self, U, qmask):
        g_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_q).type(U.type())
        r_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_r).type(U.type())
        e_ = torch.zeros(0).type(U.type())
        
        e = e_
        
        alpha = []
        
        for u_, qmask_ in zip(U, qmask):
            g_, q_, r_, e_, alpha_ = self.cell(u_, qmask_, g_hist, q_, r_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], dim=0)
            e = torch.cat([e, e_.unsqueeze(0)], dim=0)
            
            if type(alpha_)!=type(None):
                alpha.append(alpha_)
        
        return e, alpha
    

class EmotionNet(nn.Module):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting emotions
    using the EmotionRNN and a two-layer MLP.
    """
    def __init__(self, D_m, D_q, D_g, D_r, D_e, D_h, n_classes=7, dropout = 0.5):
        super(EmotionNet, self).__init__()
        self.D_m = D_m
        self.D_q = D_q
        self.D_g = D_g
        self.D_r = D_r
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes

        
        self.dropout = nn.Dropout(dropout)
        
        self.emo_rnn_b = EmotionRNN(D_m, D_q, D_g, D_r, D_e, dropout)
        
        self.emo_rnn_f = EmotionRNN(D_m, D_q, D_g, D_r, D_e, dropout)
        
        # print(f'Linear input: {2 * D_e}')
        self.linear = nn.Linear(2 * D_e, D_h)
        nn.init.kaiming_normal_(self.linear.weight)
        
        self.smax_fc = nn.Linear(D_h, n_classes)
        nn.init.kaiming_normal_(self.smax_fc.weight)
        
    def _reverse_sequence(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, dim=1).int()
        
        xfs = []
        for x, c in zip(X_, mask_sum):
            xfs.append(torch.flip(x[:c], [0]))
            
        return pad_sequence(xfs)
    
    def forward(self, U, qmask, umask):
        
        emotions_f, alpha_f = self.emo_rnn_f(U, qmask) # seq_len, batch, D_e
        
        rev_U = self._reverse_sequence(U, umask)
        rev_qmask = self._reverse_sequence(qmask, umask)

        emotions_b, alpha_b = self.emo_rnn_b(rev_U, rev_qmask)
        emotions_b = self._reverse_sequence(emotions_b, umask)
                        
        # print(f'Emotions forward shape: {emotions_f.shape}')
        # print(f'Emotions backward shape: {emotions_b.shape}')
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        
        # print(f'Emotions shape: {emotions.shape}')
        
        hidden = self.linear(emotions)
        
        hidden = self.dropout(hidden)
        
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        return log_prob, alpha_f, alpha_b