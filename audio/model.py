import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import numpy as np, itertools, random, copy, math


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
        self.e_cell = nn.GRUCell(D_m + D_q + D_g, D_e) # emotion cell

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.attention = SimpleAttention(D_g)
        
    def _select_parties(self, X, indexes):
        q0_sel = []
        for idx, j in zip(indexes, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, dim=0)
        return q0_sel
    
    def forward(self, U, qmask, g_hist, q0, r0, e0):
        qm_idx = torch.argmax(qmask, dim=1)
        s0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)
        
        ## global cumulative context state ##
        inp_g = torch.cat([s0_sel, r0_sel, U], dim=1)
        if g_hist.size()[0] == 0:
            g_ = self.g_cell(inp_g, torch.zeros(U.size()[0], self.D_g).type(U.type()))
        else:
            g_ = self.g_cell(inp_g, g_hist[-1])
            
        ## context attention ##
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_a).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist)
        
        
        ## intra speaker state ##
        inp_r = torch.cat([c_, U], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        rs_ = self.r_cell(inp_r.contiguous().view(-1, self.D_m + self.D_a), r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        
        ## Self speaker state ##
        inp_p = torch.cat([g_, U], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        qs_ = self.p_cell(inp_p.contiguous().view(-1, self.D_m + self.D_g), s0_sel.view(-1, self.D_q)).view(U.size()[0], -1, self.D_q)
        
        ## Self listener state ##
        ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_q)
        inp_pl = torch.cat([], dim=1)
        ql_ = self.pl_cell().view(U.size()[0], -1, self.D_q)
    
    
        qmask = qmask.unsqueeze(2)
        q_ = ql_ * qmask + qs_ * (1 - qmask)
        r_ = rs_ * qmask
        
        ## Emotion state ##
        inp_e = torch.cat([U, self._select_parties(q_, qm_idx), self._select_parties(g_, qm_idx)], dim=1)
        if e0.size()[0]==0:
            e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type())
        e_ = self.e_cell(inp_e, e0)
        
        ## dropout ##
        g_ = self.dropout1(g_)
        q_ = self.dropout2(q_)
        r_ = self.dropout3(r_)
        e_ = self.dropout4(e_)
        
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
    

class EmoNet(nn.Module):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting emotions
    using the EmotionRNN and a two-layer MLP.
    """
    def __init__(self, D_m, D_q, D_g, D_r, D_e, D_h, n_classes=7, residual=True, norm=0, dropout = 0.5, **kwargs):
        super(EmoNet, self).__init__(**kwargs)
        
        D_x = D_m
        
        self.norm_strategy = norm
        self.linear_in = nn.Linear(D_x, D_h)
        self.residual = residual
        
        self.r_weights = nn.Parameter(torch.Tensor([0.25, 0.25, 0.25, 0.25]))
        
        norm_train = True
        self.norm1a = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1b = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        
        self.norm3a = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(D_m, affine=norm_train)
        
        self.dropout1a = nn.Dropout(dropout)
        
        self.emo_rnn_b = EmotionRNN(D_m, D_q, D_g, D_r, D_e, dropout)
        self.emo_rnn_f = EmotionRNN(D_m, D_q, D_g, D_r, D_e, dropout)
        
        self.linear = nn.Linear(2 * D_e, n_classes)
        self.softmax_fc = nn.Linear(D_h, n_classes)
        
    def _reverse_sequence(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, dim=1).int()
        
        xfs = []
        for x, c in zip(X_, mask_sum):
            xfs.append(torch.flip(x[:c], [0]))
            
        return pad_sequence(xfs)
    
    def forward(self, r1, r2, r3, r4, qmask, umask, return_hidden=False):
        
        seq_len, batch, feature_dim = r1.size()
        
        if self.norm_strategy == 1:
            r1 = self.norm1a(r1.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r2 = self.norm1b(r2.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r3 = self.norm1c(r3.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r4 = self.norm1d(r4.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            
        elif self.norm_strategy == 2:
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=True)
            r1 = norm2(r1.transpose(0,1)).transpose(0,1)
            r2 = norm2(r2.transpose(0,1)).transpose(0,1)
            r3 = norm2(r3.transpose(0,1)).transpose(0,1)
            r4 = norm2(r4.transpose(0,1)).transpose(0,1)
            
        elif self.norm_strategy == 3:
            r1 = self.norm3a(r1.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r2 = self.norm3b(r2.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r3 = self.norm3c(r3.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r4 = self.norm3d(r4.transpose(0,1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            
        r = torch.cat([r1, r2, r3, r4], axis = -1)
        
        r = self.linear_in(r)
        
        emotions_f, alpha_f = self.emo_rnn_f(r, qmask)
        
        rev_r = self._reverse_sequence(r, umask)
        rev_qmask = self._reverse_sequence(qmask, umask)
        emotions_b, alpha_b = self.emo_rnn_b(rev_r, rev_qmask)
        
        emotions_b = self._reverse_sequence(emotions_b, umask)
        
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        
        
        hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        
        if self.residual:
            hidden = hidden + r
        
        log_prob = F.log_softmax(self.softmax_fc(hidden), 2)
        
        if return_hidden:
            return hidden, alpha_f, alpha_b, emotions
        else:
            return log_prob, alpha_f, alpha_b, emotions