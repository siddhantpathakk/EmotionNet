import torch
import torch.nn as nn


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