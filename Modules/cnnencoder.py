import torch
import sys
import torch.nn.functional as F
from torch import nn, optim


class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim, kernel_num, kernal_kind, dropout):
        super(CNNEncoder, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_kind = kernal_kind
        if kernal_kind == 0:
            self.cnn1 = nn.Conv2d(1, self.kernel_num, (3, embedding_dim))
            self.cnn2 = nn.Conv2d(1, self.kernel_num, (4, embedding_dim))
            self.cnn3 = nn.Conv2d(1, self.kernel_num, (5, embedding_dim))
        elif kernal_kind == 1:
            self.cnn1 = nn.Conv2d(1, self.kernel_num, (2, embedding_dim))
            self.cnn2 = nn.Conv2d(1, self.kernel_num, (3, embedding_dim))
            self.cnn3 = nn.Conv2d(1, self.kernel_num, (4, embedding_dim))
        else:
            print 'Kernal kind is wrong!'
            sys.exit()
        self.dropout = nn.Dropout(dropout)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, sentences):
        embeds = sentences.unsqueeze(1)
        cnn1_out = self.conv_and_pool(embeds, self.cnn1)
        cnn2_out = self.conv_and_pool(embeds, self.cnn2)
        cnn3_out = self.conv_and_pool(embeds, self.cnn3)
        sent_reps = torch.cat([cnn1_out, cnn2_out, cnn3_out], dim=1)
        sent_reps = self.dropout(sent_reps)
        return sent_reps
