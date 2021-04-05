import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgEmbeddings(nn.Module):
    """
    Img embeddings for encoder/decoder.
    """
    def __init__(self, hidden_size, model_dim):
        super(ImgEmbeddings, self).__init__()
        dropout = 0.3
        self.pre_dp = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.img_to_emb = nn.Linear(hidden_size, model_dim, bias=True)

    def forward(self, img_feats):
        img_emb = self.pre_dp(img_feats)
        img_emb = F.relu(self.img_to_emb(img_emb))
        return img_emb



