import torch
import torch.nn as nn
import torch.nn.functional as F
 

import math
import torch
import torch.nn as nn
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
 
        # [max_len, d_model] 크기의 포지셔널 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
 
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] → batch 차원 맞추기
        self.register_buffer('pe', pe)  # 학습 대상 아님, 모델과 함께 저장됨
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        # 앞 T개의 포지셔널 인코딩을 더함
        return x + self.pe[:, :T, :]
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, residual=True, n_classes=10, bh_model=False, nhead=8, d_ff=2048, causal=True):
        super().__init__()

        # dim: 256 -> d_model: 512
        self.ln1 = nn.LayerNorm(dim * 2)
        self.attn = nn.MultiheadAttention(dim * 2, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, d_ff), nn.ReLU(True),
            nn.Linear(d_ff, dim * 2)
        )
        self.dropout = nn.Dropout(0.1)
        self.causal = causal

        self.pos_encoding = PositionalEncoding(d_model=dim*2, max_len=128)

        # speaker ID 임베딩
        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        # class MLP
        self.class_mlp = nn.Linear(512, 256)
    
    # x: [128, 256, 22, 2]
    def forward(self, x, h):
        x = x.permute(0, 2, 1, 3)   # x: [128, 22, 256, 2] (트랜스포머 입력은 B, T, D 순이어야 함)
        x_b = x[:, :, :, 0]     # [128, 22, 256]
        x_h = x[:, :, :, 1]     # [128, 22, 256]
        x = torch.cat([x_b, x_h], dim=2)    # [128, 22, 512]
        x = self.pos_encoding(x)

        T = x.size(1)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()

        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(y)
        y = self.ffn(self.ln2(x))
        x = x + self.dropout(y)

        # 다시 모양 복구
        x_b = x[:, :, :256]     # [128, 22, 256]
        x_h = x[:, :, 256:]     # [128, 22, 256]
        x = torch.stack([x_b, x_h], dim=-1)     # [128, 22, 256, 2]
        x = x.permute(0, 2, 1, 3)   # [128, 256, 22, 2]

        h = self.class_cond_embedding(h.to(self.class_cond_embedding.weight.device))    # [128, 512]
        h = self.class_mlp(h)   # [128, 256]

        out = x + h[:, :, None, None]
        return out


class GatedPixelTransformer(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model
 
        self.embedding = nn.Embedding(input_dim, dim)
        self.layers = nn.ModuleList()
 
        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
            self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
 
        for i in range(5):
            residual = False if i == 0 else True

            self.layers.append(
                TransformerBlock(dim, residual, n_classes, bh_model)
            )
 
        self.output_proj_body = nn.Linear(dim, input_dim)
        self.output_proj_hand = nn.Linear(dim, input_dim)
        self.dp = nn.Dropout(0.1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        # RNN 이후 cross interaction을 위한 projection layer
        self.cross_body = nn.Linear(dim, dim)
        self.cross_hand = nn.Linear(dim, dim)
 
    """
        x: [128, 22, 2]
        aud: [128, 256, 22, 2]
    """
    def forward(self, x, label, aud=None):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)   # x: [128, 256, 22, 2]

        # cond = self.class_cond_embedding(label).unsqueeze(1).unsqueeze(1)
        # x = x + cond  # Broadcast add

        for i, layer in enumerate(self.layers):
            if i == 1 and self.audio is True:
                aud = self.embedding_aud(aud)
                a = torch.ones(aud.shape[-2]).to(aud.device)
                a = self.dp(a)
                aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
                x = self.fusion(torch.cat([x, aud], dim=1))
            x = layer(x, label)

        return self.output_conv(x)
 
    def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        if pre_latents is not None:
            x = torch.cat([pre_latents, x], dim=1)
            aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
            h0 = pre_latents.shape[1]
            h = h0 + shape[0]
        else:
            h0 = 0
            h = shape[0]

        for i in range(h0, h):
            for j in range(shape[1]):
                if self.audio:
                    logits = self.forward(x, label, aud_feat)
                else:
                    logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]
