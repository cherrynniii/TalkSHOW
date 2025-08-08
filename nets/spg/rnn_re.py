import torch
import torch.nn as nn
import torch.nn.functional as F
 

class RNNBlock(nn.Module):
    def __init__(self, dim, residual=True, n_classes=10, bh_model=False):
        super().__init__()
        self.residual = residual
        self.bh_model = bh_model

        # speaker ID 임베딩
        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        # RNN (GRU)
        self.gru = nn.GRU(dim * 2, dim * 2, batch_first=True)

        # class MLP
        self.class_mlp = nn.Linear(512, 256)

    # x: [128, 256, 22, 2]
    def forward(self, x, h):
        x_b = x[:, :, :, 0]     # [128, 256, 22]
        x_h = x[:, :, :, 1]     # [128, 256, 22]
        x = torch.cat([x_b, x_h], dim=1)    # [128, 512, 22]
        x = x.permute(0, 2, 1)      # [128, 22, 512]
        x_gru, _ = self.gru(x)
        
        x_b = x_gru[:, :, :256]     # [128, 22, 256]
        x_h = x_gru[:, :, 256:]     # [128, 22, 256]
        x_gru = torch.stack([x_b, x_h], dim=-1)     # [128, 22, 256, 2]
        x_gru = x_gru.permute(0, 2, 1, 3)   # [128, 256, 22, 2]

        h = self.class_cond_embedding(h.to(self.class_cond_embedding.weight.device))    # [128, 512]
        h = self.class_mlp(h)   # [128, 256]

        out = x_gru + h[:, :, None, None]
        return out


class GatedPixelRNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model
 
        self.embedding = nn.Embedding(input_dim, dim)
        self.class_cond_embedding = nn.Embedding(n_classes, dim)

        self.layers = nn.ModuleList()
 
        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
            self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
 
        for i in range(5):
            residual = False if i == 0 else True

            self.layers.append(
                RNNBlock(dim, residual, n_classes, bh_model)
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
