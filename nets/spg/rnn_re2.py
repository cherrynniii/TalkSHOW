import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class GatedPixelRNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model
 
        self.embedding = nn.Embedding(input_dim, dim)
        self.class_cond_embedding = nn.Embedding(n_classes, dim)
 
        if self.audio:
            self.embedding_aud = nn.Conv1d(256, dim, 1)
            self.fusion_b = nn.Linear(dim * 2, dim)
            self.fusion_h = nn.Linear(dim * 2, dim)
 
        self.rnns = nn.ModuleList([
            nn.GRU(dim * 2, dim * 2, batch_first=True) for _ in range(5)
        ])
 
        self.output_proj_body = nn.Linear(dim, input_dim)
        self.output_proj_hand = nn.Linear(dim, input_dim)
        self.dp = nn.Dropout(0.1)

        # RNN 이후 cross interaction을 위한 projection layer
        self.cross_body = nn.Linear(dim, dim)
        self.cross_hand = nn.Linear(dim, dim)
 
    def forward(self, x, label, aud=None):
        B, H, T = x.shape
        x = self.embedding(x.permute(0, 2, 1))
        cond = self.class_cond_embedding(label).unsqueeze(1).unsqueeze(1)
        x = x + cond  # Broadcast add
 
        if self.audio and aud is not None:
            aud_feat = self.embedding_aud(aud).permute(0, 2, 1)
            aud_feat = self.dp(aud_feat)
            aud_body = self.fusion_b(torch.cat([x[:, 0], aud_feat], dim=-1))      # [128, 22, 256]
            aud_hand = self.fusion_h(torch.cat([x[:, 1], aud_feat], dim=-1))      # [128, 22, 256]

        x = torch.cat([aud_body, aud_hand], dim=2)    # [128, 22, 512]
 
        for rnn in self.rnns:
            x, _ = rnn(x)
 
        out_body = x[:, :, :256]     # [128, 22, 256]
        out_hand = x[:, :, 256:]

        logits_body = self.output_proj_body(out_body)
        logits_hand = self.output_proj_hand(out_hand)

        output = torch.stack([logits_body, logits_hand], dim=2)
        output = output.permute(0, 3, 1, 2)

        return output
 
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
