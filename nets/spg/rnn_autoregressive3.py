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
            self.fusion = nn.Linear(dim * 2, dim)
 
        self.body_rnns = nn.ModuleList([
            nn.GRU(dim, dim, batch_first=True) for _ in range(n_layers)
        ])
        self.hand_rnns = nn.ModuleList([
            nn.GRU(dim, dim, batch_first=True) for _ in range(n_layers)
        ])
 
        self.output_proj_body = nn.Linear(dim, input_dim)
        self.output_proj_hand = nn.Linear(dim, input_dim)
        self.dp = nn.Dropout(0.1)

        # RNN 이후 cross interaction을 위한 projection layer
        self.cross_body = nn.Linear(dim, dim)
        self.cross_hand = nn.Linear(dim, dim)
 
    def forward(self, x, label, aud=None):
        # x: (B, 2, T)
        B, H, T = x.shape
        # print("x: ", x.shape)
        x = self.embedding(x.permute(0, 2, 1))  # (B, 2, T, D)
        cond = self.class_cond_embedding(label).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
        x = x + cond  # Broadcast add
 
        if self.audio and aud is not None:
            # aud: (B, T, 256) → (B, 256, T) → (B, dim, T)
            # print("x: ", x.shape)
            # print("aud: ", aud.shape)
            aud_feat = self.embedding_aud(aud).permute(0, 2, 1)  # (B, T, dim)
            # print("aud_feat: ", aud_feat.shape)
            aud_feat = self.dp(aud_feat)
            # print("aud_feat: ", aud_feat.shape)
            aud_body = self.fusion(torch.cat([x[:, 0], aud_feat], dim=-1))  # (B, T, D)
            aud_hand = self.fusion(torch.cat([x[:, 1], aud_feat], dim=-1))
            x = torch.stack([aud_body, aud_hand], dim=1)
            # print("111 x: ", x.shape)
 
        out_body, out_hand = x[:, 0], x[:, 1]  # (B, T, D)
 
        for i in range(len(self.body_rnns)):
            in_body = out_body
            in_hand = out_hand

            out_body, _ = self.body_rnns[i](out_body)
            out_hand, _ = self.hand_rnns[i](out_hand)
            # print("111 out_body: ", out_body.shape)

            mod_body = self.cross_body(out_hand)
            mod_hand = self.cross_hand(out_body)
            # print("111 mod_body: ", out_body.shape)

            out_body = out_body + mod_body
            out_hand = out_hand + mod_hand

            # residual
            out_body = out_body + in_body
            out_hand = out_hand + in_hand
 
        logits_body = self.output_proj_body(out_body)  # (B, T, input_dim)
        logits_hand = self.output_proj_hand(out_hand)  # (B, T, input_dim)

        output = torch.stack([logits_body, logits_hand], dim=2)
        output = output.permute(0, 3, 1, 2)
        # print("output: ", output.shape)
        return output  # (B, 2, T, input_dim)
 
    def generate(self, label, shape=(2, 64), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        B = batch_size
        _, T = shape
        x = torch.zeros((B, 2, T), dtype=torch.long, device=next(self.parameters()).device)
 
        for t in range(T):
            if self.audio and aud_feat is not None:
                logits = self.forward(x[:, :, :t + 1], label, aud_feat[:, :t + 1, :])
            else:
                logits = self.forward(x[:, :, :t + 1], label)
 
            probs = F.softmax(logits[:, :, -1, :], dim=-1)  # (B, 2, input_dim)
            sampled = torch.multinomial(probs.view(B * 2, -1), 1).view(B, 2)
            x[:, :, t] = sampled
 
        return x