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
 
    def forward(self, x, label, aud=None):
        print("0 aud: ", aud.shape)
        # x: (B, 2, T)
        B, H, T = x.shape
        label = label.to(self.class_cond_embedding.weight.device)
        # print("x: ", x.shape)
        x = self.embedding(x.permute(0, 2, 1))  # (B, 2, T, D)
        cond = self.class_cond_embedding(label).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
        x = x + cond  # Broadcast add
        print("x: ", x.shape)
 
        if self.audio and aud is not None:
            # aud: (B, T, 256) → (B, 256, T) → (B, dim, T)
            print("x: ", x.shape)
            print("aud: ", aud.shape)
            aud_feat = self.embedding_aud(aud).permute(0, 2, 1)  # (B, T, dim)
            print("aud_feat: ", aud_feat.shape)
            # print("aud_feat: ", aud_feat.shape)
            aud_feat = self.dp(aud_feat)
            # print("aud_feat: ", aud_feat.shape)
            
            x = x.permute(0, 2, 1, 3)
            print("x[]: ", x[:, 1].shape)
            print("x[]: ", x[:, 0].shape)
            print("aud_feat: ", aud_feat.shape)
            aud_body = self.fusion(torch.cat([x[:, 0], aud_feat], dim=-1))  # (B, T, D)
            aud_hand = self.fusion(torch.cat([x[:, 1], aud_feat], dim=-1))
            x = torch.stack([aud_body, aud_hand], dim=1)
 
        out_body, out_hand = x[:, 0], x[:, 1]  # (B, T, D)
 
        for layer in self.body_rnns:
            out_body, _ = layer(out_body)
        for layer in self.hand_rnns:
            out_hand, _ = layer(out_hand)
 
        logits_body = self.output_proj_body(out_body)  # (B, T, input_dim)
        logits_hand = self.output_proj_hand(out_hand)  # (B, T, input_dim)
        return torch.stack([logits_body, logits_hand], dim=1)  # (B, 2, T, input_dim)
 
    def generate(self, label, shape=(2, 64), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        print("00 audio", aud_feat.shape)
        B = batch_size
        _, T = shape
        x = torch.zeros((B, 2, T), dtype=torch.long, device=next(self.parameters()).device)
 
        for t in range(T):
            if self.audio and aud_feat is not None:
                print("010 audio", aud_feat.shape)
                aud_slice = aud_feat[:, :, :t + 1]
                logits = self.forward(x[:, :, :t + 1], label, aud_slice)
            else:
                logits = self.forward(x[:, :, :t + 1], label)
 
            probs = F.softmax(logits[:, :, -1, :], dim=-1)  # (B, 2, input_dim)
            sampled = torch.multinomial(probs.view(B * 2, -1), 1).view(B, 2)
            x[:, :, t] = sampled
 
        return x