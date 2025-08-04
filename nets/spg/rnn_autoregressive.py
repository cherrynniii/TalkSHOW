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

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.dp = nn.Dropout(0.1)
 
    def forward(self, x, label, aud=None):
        aud = aud[:, :, :, 0]
        # print("0 aud: ", aud.shape)     # [128, 256, 22]
        # x: (B, 2, T)
        B, H, T = x.shape
        label = label.to(self.class_cond_embedding.weight.device)
        # print("1 label: ", label.shape)# [128]
        # print("2 x: ", x.shape)# [128, 22, 2]
        x = self.embedding(x)  # (B, 2, T, D)
        # print("3 x: ", x.shape)# [128, 22, 2, 256]
        cond = self.class_cond_embedding(label).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
        # print("4 cond: ", cond.shape)# [128, 1, 1, 256]
        x = x + cond  # Broadcast add
        # print("5 x: ", x.shape)# [128, 22, 2, 256]
 
        if self.audio and aud is not None:
            # aud: (B, T, 256) → (B, 256, T) → (B, dim, T)
            # print("8 aud: ", aud.shape)# [128, 256, 22]
            aud_feat = self.embedding_aud(aud)
            # print("9 aud_feat: ", aud_feat.shape)
            aud_feat = self.dp(aud_feat)
            # print("10 aud_feat: ", aud_feat.shape)
            aud_feat = aud_feat.permute(0, 2, 1)
            
            # print("11 x: ", x.shape)# [128, 22, 2, 256]
            # print("12 x[:, 1]: ", x[:, :, 1].shape)
            # print("13 x[:, 0]: ", x[:, :, 0].shape)
            aud_body = self.fusion(torch.cat([x[:, :, 0], aud_feat], dim=-1))  # (B, T, D)
            aud_hand = self.fusion(torch.cat([x[:, :, 1], aud_feat], dim=-1))
            x = torch.stack([aud_body, aud_hand], dim=2)
 
        out_body, out_hand = x[:, :, 0], x[:, :, 1]  # (B, T, D)
        # print("14 out_body: ", out_body.shape)# [128, 22, 256]
        # print("15 out_hand: ", out_hand.shape)# [128, 22, 256]

        # rnn
        for layer in self.body_rnns:
            out_body, _ = layer(out_body)
        for layer in self.hand_rnns:
            out_hand, _ = layer(out_hand)


        result = torch.stack([out_body, out_hand], dim=2)# [128, 22, 2, 256]
        result = result.permute(0, 3, 1, 2)
        # print("18 result: ", result.shape)
        result = self.output_conv(result)

        return result  # 기존에는 128, 256, 22, 2였음;;
 
    def generate(self, label, shape=(2, 64), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        # print("00 audio", aud_feat.shape)
        B = batch_size
        _, T = shape
        x = torch.zeros((B, 2, T), dtype=torch.long, device=next(self.parameters()).device)
 
        for t in range(T):
            if self.audio and aud_feat is not None:
                # print("010 audio", aud_feat.shape)
                aud_slice = aud_feat[:, :, :t + 1]
                logits = self.forward(x[:, :, :t + 1], label, aud_slice)
            else:
                logits = self.forward(x[:, :, :t + 1], label)
 
            probs = F.softmax(logits[:, :, -1, :], dim=-1)  # (B, 2, input_dim)
            sampled = torch.multinomial(probs.view(B * 2, -1), 1).view(B, 2)
            x[:, :, t] = sampled
 
        return x