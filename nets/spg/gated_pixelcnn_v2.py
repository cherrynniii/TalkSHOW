import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.bh_model = bh_model

        # 클래스 라벨을 2 * dim 차원의 벡터로 임베딩 (이후 조건 정보가 됨)
        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # 커널 크기
        padding_shp = (kernel // 2, 1 if self.bh_model else 0)  # 패딩 크기

        # 이전 시간 방향 정보만 보게끔 하는 conv (윗 부분 정보를 이용한 conv)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        # vertical 정보를 horizontal에 연결할 때 사용하는 1x1 conv
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, 2)
        padding_shp = (0, 1)

        # 왼쪽에서 오른쪽으로 정보를 처리하는 conv (왼쪽 정보를 이용한 conv)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        # tanh와 sigmoid로 나눈 뒤 곱함
        self.gate = GatedActivation()

    # 마지막 row/column을 마스킹해서 미래 정보 보지 않게 만듦
    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    # x_v: vertical 입력, x_h: horizontal 입력, h: 클래스 레이블
    def forward(self, x_v, x_h, h):
        print("==========GatedMAskedConv2d==========")
        # 마스크 A일 때는 현재 위치보다 앞에 있는 정보만 보게끔 마스킹
        if self.mask_type == 'A':
            self.make_causal()

        print("h: ", h.shape)
        h = self.class_cond_embedding(h.to(self.class_cond_embedding.weight.device))    # 클래스 조건 임베딩
        print("h: ", h.shape)
        h_vert = self.vert_stack(x_v)   # vertical 처리
        print("h_vert: ", h_vert.shape)
        h_vert = h_vert[:, :, :x_v.size(-2), :]     # 크기 맞춰줌
        print("h_vert: ", h_vert.shape)
        out_v = self.gate(h_vert + h[:, :, None, None])     # 조건 정보 더하고 게이팅
        print("out_v: ", out_v.shape)

        # horizontal 처리
        if self.bh_model:
            h_horiz = self.horiz_stack(x_h)
            print("h_vert: ", h_vert.shape)
            h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
            print("h_vert: ", h_vert.shape)
            v2h = self.vert_to_horiz(h_vert)    # vertical 정보를 반영해 전달
            print("v2h: ", v2h.shape)

            out = self.gate(v2h + h_horiz + h[:, :, None, None])
            print("out: ", out.shape)
            if self.residual:
                out_h = self.horiz_resid(out) + x_h
                print("out_h: ", out_h.shape)
            else:
                out_h = self.horiz_resid(out)
                print("out_h: ", out_h.shape)
        else:
            if self.residual:
                out_v = self.horiz_resid(out_v) + x_v
                print("out_v: ", out_v.shape)
            else:
                out_v = self.horiz_resid(out_v)
                print("out_v: ", out_v.shape)
            out_h = out_v

        return out_v, out_h


"""
    input_dim: 코드북 크기 등
    dim: 임베딩 차원
    n_layers: GatedMaskedConv2d 블록 개수
    n_classes: speaker ID 클래스 수
    audio=True: 오디오 조건을 사용할지 여부
    bh_model: body-hand 구조인지 여부
"""
class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model

        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)   # 오디오 특징 채널 줄이기
            self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
            self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        # 마스크된 CNN 블록들 생성
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
            )

        # Add the output layer 출력: (B, input_dim, H, W) 각 위치에서 input_dim 코드북 확률 분포 예측
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

        self.dp = nn.Dropout(0.1)

    def forward(self, x, label, aud=None):
        print("==========GatedPixelCNN==========")
        print("x: ", x.shape)
        # 입력 인덱스를 임베딩으로 변환
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        print("x: ", x.shape)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)
        print("x: ", x.shape)

        x_v, x_h = (x, x)
        print("x_v: ", x_v.shape)
        print("x_h: ", x_h.shape)
        for i, layer in enumerate(self.layers):
            # 오디오 정보 추가
            if i == 1 and self.audio is True:
                aud = self.embedding_aud(aud)
                print("aud: ", aud.shape)
                a = torch.ones(aud.shape[-2]).to(aud.device)
                print("a: ", a.shape)
                a = self.dp(a)
                print("a: ", a.shape)
                aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
                print("aud: ", aud.shape)
                x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
                print("x_v: ", x_v.shape)
                if self.bh_model:
                    x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
                    print("x_h: ", x_h.shape)
            # PixelCNN 블록 적용
            x_v, x_h = layer(x_v, x_h, label)
            print("x_v: ", x_v.shape)
            print("x_h: ", x_h.shape)

        if self.bh_model:
            return self.output_conv(x_h)
        else:
            return self.output_conv(x_v)

    # Autoregressive하게 시퀀스 생성
    def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        print("=====generate=====")
        param = next(self.parameters())

        # PixelCNN이 생성해나갈 latent token map (인덱스 시퀀스), 즉 토큰 그리드
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        print("x: ", x.shape)
        if pre_latents is not None:
            x = torch.cat([pre_latents, x], dim=1)
            print("x: ", x.shape)
            aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
            print("aud_feat: ", aud_feat.shape)
            h0 = pre_latents.shape[1]
            print("h0: ", h0.shape)
            h = h0 + shape[0]
            print("h: ", h.shape)
        else:
            h0 = 0
            h = shape[0]

        # 한 칸씩 예측 -> softmax로 확률 -> multinomial로 샘플링
        for i in range(h0, h):  # 오디오 시퀀스 범위
            for j in range(shape[1]):   # 바디 코드북 인덱스, hand 코드북 인덱스
                if self.audio:
                    logits = self.forward(x, label, aud_feat)
                else:
                    logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(      # x에 토큰 인덱스를 샘플링하여 채워넣음
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]
