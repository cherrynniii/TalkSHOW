from nets.spg.transformer_autoregressive import GatedPixelTransformer
 
model = GatedPixelTransformer(
    input_dim=2048,
    dim=256,
    n_layers=15,
    n_classes=4,
    audio=True,
    bh_model=True
)
 
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
print(f"Total: {total_params:,}")
print(f"Trainable: {trainable_params:,}")