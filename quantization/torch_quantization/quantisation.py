import torch
import os
import yaml
from omegaconf import OmegaConf
from learning.models.score_network import ScoreNetMultiPair
from learning.models.refine_network import RefineNet
# Assuming the configuration and model checkpoint paths are correct
root_path = 'weights/2024-01-11-20-02-45/'
config_path = root_path + 'config.yml'
model_path = root_path + 'model_best.pth'

# Load configuration using OmegaConf
cfg = OmegaConf.load(config_path)

# Adjust the configuration as needed for loading
cfg['ckpt_dir'] = model_path
cfg['enable_amp'] = True

# Make sure this matches the configuration used during training
cfg['c_in'] = 6  # Set this according to your checkpoint details

# Initialize the model
print("c_in before model initialization:", cfg['c_in'])
model = ScoreNetMultiPair(cfg=cfg, c_in=cfg['c_in']).cuda()
# model = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()

# Load model weights
ckpt = torch.load(cfg['ckpt_dir'])
if 'model' in ckpt:
    ckpt = ckpt['model']
model.load_state_dict(ckpt)

model.eval()  # Set the model to evaluation mode

model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MultiheadAttention},
    dtype=torch.qint8
)

# Save the quantized model
torch.save(model_quantized.state_dict(), root_path + 'model_quantized.pth')
print(f"Quantization complete and model saved - {root_path}")
