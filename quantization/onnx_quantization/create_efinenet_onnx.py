import torch.onnx
import numpy as np
import os 

from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet


print("welcome")
amp = True
run_name = "2023-10-28-18-33-37"
model_name = 'model_best.pth'

cwd = os.getcwd()
ckpt_dir = f'weights/{run_name}/{model_name}'

cfg = OmegaConf.load(f'weights/{run_name}/config.yml')

cfg['ckpt_dir'] = ckpt_dir
cfg['enable_amp'] = True

########## Defaults, to be backward compatible
if 'use_normal' not in cfg:
    cfg['use_normal'] = False
if 'use_mask' not in cfg:
    cfg['use_mask'] = False
if 'use_BN' not in cfg:
    cfg['use_BN'] = False
if 'c_in' not in cfg:
    cfg['c_in'] = 4
if 'crop_ratio' not in cfg or cfg['crop_ratio'] is None:
    cfg['crop_ratio'] = 1.2
if 'n_view' not in cfg:
    cfg['n_view'] = 1
if 'trans_rep' not in cfg:
    cfg['trans_rep'] = 'tracknet'
if 'rot_rep' not in cfg:
    cfg['rot_rep'] = 'axis_angle'
if 'zfar' not in cfg:
    cfg['zfar'] = 3
if 'normalize_xyz' not in cfg:
    cfg['normalize_xyz'] = False
if isinstance(cfg['zfar'], str) and 'inf' in cfg['zfar'].lower():
    cfg['zfar'] = np.inf
if 'normal_uint8' not in cfg:
    cfg['normal_uint8'] = False
# print(f"cfg: \n {OmegaConf.to_yaml(cfg)}")
print("model initialization")
model = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()

A = torch.load('A.pt')
B = torch.load('B.pt')

print(A.shape)
print(B.shape)

torch.onnx.export(model,
	args=(A, B),
	f="refinenet.onnx",
	input_names=["input_1", "input_2"],
	output_names=["output1"])

out = model(A, B)

print(type(out))
print(out)

