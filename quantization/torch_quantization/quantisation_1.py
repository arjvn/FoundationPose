import torch
import os
from omegaconf import OmegaConf
import logging

from learning.models.score_network import ScoreNetMultiPair

def load_model(cfg, model_path):
    model = ScoreNetMultiPair(cfg=cfg, c_in=cfg.c_in).cuda()
    model.eval()
    
    # Load the pre-trained model weights
    try:
        ckpt = torch.load(model_path)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return None

    return model

def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.ReLU},
        dtype=torch.qint8
    )
    return quantized_model

def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    run_name = "2024-01-11-20-02-45"
    
    original_model_name = 'model_best.pth'
    quantized_model_name = 'model_quantized_1.pth'
    
    config_path = f'weights/{run_name}/config.yml'
    original_model_path = f'weights/{run_name}/{original_model_name}'
    quantized_model_path = f'weights/{run_name}/{quantized_model_name}'

    # Load configuration
    cfg = OmegaConf.load(config_path)
    cfg['ckpt_dir'] = original_model_path  # Make sure to load the original model
    cfg['enable_amp'] = True
    if 'c_in' not in cfg:
        cfg['c_in'] = 6

    model = load_model(cfg, original_model_path)
    if model is not None:
        quantized_model = quantize_model(model)
        torch.save(quantized_model.state_dict(), quantized_model_path)
        logging.info("Model quantization completed and saved.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()