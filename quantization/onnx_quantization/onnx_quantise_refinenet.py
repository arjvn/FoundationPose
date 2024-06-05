import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load your ONNX model
model_path = 'path_to_save_your_model/model.onnx'
onnx_model = onnx.load(model_path)

# Specify the path for the quantized model
quantized_model_path = 'path_to_save_your_model/quantized_model.onnx'

# Perform quantization
quantized_model = quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8)

print("Quantized model saved.")
