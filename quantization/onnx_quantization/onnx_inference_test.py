import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("path_to_your_model/model.onnx")

# Print model input and output details (optional but helpful for debugging)
print("Model Inputs:")
for input in session.get_inputs():
    print(input.name, input.shape, input.type)

print("Model Outputs:")
for output in session.get_outputs():
    print(output.name, output.shape, output.type)

# Prepare your input data: modify this according to your model's requirements
# Here, I assume the model expects two inputs named 'input_A' and 'input_B'
input_data_A = np.random.randn(1, 4, 224, 224).astype(np.float32)  # Example input
input_data_B = np.random.randn(1, 4, 224, 224).astype(np.float32)  # Example input

# Prepare the input dictionary
input_dict = {
    "input_A": input_data_A,
    "input_B": input_data_B
}

# Run the model (perform inference)
outputs = session.run(None, input_dict)

# The outputs variable is a list of output results corresponding to the model outputs
# Process the results as needed
print("Inference results:", outputs)

# Access specific outputs, e.g., 'output_trans' and 'output_rot'
trans_output = outputs[0]  # Adjust index based on the actual output order
rot_output = outputs[1]   # Adjust index based on the actual output order
print("Translation Output:", trans_output)
print("Rotation Output:", rot_output)
