# Model Quantisation

Quantization is a technique to reduce the numerical precision of model weights and activations. This approach can lead to smaller model sizes and faster inference times with minimal loss in accuracy. This README outlines our project's approach and results in quantizing two key models using PyTorch and ONNX frameworks.

## Scope of the Project

The primary goal of this project is to significantly reduce the time taken to process the first frame in our image-based scoring and pose refinement tasks. This is critical for real-time applications where initial latency can detrimentally system responsiveness.

## Method 1: Quantization using `torch.quantization`

### Pre-Quantization of the Model

#### Objective

To decrease model size for efficient deployment, especially in environments with limited storage and processing capabilities.

#### Results

- **ScoreNetMultiPair model**: Reduced from 182MB to 62MB.
- **PoseRefinePredictor model**: Slightly reduced from 63MB to 62MB.

#### Challenges

Quantizing certain layers, especially those involved in detailed computations such as attention heads and transformer layers, has been problematic. These components are critical for the model's accuracy and are sensitive to precision loss.

#### Storage

Quantized models are stored under the `weights` folder as `model_quantized.pth`.

### Dynamic Quantization (At Runtime)

#### Objective

To reduce initialization and runtime processing times dynamically without compromising the responsiveness of the models.

#### Results

- **Initialization Time**:
  - **ScoreNetMultiPair model**: Slightly increased from 0.77s to 0.80s.
  - **PoseRefinePredictor model**: Decreased from 0.27s to 0.23s.

- **Inference Time**:
  - The time taken to process the first frame did not improve as anticipated:
    - **ScoreNetMultiPair**: 2.37s
    - **PoseRefinePredictor**: 2.42s

### Key Observations

- The **ScoreNetMultiPair** model accounts for approximately 85% of the total time taken to estimate the first frame, indicating it as the primary target for further optimization.

### Considerations and Future Work

- **Optimization Focus**: Given the significant role of the ScoreNetMultiPair model in first frame processing time, targeted optimizations in its quantization process are necessary.
- **Layer-Specific Quantization**: Developing more refined quantization techniques for attention and transformer layers could mitigate performance losses.
- **Evaluation Metrics**: Additional metrics and profiling are required to understand the bottlenecks and optimize the quantization process effectively.

## Method 2: Quantization using ONNX

As an alternative, testing will be carried out via conversion of the models to the ONNX format. This involves three key steps:

1. First the models need to be converted to the ONNX format
2. The ONNX models need to be carefully quantized - noting the different quantization methods and levels of granularity
3. Finally and the most time consuming - the inference script needs to be modifies to allow for use of the ONNX runtime as quantized ONNX models can not directly be used by torch.load().

### Testing ONNX Given Torch Quantization Results

Regarding whether testing ONNX is a good idea following the results from torch.quantization, there are several points to consider:

- Comparison of Performance: Even if torch.quantization didn’t significantly enhance performance metrics as expected, ONNX might yield different results due to its distinct execution environment and optimization capabilities. This makes it worth testing.

- Different Quantization Techniques: ONNX provides different quantization approaches (like quantizing operators not supported by PyTorch or using different granularity in quantization), which might prove more effective depending on the model architecture.

- Robustness and Verification: Using ONNX can also be part of a robustness strategy, verifying that the model behaves consistently across different platforms and frameworks, which is crucial for deployment in varied environments.

- Fallback or Alternative Strategy: If torch.quantization falls short in certain aspects, ONNX might offer alternative solutions or workarounds that are more effective for specific models or components of your pipeline.
