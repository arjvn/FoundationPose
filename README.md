# 🚀 FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects


We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.

<img src="assets/intro.jpg" width="70%">

## 🌟 Quick start

With this one command you can run a demo of the code. If you have your own data repalce the ```TEST_SCENE_DIR``` with the name of your dataset. 
Please place your data in the ```test_data``` folder.

```
make leap-run TEST_SCENE_DIR="avocado_translate_1"
```

- 🥑 avocado_translate_1 is our test dataset. If used the test data is automatically downloaded from here and placed in the ```test_data``` folder by the ```run_container.sh```.
- 🏋️‍♂️ the model files are downloaded from here and placed in the ```weights``` folder by the dockerfile.


## 💡 Functionality

### 1. 🔬 Mesh creation:
This repository contains pure python scripts to create object masks, bounding box labels, and 3D reconstructed object mesh (.ply) for object sequences filmed with an RGB-D camera. This project can prepare training and testing data for various deep learning projects such as 6D object pose estimation projects singleshotpose, and many object detection (e.g., faster rcnn) and instance segmentation (e.g., mask rcnn) projects. Ideally, if you have realsense cameras and have some experience with MeshLab or Blender, creating your customized dataset should be as easy as executing a few command line arguments.

This codes in this repository implement a raw 3D model acquisition pipeline through aruco markers and ICP registration. The raw 3D model obtained needs to be processed and noise-removed in a mesh processing software. After this step, there are functions to generate required labels in automatically.

The codes are currently written for a single object of interest per frame. They can be modified to create a dataset that has several items within a frame.

![cover](assets/cover.png) ![mask](assets/sugar.gif)

A custom mesh of an avocado has been created and provided for the use of this project.

<p style="display: flex; align-items: center; justify-content: space-around;">
  <img src="assets/avodaco_point_cloud_zoomed.jpeg" width="33%" />
  <img src="assets/avodaco_point_mesh.jpeg" width="29%" />
  <img src="assets/avocado_aruco_track.gif" width="33%" />
</p>

### 2. 🎥 Pose Tracking with RGBD Realsense Camera

<img src="assets/track_avocado_translate_1.gif" width="70%">



## 🛠️ Prerequisites / Common Debugging 

Before you begin, ensure your system meets the following requirements to make the most of FoundationPose:

- **🐳 Docker**: You must have Docker installed on your system. For GPU support, ensure you're using Docker Engine version 19.03 or newer, which natively supports NVIDIA GPUs via the NVIDIA Container Toolkit.

- **🏗️ NVIDIA Docker Toolkit***: If you plan to utilize GPU acceleration, install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to allow Docker to utilize the GPU directly.

- **🐍 Python 3.8+**: The software is compatible with Python 3.8 or higher. Ensure that your Python environment is set up correctly, preferably managed through virtual environments like `conda` or `venv`.

- **🖥️ CUDA-Compatible GPU Setup**: A CUDA-compatible NVIDIA GPU is necessary to take full advantage of the foundation model's capabilities. Ensure you have the latest compatible NVIDIA drivers and CUDA version installed that match the toolkit requirements.

### Setting Up Docker for GPU Access

1. **Install NVIDIA Drivers**: Ensure you have the latest NVIDIA drivers installed, compatible with your CUDA version.

2. **Install Docker**: If not already installed, you can download and install Docker from the [official site](https://docs.docker.com/get-docker/).

3. **Set up the NVIDIA Container Toolkit**:
   - Add the NVIDIA package repositories:
     ```bash
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     ```
   - Install the NVIDIA runtime:
     ```bash
     sudo apt-get update
     sudo apt-get install -y nvidia-docker2
     ```
   - Restart the Docker daemon to apply changes:
     ```bash
     sudo systemctl restart docker
     ```

4. **Test NVIDIA Docker Installation**:
   - Run a test container to verify that Docker can access your GPU:
     ```bash
     docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
     ```

   This command should output information about your GPU, confirming that Docker is correctly configured to access the hardware.

### Additional Requirements

- **🌐 Internet Connection**: A stable internet connection is required to download dependencies, Docker images, and datasets.
- **Disk Space**: Ensure you have sufficient disk space, especially for storing large datasets and building Docker images. At least 50 GB of free space is recommended for a full setup.

With these prerequisites met, you can proceed with the installation and setup of the module.

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
    - **Post Quantization**: 2.37s
    - **Without Quantization**: 2.42s

### Key Observations

- The **ScoreNetMultiPair** model accounts for approximately 85% of the total time taken to estimate the first frame, indicating it as the primary target for further optimization.

### Considerations and Future Work

- **Optimization Focus**: Given the significant role of the ScoreNetMultiPair model in first frame processing time, targeted optimizations in its quantization process are necessary.
- **Layer-Specific Quantization**: Developing more refined quantization techniques for attention and transformer layers could mitigate performance losses.
- **Evaluation Metrics**: Additional metrics and profiling are required to understand the bottlenecks and optimize the quantization process effectively.

### Running Dynamic Quantization

There are two parameters which quantize the models at run time:

- Set quantized to True in ```learning.training.predict_pose_refine.PoseRefinePredictor.__init__``` to quantize the PoseRefinePredictor model

```python
class PoseRefinePredictor:
  def __init__(self, quantized=True):
```

- Set quantized to True in ```learning.training.predict_score.ScorePredictor.__init__``` to quantize the PoseRefinePredictor model

```python
class ScorePredictor:
  def __init__(self, amp=True, quantized=True):
```

Note: these have been set to true - merely run python ```run_demo.py``` or any other interface script that you have.

## Method 2: Quantization using ONNX

As an alternative, testing will be carried out via conversion of the models to the ONNX format. This involves three key steps:

1. First the models need to be converted to the ONNX format
2. The ONNX models need to be carefully quantized - noting the different quantization methods and levels of granularity
3. Finally and the most time consuming - the inference script needs to be modifies to allow for use of the ONNX runtime as quantized ONNX models can not directly be used by torch.load().

Task 1 has been completed and the script can be found in  ```quantization/onnx_quantization/create_refinenet_onnx.py```. Work on task 2 and 3 is on going and scripts can be found in the same folder ```quantization/onnx_quantization```. Note that testing is still being carried out on these.

### Testing ONNX Given Torch Quantization Results

Regarding whether testing ONNX is a good idea following the results from torch.quantization, there are several points to consider:

- Comparison of Performance: Even if torch.quantization didn’t significantly enhance performance metrics as expected, ONNX might yield different results due to its distinct execution environment and optimization capabilities. This makes it worth testing.

- Different Quantization Techniques: ONNX provides different quantization approaches (like quantizing operators not supported by PyTorch or using different granularity in quantization), which might prove more effective depending on the model architecture.

- Robustness and Verification: Using ONNX can also be part of a robustness strategy, verifying that the model behaves consistently across different platforms and frameworks, which is crucial for deployment in varied environments.

- Fallback or Alternative Strategy: If torch.quantization falls short in certain aspects, ONNX might offer alternative solutions or workarounds that are more effective for specific models or components of your pipeline.
