# 🦊 Robo-DM

**An Efficient and Scalable Data Collection and Management Framework For Robotics Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/BerkeleyAutomation/fog_x)](LICENSE)
[![Tests](https://github.com/BerkeleyAutomation/fog_x/workflows/Tests/badge.svg)](https://github.com/BerkeleyAutomation/fog_x/actions)

fog_x is a high-performance robotics data management framework that enables efficient collection, storage, and retrieval of multimodal robotics trajectories. Built with speed 🚀 and memory efficiency 📈 in mind, fog_x provides native support for various robotics data formats and cloud storage systems.

## ✨ Key Features

- **🚀 High Performance**: Optimized for speed with active metadata and lazily-loaded trajectory data
- **📈 Memory Efficient**: Smart data loading and compression strategies minimize memory usage
- **🎥 Advanced Video Compression**: Support for multiple codecs (H.264, H.265, AV1, FFV1) with automatic codec selection
- **☁️ Cloud Native**: Built-in support for cloud storage systems (AWS S3, etc.)
- **🔄 Format Compatibility**: Native support for Open-X-Embodiment, HuggingFace datasets, RLDS, and HDF5
- **🎯 Flexible Data Types**: Handle images, videos, sensor data, and custom features seamlessly
- **🏗️ Distributed Ready**: Flexible dataset partitioning for distributed training workflows
- **🧪 Test Coverage**: Comprehensive test suite with benchmarking capabilities

## 🛠️ Installation

### Basic Installation

```bash
git clone https://github.com/BerkeleyAutomation/fog_x.git
cd fog_x
pip install -e .
```

### Installation with Optional Dependencies

```bash
# For HuggingFace integration
pip install -e .[hf]

# For Open-X-Embodiment support
pip install -e .[rtx]

# For AWS cloud storage
pip install -e .[aws]

# For PyTorch integration
pip install -e .[torch]

# Install all optional dependencies
pip install -e .[all]
```

## 🚀 Quick Start

### Basic Data Collection and Loading

```python
import numpy as np
import fog_x

# Create a new trajectory for data collection
trajectory = fog_x.Trajectory(path="/tmp/robot_demo.vla", mode="w")

# Collect multimodal robotics data
for step in range(100):
    # Add camera observations
    trajectory.add("camera/rgb", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    trajectory.add("camera/depth", np.random.rand(480, 640).astype(np.float32))
    
    # Add robot state
    trajectory.add("robot/joint_positions", np.random.rand(7).astype(np.float32))
    trajectory.add("robot/joint_velocities", np.random.rand(7).astype(np.float32))
    trajectory.add("robot/end_effector_pose", np.random.rand(4, 4).astype(np.float32))
    
    # Add action data
    trajectory.add("action/gripper_action", np.random.rand(1).astype(np.float32))

# Save and close the trajectory
trajectory.close()

# Load the trajectory for training
trajectory = fog_x.Trajectory(path="/tmp/robot_demo.vla", mode="r")
data = trajectory.load()

print(f"Loaded trajectory with {len(data['camera/rgb'])} timesteps")
print(f"Camera RGB shape: {data['camera/rgb'][0].shape}")
print(f"Joint positions shape: {data['robot/joint_positions'][0].shape}")
```

### Batch Data Creation

```python
import fog_x

# Create trajectory from dictionary of lists
data = {
    "observation/image": [np.random.randint(0, 255, (224, 224, 3)) for _ in range(50)],
    "observation/state": [np.random.rand(10) for _ in range(50)],
    "action": [np.random.rand(7) for _ in range(50)],
}

trajectory = fog_x.Trajectory.from_dict_of_lists(
    data=data,
    path="/tmp/batch_trajectory.vla",
    video_codec="libaom-av1"  # Use AV1 codec for efficient compression
)
```

### Advanced Configuration

```python
import fog_x

# Configure video compression settings
trajectory = fog_x.Trajectory(
    path="/tmp/compressed_demo.vla",
    mode="w",
    video_codec="libx265",  # Use H.265 codec
    codec_options={
        "crf": "23",        # Quality setting (lower = higher quality)
        "preset": "fast"    # Encoding speed
    }
)

# Use hierarchical feature names
trajectory.add("sensors/lidar/points", lidar_data)
trajectory.add("sensors/camera/front/rgb", front_camera)
trajectory.add("sensors/camera/wrist/rgb", wrist_camera)
trajectory.add("control/arm/joint_positions", joint_positions)
```

## 📊 Data Loaders

fog_x includes specialized loaders for common robotics datasets:

### HDF5 Loader

```python
from fog_x.loader import HDF5Loader

# Convert HDF5 datasets to fog_x format
loader = HDF5Loader()
loader.convert_to_trajectory(
    input_path="/path/to/dataset.h5",
    output_path="/path/to/output.vla"
)
```

### RLDS (Reverb Dataset) Loader

```python
from fog_x.loader import RLDSLoader

# Load from RLDS format
loader = RLDSLoader()
trajectory = loader.load_from_rlds(
    dataset_path="/path/to/rlds_dataset",
    output_path="/path/to/output.vla"
)
```

### VLA (Video Language Action) Loader

```python
from fog_x.loader import VLALoader

# Efficient VLA data loading
loader = VLALoader()
dataset = loader.load_dataset("/path/to/vla_files")
```

## 🎥 Video Codec Support

fog_x supports multiple video codecs for efficient storage of visual data:

| Codec | Use Case | Compression | Quality |
|-------|----------|-------------|---------|
| `rawvideo` | Lossless, fast I/O | None | Perfect |
| `ffv1` | Lossless compression | High | Perfect |
| `libx264` | General purpose | Very High | Excellent |
| `libx265` | Better compression | Very High | Excellent |
| `libaom-av1` | Best compression | Highest | Excellent |
| `auto` | Automatic selection | Optimal | Optimal |

```python
# Automatic codec selection based on data characteristics
trajectory = fog_x.Trajectory(path="auto.vla", mode="w", video_codec="auto")

# Manual codec selection for specific needs
trajectory = fog_x.Trajectory(path="lossless.vla", mode="w", video_codec="ffv1")
```

## ☁️ Cloud Storage Integration

```python
import fog_x

# Direct S3 integration (requires aws optional dependencies)
trajectory = fog_x.Trajectory(
    path="s3://my-bucket/trajectories/demo.vla",
    mode="w"
)

# Add data as usual
trajectory.add("observation", image_data)
trajectory.close()

# Load from cloud storage
trajectory = fog_x.Trajectory(
    path="s3://my-bucket/trajectories/demo.vla",
    mode="r"
)
data = trajectory.load()
```

## 🏭 Factory Pattern for Advanced Use Cases

```python
from fog_x import TrajectoryFactory

# Create factory with custom dependencies
factory = TrajectoryFactory(
    filesystem=custom_filesystem,
    time_provider=custom_timer
)

# Create trajectories with dependency injection
trajectory = factory.create_trajectory(
    path="/tmp/test.vla",
    mode="w",
    video_codec="libaom-av1"
)
```

## 🔧 API Reference

### Core Classes

- **`Trajectory`**: Main class for data collection and loading
- **`FeatureType`**: Type system for trajectory features
- **`TrajectoryFactory`**: Factory for creating trajectory instances
- **`CodecConfig`**: Video codec configuration management

### Key Methods

- **`add(feature, data, timestamp=None)`**: Add single feature to trajectory
- **`add_by_dict(data, timestamp=None)`**: Add multiple features from dictionary
- **`load(return_type="numpy")`**: Load trajectory data
- **`close(compact=True)`**: Close and optionally compact trajectory
- **`from_dict_of_lists(data, path, ...)`**: Create trajectory from structured data

## 📈 Performance & Benchmarks

Run benchmarks to test performance on your system:

```bash
# Run comprehensive benchmarks
python -m pytest tests/test_trajectory.py::test_benchmark -v

# Run specific codec benchmarks
python tests/benchmark_codecs.py
```

## 🧪 Development & Testing

### Running Tests

```bash
# Install development dependencies
pip install -e .[test]

# Run all tests
make test

# Run specific test categories
pytest tests/test_trajectory.py -v
pytest tests/test_loaders.py -v
```

### Code Quality

```bash
# Format code
make fmt

# Run linters
make lint

# Generate documentation
make docs
```

## 📝 Examples

Explore the `examples/` directory for more detailed usage patterns:

- **[Basic Data Collection](./examples/data_collection_and_load.py)**: Simple data collection and loading
- **[Benchmark Scripts](./tests/)**: Performance testing and optimization

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up development environment
- Running tests and benchmarks  
- Code style and formatting
- Submitting pull requests

## 📄 License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.


## 📚 Citation

If you use fog_x in your research, please cite:

```bibtex
@article{chen2025robo,
  title={Robo-DM: Data Management For Large Robot Datasets},
  author={Chen, Kaiyuan and Fu, Letian and Huang, David and Zhang, Yanxiang and Chen, Lawrence Yunliang and Huang, Huang and Hari, Kush and Balakrishna, Ashwin and Xiao, Ted and Sanketi, Pannag R and others},
  journal={arXiv preprint arXiv:2505.15558},
  year={2025}
}
```
