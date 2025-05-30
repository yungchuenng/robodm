"""Benchmark tests comparing VLA, HDF5, and TFDS formats."""

import pytest
import numpy as np
import os
import time
import tempfile
import shutil
from typing import Dict, List, Any, Tuple
import csv
from dataclasses import dataclass

from fog_x import Trajectory
from .test_fixtures import BenchmarkDataset


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    format_name: str
    file_size_mb: float
    creation_time_sec: float
    loading_time_sec: float
    data_size_mb: float
    compression_ratio: float
    num_samples: int


class BenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.results: List[BenchmarkResult] = []
    
    def create_test_data(self, num_samples: int = 100) -> Dict[str, List[Any]]:
        """Create standardized test data for benchmarking."""
        np.random.seed(42)  # For reproducible results
        
        return {
            "observation/image": [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                for _ in range(num_samples)
            ],
            "observation/depth": [
                np.random.random((480, 640)).astype(np.float32) 
                for _ in range(num_samples)
            ],
            "observation/joint_positions": [
                np.random.random(7).astype(np.float32) 
                for _ in range(num_samples)
            ],
            "observation/joint_velocities": [
                np.random.random(7).astype(np.float32) 
                for _ in range(num_samples)
            ],
            "action": [
                np.random.random(7).astype(np.float32) 
                for _ in range(num_samples)
            ],
            "reward": [
                np.float32(np.random.random()) 
                for _ in range(num_samples)
            ],
            "episode_id": [
                i // 10 for i in range(num_samples)  # Group into episodes
            ],
        }
    
    def calculate_data_size(self, data: Dict[str, List[Any]]) -> float:
        """Calculate the uncompressed size of data in MB."""
        total_bytes = 0
        for key, values in data.items():
            if isinstance(values[0], np.ndarray):
                array = np.stack(values)
                total_bytes += array.nbytes
            else:
                # Estimate for scalars
                total_bytes += len(values) * 4  # Assume 4 bytes per scalar
        return total_bytes / (1024 * 1024)
    
    def benchmark_vla_format(
        self, 
        data: Dict[str, List[Any]], 
        video_codec: str = "ffv1"
    ) -> BenchmarkResult:
        """Benchmark VLA format creation and loading."""
        format_name = f"VLA_{video_codec}"
        path = os.path.join(self.temp_dir, f"{format_name.lower()}.vla")
        
        # Benchmark creation
        start_time = time.time()
        traj = Trajectory.from_dict_of_lists(
            data, 
            path, 
            video_codec=video_codec
        )
        creation_time = time.time() - start_time
        
        # Get file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Benchmark loading
        start_time = time.time()
        loaded_traj = Trajectory(path, mode="r")
        loaded_data = loaded_traj.load()
        loading_time = time.time() - start_time
        
        # Calculate metrics
        data_size_mb = self.calculate_data_size(data)
        compression_ratio = data_size_mb / file_size_mb if file_size_mb > 0 else 0
        
        return BenchmarkResult(
            format_name=format_name,
            file_size_mb=file_size_mb,
            creation_time_sec=creation_time,
            loading_time_sec=loading_time,
            data_size_mb=data_size_mb,
            compression_ratio=compression_ratio,
            num_samples=len(data[list(data.keys())[0]])
        )
    
    def benchmark_hdf5_format(self, data: Dict[str, List[Any]]) -> BenchmarkResult:
        """Benchmark HDF5 format creation and loading."""
        path = os.path.join(self.temp_dir, "hdf5.h5")
        
        # Benchmark creation
        start_time = time.time()
        BenchmarkDataset.create_hdf5_dataset(path, data)
        creation_time = time.time() - start_time
        
        # Get file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Benchmark loading
        start_time = time.time()
        import h5py
        with h5py.File(path, 'r') as f:
            loaded_data = {key: np.array(f[key]) for key in f.keys()}
        loading_time = time.time() - start_time
        
        # Calculate metrics
        data_size_mb = self.calculate_data_size(data)
        compression_ratio = data_size_mb / file_size_mb if file_size_mb > 0 else 0
        
        return BenchmarkResult(
            format_name="HDF5",
            file_size_mb=file_size_mb,
            creation_time_sec=creation_time,
            loading_time_sec=loading_time,
            data_size_mb=data_size_mb,
            compression_ratio=compression_ratio,
            num_samples=len(data[list(data.keys())[0]])
        )
    
    def benchmark_tfds_format(self, data: Dict[str, List[Any]]) -> BenchmarkResult:
        """Benchmark TFDS-like format (using TFRecord)."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available for TFDS benchmarking")
        
        path = os.path.join(self.temp_dir, "tfds.tfrecord")
        
        # Benchmark creation
        start_time = time.time()
        self._create_tfrecord(data, path)
        creation_time = time.time() - start_time
        
        # Get file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Benchmark loading
        start_time = time.time()
        loaded_data = self._load_tfrecord(path, data)
        loading_time = time.time() - start_time
        
        # Calculate metrics
        data_size_mb = self.calculate_data_size(data)
        compression_ratio = data_size_mb / file_size_mb if file_size_mb > 0 else 0
        
        return BenchmarkResult(
            format_name="TFDS",
            file_size_mb=file_size_mb,
            creation_time_sec=creation_time,
            loading_time_sec=loading_time,
            data_size_mb=data_size_mb,
            compression_ratio=compression_ratio,
            num_samples=len(data[list(data.keys())[0]])
        )
    
    def _create_tfrecord(self, data: Dict[str, List[Any]], path: str):
        """Create TFRecord file from data."""
        import tensorflow as tf
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
        with tf.io.TFRecordWriter(path) as writer:
            num_samples = len(data[list(data.keys())[0]])
            for i in range(num_samples):
                features = {}
                for key, values in data.items():
                    value = values[i]
                    if isinstance(value, np.ndarray):
                        if value.dtype == np.uint8:
                            features[key] = _bytes_feature(value.tobytes())
                        else:
                            features[key] = _float_feature(value)
                    else:
                        features[key] = _int64_feature(int(value))
                
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
    
    def _load_tfrecord(self, path: str, original_data: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
        """Load TFRecord file."""
        import tensorflow as tf
        
        # Create feature description
        feature_description = {}
        sample_data = {k: v[0] for k, v in original_data.items()}
        
        for key, value in sample_data.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.uint8:
                    feature_description[key] = tf.io.FixedLenFeature([], tf.string)
                else:
                    feature_description[key] = tf.io.FixedLenFeature([value.size], tf.float32)
            else:
                feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
        
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(_parse_function)
        
        # Convert to numpy
        loaded_data = {}
        all_examples = list(dataset.as_numpy_iterator())
        
        for key in feature_description.keys():
            if key in sample_data and isinstance(sample_data[key], np.ndarray):
                if sample_data[key].dtype == np.uint8:
                    # Reconstruct image data
                    arrays = []
                    for example in all_examples:
                        array = np.frombuffer(example[key], dtype=np.uint8).reshape(sample_data[key].shape)
                        arrays.append(array)
                    loaded_data[key] = np.stack(arrays)
                else:
                    # Reconstruct float data
                    arrays = []
                    for example in all_examples:
                        array = example[key].reshape(sample_data[key].shape)
                        arrays.append(array)
                    loaded_data[key] = np.stack(arrays)
            else:
                # Scalar data
                loaded_data[key] = np.array([example[key] for example in all_examples])
        
        return loaded_data
    
    def run_comprehensive_benchmark(self, num_samples: int = 100) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across all formats."""
        data = self.create_test_data(num_samples)
        results = []
        
        # Benchmark VLA formats
        results.append(self.benchmark_vla_format(data, video_codec="ffv1"))
        results.append(self.benchmark_vla_format(data, video_codec="h264"))
        
        # Benchmark HDF5
        results.append(self.benchmark_hdf5_format(data))
        
        # Benchmark TFDS (if available)
        try:
            results.append(self.benchmark_tfds_format(data))
        except Exception as e:
            print(f"Skipping TFDS benchmark: {e}")
        
        self.results.extend(results)
        return results
    
    def save_results_csv(self, filename: str):
        """Save benchmark results to CSV."""
        if not self.results:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'Format', 'File Size (MB)', 'Creation Time (s)', 'Loading Time (s)',
                'Data Size (MB)', 'Compression Ratio', 'Num Samples'
            ])
            # Data
            for result in self.results:
                writer.writerow([
                    result.format_name,
                    f"{result.file_size_mb:.2f}",
                    f"{result.creation_time_sec:.3f}",
                    f"{result.loading_time_sec:.3f}",
                    f"{result.data_size_mb:.2f}",
                    f"{result.compression_ratio:.2f}",
                    result.num_samples
                ])


class TestBenchmark:
    """Test class for benchmarking functionality."""
    
    def test_benchmark_small_dataset(self, temp_dir):
        """Benchmark with small dataset for fast testing."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=10)
        
        assert len(results) >= 3  # At least VLA lossless, VLA lossy, HDF5
        
        # Check that all results have reasonable values
        for result in results:
            assert result.file_size_mb > 0
            assert result.creation_time_sec >= 0
            assert result.loading_time_sec >= 0
            assert result.data_size_mb > 0
            assert result.compression_ratio > 0
            assert result.num_samples == 10
    
    def test_benchmark_medium_dataset(self, temp_dir):
        """Benchmark with medium dataset."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=50)
        
        # Check that results scale appropriately
        for result in results:
            assert result.file_size_mb > 0
            assert result.num_samples == 50
        
        # VLA lossy should generally be smaller than lossless
        vla_lossy = next((r for r in results if r.format_name == "VLA_h264"), None)
        vla_lossless = next((r for r in results if r.format_name == "VLA_ffv1"), None)
        
        if vla_lossy and vla_lossless:
            # Lossy should have higher compression ratio (smaller file)
            assert vla_lossy.compression_ratio >= vla_lossless.compression_ratio * 0.8
    
    @pytest.mark.slow
    def test_benchmark_large_dataset(self, temp_dir):
        """Benchmark with large dataset (marked as slow test)."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=200)
        
        # Save results for analysis
        csv_path = os.path.join(temp_dir, "benchmark_results.csv")
        runner.save_results_csv(csv_path)
        assert os.path.exists(csv_path)
        
        # Analyze results
        for result in results:
            print(f"{result.format_name}: "
                  f"Size={result.file_size_mb:.2f}MB, "
                  f"Creation={result.creation_time_sec:.3f}s, "
                  f"Loading={result.loading_time_sec:.3f}s, "
                  f"Compression={result.compression_ratio:.2f}x")
    
    def test_benchmark_compression_ratios(self, temp_dir):
        """Test compression ratios across formats."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=30)
        
        # All formats should achieve some compression
        for result in results:
            assert result.compression_ratio >= 1.0, f"{result.format_name} has compression ratio < 1.0"
    
    def test_benchmark_loading_speed_comparison(self, temp_dir):
        """Compare loading speeds across formats."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=50)
        
        # Find the fastest loading format
        fastest = min(results, key=lambda r: r.loading_time_sec)
        slowest = max(results, key=lambda r: r.loading_time_sec)
        
        print(f"Fastest loading: {fastest.format_name} ({fastest.loading_time_sec:.3f}s)")
        print(f"Slowest loading: {slowest.format_name} ({slowest.loading_time_sec:.3f}s)")
        
        # Ensure reasonable loading times (should load 50 samples in under 10 seconds)
        for result in results:
            assert result.loading_time_sec < 10.0, f"{result.format_name} takes too long to load"
    
    def test_benchmark_creation_speed_comparison(self, temp_dir):
        """Compare creation speeds across formats."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=50)
        
        # Find the fastest creation format
        fastest = min(results, key=lambda r: r.creation_time_sec)
        slowest = max(results, key=lambda r: r.creation_time_sec)
        
        print(f"Fastest creation: {fastest.format_name} ({fastest.creation_time_sec:.3f}s)")
        print(f"Slowest creation: {slowest.format_name} ({slowest.creation_time_sec:.3f}s)")
        
        # Ensure reasonable creation times
        for result in results:
            assert result.creation_time_sec < 30.0, f"{result.format_name} takes too long to create"
    
    def test_benchmark_file_size_comparison(self, temp_dir):
        """Compare file sizes across formats."""
        runner = BenchmarkRunner(temp_dir)
        results = runner.run_comprehensive_benchmark(num_samples=50)
        
        # Find the most and least space-efficient formats
        smallest = min(results, key=lambda r: r.file_size_mb)
        largest = max(results, key=lambda r: r.file_size_mb)
        
        print(f"Smallest file: {smallest.format_name} ({smallest.file_size_mb:.2f}MB)")
        print(f"Largest file: {largest.format_name} ({largest.file_size_mb:.2f}MB)")
        
        # VLA lossy should generally be among the smallest
        vla_lossy = next((r for r in results if r.format_name == "VLA_h264"), None)
        if vla_lossy:
            # Should be in the smaller half of file sizes
            median_size = sorted([r.file_size_mb for r in results])[len(results)//2]
            assert vla_lossy.file_size_mb <= median_size
    
    def test_benchmark_scalability(self, temp_dir):
        """Test how formats scale with dataset size."""
        runner = BenchmarkRunner(temp_dir)
        
        sizes = [10, 20, 40]
        all_results = []
        
        for size in sizes:
            results = runner.run_comprehensive_benchmark(num_samples=size)
            all_results.extend(results)
        
        # Group results by format
        format_results = {}
        for result in all_results:
            if result.format_name not in format_results:
                format_results[result.format_name] = []
            format_results[result.format_name].append(result)
        
        # Check that file sizes scale roughly linearly with data size
        for format_name, results in format_results.items():
            if len(results) >= 2:
                results_sorted = sorted(results, key=lambda r: r.num_samples)
                
                # File size should increase with more samples
                for i in range(1, len(results_sorted)):
                    assert results_sorted[i].file_size_mb >= results_sorted[i-1].file_size_mb


class TestBenchmarkIntegration:
    """Integration tests for benchmarking with real-world scenarios."""
    
    def test_robotics_dataset_benchmark(self, temp_dir):
        """Benchmark with realistic robotics dataset structure."""
        runner = BenchmarkRunner(temp_dir)
        
        # Create realistic robotics data
        data = {
            "observation/rgb_image": [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                for _ in range(25)
            ],
            "observation/depth_image": [
                np.random.random((224, 224)).astype(np.float32) 
                for _ in range(25)
            ],
            "observation/robot_state": [
                np.random.random(14).astype(np.float32)  # 7 joints + 7 velocities
                for _ in range(25)
            ],
            "action": [
                np.random.random(7).astype(np.float32) 
                for _ in range(25)
            ],
            "language_instruction": [
                f"instruction_{i}" for i in range(25)
            ],
        }
        
        results = []
        
        # Test each format
        for video_codec in ["ffv1", "h264"]:
            result = runner.benchmark_vla_format(data, video_codec=video_codec)
            results.append(result)
        
        result = runner.benchmark_hdf5_format(data)
        results.append(result)
        
        # Verify results are reasonable for robotics data
        for result in results:
            assert result.file_size_mb > 0
            assert result.compression_ratio >= 1.0
            # Images should compress well
            if "image" in str(data.keys()):
                assert result.compression_ratio > 2.0 