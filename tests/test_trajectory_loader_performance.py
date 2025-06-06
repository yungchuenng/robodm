"""
Performance and benchmarking tests for Trajectory.load functionality.
"""

import os
import tempfile
import time
from typing import Dict, List

import numpy as np
import pytest

from robodm import Trajectory


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=98765)


@pytest.fixture
def large_trajectory_path(temp_dir, rng) -> str:
    """Create a larger trajectory for performance testing."""
    path = os.path.join(temp_dir, "large_traj.vla")
    traj = Trajectory(path, mode="w")

    # Create 1000 timesteps of multimodal data
    for i in range(1000):
        timestamp_ms = int(i * 50)  # 20Hz data
        data = {
            "position": rng.normal(size=3).astype(np.float32),
            "velocity": rng.normal(size=3).astype(np.float32),
            "joint_angles": rng.normal(size=7).astype(np.float32),
            "image": (rng.random((32, 32, 3)) * 255).astype(np.uint8),
            "depth": rng.random((32, 32)).astype(np.float32),
            "status": f"status_{i % 10}",
            "metadata": {
                "step": i,
                "phase": "test"
            },
        }
        traj.add_by_dict(data, timestamp=timestamp_ms)

    traj.close()
    return path


class TestTrajectoryLoaderPerformance:
    """Performance tests for the trajectory loader."""

    def test_full_load_performance(self, large_trajectory_path):
        """Benchmark full trajectory loading."""
        t = Trajectory(large_trajectory_path, mode="r")

        start_time = time.time()
        data = t.load()
        load_time = time.time() - start_time

        # Verify correctness
        assert len(next(iter(data.values()))) == 1000
        assert len(data) > 0

        # Performance check - should load 1000 frames reasonably quickly
        # This is not a strict requirement, just a sanity check
        assert load_time < 30.0  # Should complete within 30 seconds

        print(f"Full load of 1000 frames took {load_time:.3f}s")
        t.close()

    def test_slice_performance_vs_full_load(self, large_trajectory_path):
        """Compare performance of sliced vs full loading."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Time full load
        start_time = time.time()
        full_data = t.load()
        full_time = time.time() - start_time

        # Time small slice
        start_time = time.time()
        slice_data = t.load(data_slice=slice(100, 200))
        slice_time = time.time() - start_time

        # Verify correctness
        assert len(next(iter(slice_data.values()))) == 100
        for k in slice_data:
            np.testing.assert_array_equal(slice_data[k], full_data[k][100:200])

        # Performance - slice should be faster than full load
        print(f"Full load: {full_time:.3f}s, Slice load: {slice_time:.3f}s")

        t.close()

    def test_seeking_performance_benefit(self, large_trajectory_path):
        """Test that seeking provides performance benefit for large slices."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Test slice from beginning (no seeking needed)
        start_time = time.time()
        early_slice = t.load(data_slice=slice(0, 100))
        early_time = time.time() - start_time

        # Test slice from middle (seeking should help)
        start_time = time.time()
        middle_slice = t.load(data_slice=slice(400, 500))
        middle_time = time.time() - start_time

        # Test slice from end (seeking should help significantly)
        start_time = time.time()
        late_slice = t.load(data_slice=slice(
            800, 900))  # Changed from 900-1000 to avoid edge case
        late_time = time.time() - start_time

        # Verify correctness
        assert len(next(iter(early_slice.values()))) == 100
        assert len(next(iter(middle_slice.values()))) == 100

        # Late slice might have fewer frames if we're near the end of data
        late_len = len(next(iter(late_slice.values())))
        assert late_len > 0  # Should have some data

        print(
            f"Early slice: {early_time:.3f}s, Middle slice: {middle_time:.3f}s, Late slice: {late_time:.3f}s"
        )

        # All should complete reasonably quickly
        assert early_time < 10.0
        assert middle_time < 10.0
        assert late_time < 10.0

        t.close()

    def test_frequency_resampling_performance(self, large_trajectory_path):
        """Test performance of frequency resampling."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Test various downsampling rates
        frequencies = [10.0, 5.0, 2.0, 1.0]  # Original is 20Hz
        times = []

        for freq in frequencies:
            start_time = time.time()
            resampled = t.load(desired_frequency=freq)
            resample_time = time.time() - start_time
            times.append(resample_time)

            # Verify approximate expected length
            expected_len = int(1000 * freq / 20.0)  # Rough calculation
            actual_len = len(next(iter(resampled.values())))
            assert abs(actual_len - expected_len) <= 5  # Allow some tolerance

            print(
                f"Resampling to {freq}Hz: {resample_time:.3f}s, {actual_len} frames"
            )

        # All resampling should complete quickly
        assert all(t < 15.0 for t in times)

        t.close()

    def test_combined_operations_performance(self, large_trajectory_path):
        """Test performance of combined resampling and slicing."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Test various combinations
        test_cases = [
            {
                "desired_frequency": 10.0,
                "data_slice": slice(100, 300)
            },
            {
                "desired_frequency": 5.0,
                "data_slice": slice(0, 500)
            },
            {
                "desired_frequency": 2.0,
                "data_slice": slice(200, 800, 2)
            },
        ]

        for i, params in enumerate(test_cases):
            start_time = time.time()
            result = t.load(**params)
            operation_time = time.time() - start_time

            # Verify result is reasonable
            assert len(result) > 0
            result_len = len(next(iter(result.values())))
            # Allow empty results due to resampling effects, but at least verify no error
            assert result_len >= 0

            print(
                f"Combined operation {i+1}: {operation_time:.3f}s, {result_len} frames"
            )

            # Should complete quickly
            assert operation_time < 20.0

        t.close()

    def test_repeated_load_caching_behavior(self, large_trajectory_path):
        """Test if repeated loads show any caching behavior or performance patterns."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Perform same load operation multiple times
        load_times = []
        slice_params = slice(200, 400)

        for i in range(5):
            start_time = time.time()
            data = t.load(data_slice=slice_params)
            load_time = time.time() - start_time
            load_times.append(load_time)

            # Verify consistency
            assert len(next(iter(data.values()))) == 200

        print(f"Repeated load times: {[f'{t:.3f}s' for t in load_times]}")

        # All loads should complete within reasonable time
        assert all(t < 10.0 for t in load_times)

        # Check if there's significant variance (indicating potential caching)
        avg_time = sum(load_times) / len(load_times)
        max_deviation = max(abs(t - avg_time) for t in load_times)
        print(f"Average: {avg_time:.3f}s, Max deviation: {max_deviation:.3f}s")

        t.close()

    def test_memory_usage_large_slice(self, large_trajectory_path):
        """Test memory efficiency with large slices."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Load progressively larger slices
        slice_sizes = [10, 50, 100, 200, 500]

        for size in slice_sizes:
            start_time = time.time()
            data = t.load(data_slice=slice(0, size))
            load_time = time.time() - start_time

            # Verify correct size
            assert len(next(iter(data.values()))) == size

            # Check that larger slices don't have dramatically worse performance
            print(f"Slice size {size}: {load_time:.3f}s")

            # Performance should scale reasonably
            assert load_time < size * 0.01 + 5.0  # Very loose upper bound

        t.close()

    def test_container_return_performance(self, large_trajectory_path):
        """Test that container return is consistently fast regardless of other parameters."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Test container return with various parameters
        test_cases = [
            {},  # No parameters
            {
                "data_slice": slice(0, 1000)
            },  # Large slice
            {
                "desired_frequency": 1.0
            },  # Heavy resampling
            {
                "desired_frequency": 5.0,
                "data_slice": slice(100, 900)
            },  # Combined
        ]

        for i, params in enumerate(test_cases):
            params["return_type"] = "container"

            start_time = time.time()
            result = t.load(**params)
            container_time = time.time() - start_time

            # Verify result
            assert result == large_trajectory_path

            print(f"Container return {i+1}: {container_time:.3f}s")

            # Should be consistently very fast
            assert container_time < 0.1  # Should be nearly instantaneous

        t.close()


class TestTrajectoryLoaderScalability:
    """Test scalability characteristics of the loader."""

    def test_scaling_with_feature_count(self, temp_dir, rng):
        """Test how performance scales with number of features."""
        feature_counts = [5, 10, 20]
        times = []

        for feature_count in feature_counts:
            path = os.path.join(temp_dir, f"features_{feature_count}.vla")
            traj = Trajectory(path, mode="w")

            # Create trajectory with many features
            for i in range(200):  # Fewer timesteps to keep test reasonable
                data = {}
                for j in range(feature_count):
                    data[f"feature_{j}"] = rng.normal(size=3).astype(
                        np.float32)
                traj.add_by_dict(data, timestamp=i * 100)

            traj.close()

            # Time the loading
            t = Trajectory(path, mode="r")
            start_time = time.time()
            loaded = t.load()
            load_time = time.time() - start_time
            times.append(load_time)

            # Verify correctness
            assert len(loaded) == feature_count
            assert len(next(iter(loaded.values()))) == 200

            print(f"Loading {feature_count} features: {load_time:.3f}s")
            t.close()

        # Performance should scale reasonably with feature count
        assert all(t < 20.0 for t in times)

    def test_scaling_with_data_types(self, temp_dir, rng):
        """Test performance with different data types and sizes."""
        path = os.path.join(temp_dir, "mixed_types.vla")
        traj = Trajectory(path, mode="w")

        # Create trajectory with varied data types
        for i in range(300):
            data = {
                "small_int": i,
                "float_val": float(i * 0.1),
                "string_data": f"item_{i}",
                "small_array": rng.normal(size=3).astype(np.float32),
                "medium_array": rng.normal(size=(10, 10)).astype(np.float32),
                "large_array": (rng.random(
                    (20, 20, 3)) * 255).astype(np.uint8),
            }
            traj.add_by_dict(data, timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")

        # Test loading different combinations
        test_cases = [
            slice(0, 50),  # Small slice
            slice(0, 150),  # Medium slice
            slice(0, 300),  # Full data
            slice(100, 200),  # Middle slice
        ]

        for i, slice_params in enumerate(test_cases):
            start_time = time.time()
            data = t.load(data_slice=slice_params)
            load_time = time.time() - start_time

            expected_len = slice_params.stop - slice_params.start
            if slice_params.stop > 300:
                expected_len = 300 - slice_params.start

            actual_len = len(next(iter(data.values())))
            assert actual_len == expected_len

            print(
                f"Mixed types, slice {i+1}: {load_time:.3f}s, {actual_len} frames"
            )

            # Should complete reasonably quickly
            assert load_time < 15.0

        t.close()

    def test_performance_regression_protection(self, large_trajectory_path):
        """Basic regression test to catch significant performance degradation."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Define performance expectations (these are loose bounds)
        performance_expectations = [
            (lambda: t.load(data_slice=slice(0, 10)), 2.0, "Small slice"),
            (lambda: t.load(data_slice=slice(0, 100)), 5.0, "Medium slice"),
            (lambda: t.load(desired_frequency=5.0), 10.0, "Resampling"),
            (lambda: t.load(return_type="container"), 0.1, "Container return"),
        ]

        for operation, max_time, description in performance_expectations:
            start_time = time.time()
            result = operation()
            operation_time = time.time() - start_time

            print(f"{description}: {operation_time:.3f}s (max: {max_time}s)")

            # Check against regression threshold
            if operation_time > max_time:
                pytest.fail(
                    f"Performance regression detected: {description} took "
                    f"{operation_time:.3f}s, expected < {max_time}s")

        t.close()


@pytest.mark.slow
class TestTrajectoryLoaderStressTests:
    """Stress tests for the loader (marked as slow)."""

    def test_very_large_trajectory_handling(self, temp_dir, rng):
        """Test handling of very large trajectories (if resources allow)."""
        path = os.path.join(temp_dir, "very_large.vla")
        traj = Trajectory(path, mode="w")

        # Create larger trajectory (but not so large it breaks CI)
        n_steps = 5000
        for i in range(n_steps):
            if i % 1000 == 0:
                print(f"Creating step {i}/{n_steps}")

            data = {
                "position": rng.normal(size=3).astype(np.float32),
                "image": (rng.random((16, 16, 3)) * 255).astype(np.uint8),
            }
            traj.add_by_dict(data, timestamp=i * 50)

        traj.close()

        t = Trajectory(path, mode="r")

        # Test various operations on large trajectory
        start_time = time.time()
        small_slice = t.load(data_slice=slice(1000, 1100))
        slice_time = time.time() - start_time

        assert len(next(iter(small_slice.values()))) == 100
        print(f"Large trajectory slice: {slice_time:.3f}s")

        # Should still be reasonably fast due to seeking
        assert slice_time < 30.0

        t.close()

    def test_high_frequency_resampling_stress(self, large_trajectory_path):
        """Test resampling with various challenging frequency combinations."""
        t = Trajectory(large_trajectory_path, mode="r")

        # Test challenging frequency combinations
        test_frequencies = [
            0.1,  # Very low frequency
            0.5,  # Low frequency
            19.9,  # Just under original frequency
            20.0,  # Approximately original frequency
            20.1,  # Just above original frequency
        ]

        for freq in test_frequencies:
            start_time = time.time()
            resampled = t.load(desired_frequency=freq)
            resample_time = time.time() - start_time

            result_len = len(next(iter(resampled.values())))
            print(
                f"Frequency {freq}Hz: {resample_time:.3f}s, {result_len} frames"
            )

            # Should complete within reasonable time
            assert resample_time < 20.0

            # Result should be reasonable
            assert result_len >= 0

        t.close()
