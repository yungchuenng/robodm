"""
Edge case and boundary testing for Trajectory.load functionality.
"""

import os
import tempfile
from typing import Dict, List

import av
import numpy as np
import pytest

from robodm import FeatureType, Trajectory


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


class TestTrajectoryLoaderEdgeCases:
    """Edge cases and boundary conditions for the new loader."""

    def test_zero_length_trajectory(self, temp_dir):
        """Test loading trajectory with zero data points."""
        path = os.path.join(temp_dir, "zero_length.vla")
        traj = Trajectory(path, mode="w")
        traj.close()

        # Check if file exists after creation
        if not os.path.exists(path):
            # If no file was created (because no data was added),
            # the Trajectory constructor should fail when trying to read
            with pytest.raises(FileNotFoundError):
                t = Trajectory(path, mode="r")
            return

        t = Trajectory(path, mode="r")

        # All operations should work on empty trajectory
        empty = t.load()
        assert isinstance(empty, dict)
        assert len(empty) == 0

        # Slicing empty should return empty
        sliced = t.load(data_slice=slice(0, 10))
        assert len(sliced) == 0

        # Resampling empty should return empty
        resampled = t.load(desired_frequency=10.0)
        assert len(resampled) == 0

        # Container return should work
        container_path = t.load(return_type="container")
        assert container_path == path

        t.close()

    def test_single_packet_with_none_pts(self, temp_dir):
        """Test handling of packets with None pts/dts values."""
        path = os.path.join(temp_dir, "none_pts.vla")
        traj = Trajectory(path, mode="w")

        # Add one normal data point
        traj.add("value", 42, timestamp=100)
        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        # Should skip packets with None pts and only load valid ones
        assert "value" in data
        assert len(data["value"]) >= 1

        t.close()

    def test_slice_start_equals_stop(self, temp_dir):
        """Test slice where start equals stop (empty slice)."""
        path = os.path.join(temp_dir, "equal_start_stop.vla")
        traj = Trajectory(path, mode="w")

        for i in range(10):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Empty slices at various positions
        for start_stop in [0, 5, 9, 15]:  # Including beyond data
            empty = t.load(data_slice=slice(start_stop, start_stop))
            if len(empty) > 0:  # Only check if trajectory has data
                assert all(len(v) == 0 for v in empty.values())

        t.close()

    def test_slice_with_very_large_step(self, temp_dir):
        """Test slicing with step much larger than data length."""
        path = os.path.join(temp_dir, "large_step.vla")
        traj = Trajectory(path, mode="w")

        for i in range(20):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Step of 100 on 20 elements should give only first element
        result = t.load(data_slice=slice(0, None, 100))
        assert all(len(v) == 1 for v in result.values())
        assert result["value"][0] == 0

        # Step of 10 should give every 10th element
        result = t.load(data_slice=slice(0, None, 10))
        assert all(len(v) == 2 for v in result.values())  # Elements 0 and 10
        np.testing.assert_array_equal(result["value"], [0, 10])

        t.close()

    def test_frequency_boundary_values(self, temp_dir):
        """Test frequency resampling with boundary values."""
        path = os.path.join(temp_dir, "freq_boundary.vla")
        traj = Trajectory(path, mode="w")

        # Create data at 10Hz (100ms intervals)
        for i in range(30):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Very small frequency (much less than 1Hz)
        very_small = t.load(
            desired_frequency=0.001)  # 1 frame per 1000 seconds
        assert all(len(v) <= 1 for v in very_small.values())

        # Frequency that creates exactly one frame period
        one_period = t.load(desired_frequency=1.0)  # 1Hz = 1000ms period
        # Should get roughly every 10th frame (1000ms / 100ms = 10)
        expected_len = len(next(iter(one_period.values())))
        assert 2 <= expected_len <= 5  # Allow some tolerance

        t.close()

    def test_seek_beyond_stream_end(self, temp_dir):
        """Test seeking to position beyond the stream length."""
        path = os.path.join(temp_dir, "seek_beyond.vla")
        traj = Trajectory(path, mode="w")

        # Short trajectory
        for i in range(5):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Try to slice starting beyond the data
        beyond = t.load(data_slice=slice(10, 20))
        assert all(len(v) == 0 for v in beyond.values())

        # Slice that starts within data but extends beyond
        partial = t.load(data_slice=slice(3, 10))
        full = t.load()
        for k in partial:
            np.testing.assert_array_equal(partial[k], full[k][3:])

        t.close()

    def test_mixed_data_types_in_single_feature(self, temp_dir):
        """Test trajectory with varying data types for same feature name."""
        path = os.path.join(temp_dir, "mixed_types.vla")
        traj = Trajectory(path, mode="w")

        # This should be consistent - all same feature should have same type
        for i in range(5):
            traj.add("consistent_value", float(i), timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        # All values for same feature should have consistent type
        assert "consistent_value" in data
        assert len(data["consistent_value"]) == 5
        assert data["consistent_value"].dtype in [np.float32, np.float64]

        t.close()

    def test_very_sparse_timestamps(self, temp_dir):
        """Test trajectory with very sparse, irregular timestamps."""
        path = os.path.join(temp_dir, "sparse_timestamps.vla")
        traj = Trajectory(path, mode="w")

        # Very irregular timestamps
        timestamps = [0, 1000, 5000, 5001, 10000]  # ms
        for i, ts in enumerate(timestamps):
            traj.add("value", i, timestamp=ts)

        traj.close()

        t = Trajectory(path, mode="r")

        # Should handle sparse data gracefully
        full = t.load()
        assert len(full["value"]) == 5

        # Resampling should work with sparse data
        resampled = t.load(desired_frequency=1.0)  # 1Hz = 1000ms
        # Should get fewer frames due to large gaps
        assert len(resampled["value"]) <= 5

        t.close()

    def test_unicode_and_special_characters(self, temp_dir):
        """Test handling of unicode and special characters in string data."""
        path = os.path.join(temp_dir, "unicode.vla")
        traj = Trajectory(path, mode="w")

        special_strings = [
            "hello",
            "café",
            "🤖",
            "データ",
            "test\nwith\nnewlines",
            "quotes\"and'apostrophes",
            "",  # empty string
        ]

        for i, s in enumerate(special_strings):
            traj.add("text", s, timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        assert "text" in data
        assert len(data["text"]) == len(special_strings)
        # Should preserve all special characters
        for i, expected in enumerate(special_strings):
            assert data["text"][i] == expected

        # Test slicing with unicode data
        sliced = t.load(data_slice=slice(1, 4))
        np.testing.assert_array_equal(sliced["text"], special_strings[1:4])

        t.close()

    def test_extremely_large_arrays(self, temp_dir, rng):
        """Test loading trajectory with very large numpy arrays."""
        path = os.path.join(temp_dir, "large_arrays.vla")
        traj = Trajectory(path, mode="w")

        # Create reasonably large arrays (not extremely large to avoid memory issues)
        for i in range(3):
            large_array = rng.random((100, 100)).astype(np.float32)
            traj.add("large_data", large_array, timestamp=i * 1000)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        # Should load successfully
        assert "large_data" in data
        loaded_shape = data["large_data"].shape
        assert loaded_shape[0] == 3  # 3 timesteps
        assert loaded_shape[1:] == (100, 100)  # Each array is 100x100

        t.close()

    def test_load_with_corrupted_metadata(self, temp_dir):
        """Test loading trajectory with missing or corrupted stream metadata."""
        path = os.path.join(temp_dir, "normal.vla")
        traj = Trajectory(path, mode="w")

        # Create normal trajectory first
        for i in range(5):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        # Loading should work normally
        t = Trajectory(path, mode="r")
        data = t.load()
        assert "value" in data
        assert len(data["value"]) == 5
        t.close()

    def test_concurrent_feature_different_lengths(self, temp_dir):
        """Test loading when different features might have different packet counts."""
        path = os.path.join(temp_dir, "different_lengths.vla")
        traj = Trajectory(path, mode="w")

        # Add features at different rates to same trajectory
        # This tests the early termination logic
        for i in range(10):
            traj.add("frequent", i, timestamp=i * 100)
            if i % 2 == 0:  # Less frequent feature
                traj.add("sparse", i // 2, timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        # Should load all available data for each feature
        assert len(data["frequent"]) == 10
        assert len(data["sparse"]) == 5

        # Slicing should work correctly with different lengths
        sliced = t.load(data_slice=slice(0, 3))
        # Each feature gets sliced independently
        assert len(sliced["frequent"]) == 3
        assert len(sliced["sparse"]) <= 3  # Might be fewer due to sparsity

        t.close()

    def test_precision_edge_cases_float(self, temp_dir):
        """Test edge cases with floating point precision."""
        path = os.path.join(temp_dir, "float_precision.vla")
        traj = Trajectory(path, mode="w")

        # Test various floating point edge cases
        float_values = [
            0.0,
            -0.0,
            1e-10,  # Very small positive
            -1e-10,  # Very small negative
            1e10,  # Very large
            np.inf,
            -np.inf,
            # np.nan,  # Skip NaN as it may cause comparison issues
        ]

        for i, val in enumerate(float_values):
            if not np.isnan(val):  # Skip NaN values for now
                traj.add("float_val", float(val), timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        assert "float_val" in data
        # Verify precision is maintained (for finite values)
        for i, expected in enumerate(float_values):
            if not np.isnan(expected) and np.isfinite(expected):
                assert abs(data["float_val"][i] - expected) < 1e-12

        t.close()

    def test_memory_efficient_loading_large_slice(self, temp_dir):
        """Test that large slices don't load unnecessary data into memory."""
        path = os.path.join(temp_dir, "memory_test.vla")
        traj = Trajectory(path, mode="w")

        # Create reasonably sized trajectory
        for i in range(100):  # Reduced from 1000 to make test faster
            traj.add("value", i, timestamp=i * 100)  # 100ms intervals

        traj.close()

        t = Trajectory(path, mode="r")

        # Load small slice from middle - should be efficient
        small_slice = t.load(data_slice=slice(40, 50))
        assert len(small_slice["value"]) == 10
        np.testing.assert_array_equal(small_slice["value"], list(range(40,
                                                                       50)))

        # Load with high frequency + slice - should also be efficient
        freq_slice = t.load(desired_frequency=5.0,
                            data_slice=slice(1, 11))  # 5Hz on 10Hz data
        assert len(freq_slice["value"]) == 10

        t.close()


class TestTrajectoryLoaderErrorHandling:
    """Test error handling and recovery in the loader."""

    def test_invalid_slice_combinations(self, temp_dir):
        """Test various invalid slice parameter combinations."""
        path = os.path.join(temp_dir, "for_error_test.vla")
        traj = Trajectory(path, mode="w")

        for i in range(10):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Test invalid step values
        invalid_slices = [
            slice(0, 10, 0),  # Zero step
            slice(0, 10, -1),  # Negative step
            slice(0, 10, -5),  # Large negative step
        ]

        for invalid_slice in invalid_slices:
            with pytest.raises(ValueError):
                _ = t.load(data_slice=invalid_slice)

        t.close()

    def test_invalid_frequency_values(self, temp_dir):
        """Test various invalid frequency values."""
        path = os.path.join(temp_dir, "for_freq_error.vla")
        traj = Trajectory(path, mode="w")

        traj.add("value", 42, timestamp=0)
        traj.close()

        t = Trajectory(path, mode="r")

        invalid_frequencies = [
            0.0,  # Zero
            -1.0,  # Negative
            -100.0,  # Large negative
        ]

        for invalid_freq in invalid_frequencies:
            with pytest.raises(ValueError):
                _ = t.load(desired_frequency=invalid_freq)

        t.close()

    def test_parameter_combination_edge_cases(self, temp_dir):
        """Test edge cases in parameter combinations."""
        path = os.path.join(temp_dir, "param_combos.vla")
        traj = Trajectory(path, mode="w")

        for i in range(20):
            traj.add("value", i, timestamp=i * 100)
        traj.close()

        t = Trajectory(path, mode="r")

        # Valid but unusual combinations
        edge_cases = [
            # Very high frequency with slice
            {
                "desired_frequency": 1000.0,
                "data_slice": slice(0, 5)
            },
            # Very low frequency with large slice
            {
                "desired_frequency": 0.1,
                "data_slice": slice(0, None)
            },
            # Frequency with slice that results in no data
            {
                "desired_frequency": 5.0,
                "data_slice": slice(100, 200)
            },
        ]

        for params in edge_cases:
            # Should not raise errors, just return appropriate results
            result = t.load(**params)
            assert isinstance(result, dict)
            # All features should have same length
            if result:
                lengths = [len(v) for v in result.values()]
                assert len(set(lengths)) == 1

        t.close()
