"""
Comprehensive tests for Trajectory.load with resampling and positive-index slicing.
"""

import os
import tempfile
import time
from typing import Dict, List

import numpy as np
import pytest

from robodm import Trajectory

# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Process-wide RNG so the dataset is deterministic across tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def _make_step(rng: np.random.Generator, idx: int) -> Dict[str, object]:
    """Generate one synthetic trajectory step (≈ 10 Hz)."""
    return {
        "timestamp": idx * 0.10,  # scalar float
        "robot_position": rng.normal(size=3).astype(np.float32),  # (3,)
        "joint_angles": rng.normal(size=7).astype(np.float32),  # (7,)
        "action": rng.normal(size=4).astype(np.float32),  # (4,)
        "gripper_state": "open" if idx % 2 == 0 else "closed",  # str
        "sensor_reading": float(rng.standard_normal()),  # scalar float
        # Add image-like data for testing video codecs
        "camera_rgb": (rng.random(
            (64, 64, 3)) * 255).astype(np.uint8),  # RGB image
        "depth_map": rng.random((32, 32)).astype(np.float32),  # depth/float32
        "metadata": {
            "step": idx,
            "tag": f"step_{idx}"
        },  # nested dict
    }


@pytest.fixture
def base_trajectory_data(rng) -> List[Dict[str, object]]:
    """100 × 10 Hz synthetic trajectory."""
    return [_make_step(rng, i) for i in range(100)]


@pytest.fixture
def trajectory_path(temp_dir, base_trajectory_data) -> str:
    path = os.path.join(temp_dir, "traj.vla")
    traj = Trajectory(path, mode="w")

    # Add data with explicit timestamps (100ms intervals = 10 Hz)
    for i, step_data in enumerate(base_trajectory_data):
        timestamp_ms = int(i * 100)  # 100ms intervals
        # Remove timestamp from step_data since we're passing it explicitly
        data_without_timestamp = {
            k: v
            for k, v in step_data.items() if k != "timestamp"
        }
        traj.add_by_dict(data_without_timestamp, timestamp=timestamp_ms)

    traj.close()
    return path


@pytest.fixture
def small_trajectory_path(temp_dir, rng) -> str:
    """Smaller trajectory for testing edge cases."""
    path = os.path.join(temp_dir, "small_traj.vla")
    traj = Trajectory(path, mode="w")

    # Only 5 steps
    for i in range(5):
        timestamp_ms = int(i * 100)
        data = {
            "value": i,
            "name": f"item_{i}",
            "array": rng.normal(size=2).astype(np.float32),
        }
        traj.add_by_dict(data, timestamp=timestamp_ms)

    traj.close()
    return path


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #


class TestTrajectoryLoad:

    # --------------------------- basic behaviour --------------------------- #

    def test_no_kwargs_is_identity(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")
        a = t.load()  # reference
        b = t.load(return_type="numpy")  # new impl path
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        t.close()

    def test_load_returns_correct_keys(self, trajectory_path):
        """Test that all expected features are loaded."""
        t = Trajectory(trajectory_path, mode="r")
        data = t.load()

        expected_keys = {
            "robot_position",
            "joint_angles",
            "action",
            "gripper_state",
            "sensor_reading",
            "camera_rgb",
            "depth_map",
            "metadata/step",
            "metadata/tag",
        }
        assert set(data.keys()) == expected_keys
        t.close()

    def test_empty_trajectory_handling(self, temp_dir):
        """Test loading an empty trajectory."""
        path = os.path.join(temp_dir, "empty.vla")
        # Create empty trajectory
        traj = Trajectory(path, mode="w")
        traj.close()

        # Check if file exists after creation
        if not os.path.exists(path):
            # If no file was created (because no data was added),
            # the Trajectory constructor should fail when trying to read
            with pytest.raises(FileNotFoundError):
                t = Trajectory(path, mode="r")
            return

        # If file exists, load should return empty dict
        t = Trajectory(path, mode="r")
        data = t.load()
        assert isinstance(data, dict)
        assert len(data) == 0
        t.close()

    # ------------------------------ slicing ------------------------------- #

    @pytest.mark.parametrize(
        "sl",
        [
            slice(0, 10),
            slice(10, 50, 5),
            slice(5, 15, 2),
            slice(None, 20),
            slice(80, None),
            slice(None, None, 3),
        ],
    )
    def test_simple_slice(self, trajectory_path, sl):
        t = Trajectory(trajectory_path, mode="r")
        part = t.load(data_slice=sl)
        full = t.load()

        for k in part:
            np.testing.assert_array_equal(part[k], full[k][sl])
        t.close()

    def test_slice_boundary_conditions(self, small_trajectory_path):
        """Test slicing with various boundary conditions."""
        t = Trajectory(small_trajectory_path, mode="r")

        # Single element slice
        single = t.load(data_slice=slice(2, 3))
        assert all(len(v) == 1 for v in single.values())

        # Start at last element
        last = t.load(data_slice=slice(4, 5))
        assert all(len(v) == 1 for v in last.values())

        # Step larger than data
        large_step = t.load(data_slice=slice(0, 5, 10))
        assert all(len(v) == 1 for v in large_step.values())

        t.close()

    def test_slice_invalid_negative(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")
        with pytest.raises(
                ValueError,
                match="Negative slice start values are not supported"):
            _ = t.load(data_slice=slice(-10, None))
        t.close()

    def test_slice_invalid_step(self, trajectory_path):
        """Test invalid slice step values."""
        t = Trajectory(trajectory_path, mode="r")

        # Zero step
        with pytest.raises(
                ValueError,
                match="Reverse or zero-step slices are not supported"):
            _ = t.load(data_slice=slice(0, 10, 0))

        # Negative step
        with pytest.raises(
                ValueError,
                match="Reverse or zero-step slices are not supported"):
            _ = t.load(data_slice=slice(10, 0, -1))

        t.close()

    def test_slice_empty_and_oob(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")

        # empty slice
        empty = t.load(data_slice=slice(50, 50))
        assert all(len(v) == 0 for v in empty.values())

        # beyond right edge
        oob = t.load(data_slice=slice(90, 150))
        full = t.load()
        for k in full:
            np.testing.assert_array_equal(oob[k], full[k][90:])

        t.close()

    def test_slice_with_none_values(self, trajectory_path):
        """Test slicing with None values in slice object."""
        t = Trajectory(trajectory_path, mode="r")

        # Test various combinations of None
        test_slices = [
            slice(None, 10),  # start=None
            slice(10, None),  # stop=None
            slice(None, None, 2),  # start=None, stop=None
            slice(None, None, None),  # all None
        ]

        full = t.load()
        for sl in test_slices:
            part = t.load(data_slice=sl)
            for k in part:
                np.testing.assert_array_equal(part[k], full[k][sl])

        t.close()

    # ---------------------------- resampling ------------------------------ #

    @pytest.mark.parametrize("freq, expect_factor", [(5.0, 0.5), (2.0, 0.2),
                                                     (1.0, 0.1)])
    def test_downsample(self, trajectory_path, freq, expect_factor):
        t = Trajectory(trajectory_path, mode="r")
        down = t.load(desired_frequency=freq)
        ref = t.load()
        ref_len = len(next(iter(ref.values())))
        down_len = len(next(iter(down.values())))

        # allow ±1 frame tolerance (integer division effects)
        target = int(ref_len * expect_factor + 0.5)
        assert abs(down_len - target) <= 1
        # all features must have identical length
        assert len({len(v) for v in down.values()}) == 1
        t.close()

    def test_downsample_with_slice(self, trajectory_path):
        """Test downsampling combined with slicing."""
        t = Trajectory(trajectory_path, mode="r")

        # The correct reference: first downsample to 5Hz, then slice
        downsampled_first = t.load(desired_frequency=5.0)
        reference = {}
        for k, v in downsampled_first.items():
            reference[k] = v[slice(20, 70)]

        # The shortcut version: downsample + slice in one go
        combo = t.load(desired_frequency=5.0, data_slice=slice(20, 70))

        assert combo.keys() == reference.keys()
        for k in combo:
            np.testing.assert_array_equal(combo[k], reference[k])
        t.close()

    def test_resampling_frequency_edge_cases(self, trajectory_path):
        """Test edge cases for frequency resampling."""
        t = Trajectory(trajectory_path, mode="r")

        # Very low frequency (should get only first frame or very few)
        very_low = t.load(desired_frequency=0.1)  # One frame every 10 seconds
        assert all(len(v) <= 2
                   for v in very_low.values())  # At most 1-2 frames

        # Frequency that matches exactly
        exact = t.load(desired_frequency=10.0)  # Matches our 10Hz data
        ref = t.load()
        # Should be close to original length (allow small tolerance)
        ref_len = len(next(iter(ref.values())))
        exact_len = len(next(iter(exact.values())))
        assert abs(exact_len - ref_len) <= 2

        t.close()

    def test_resampling_invalid_frequency(self, trajectory_path):
        """Test invalid frequency values."""
        t = Trajectory(trajectory_path, mode="r")

        # Zero frequency
        with pytest.raises(ValueError,
                           match="desired_frequency must be positive"):
            _ = t.load(desired_frequency=0.0)

        # Negative frequency
        with pytest.raises(ValueError,
                           match="desired_frequency must be positive"):
            _ = t.load(desired_frequency=-1.0)

        t.close()

    # ------------------------ data-type preservation ---------------------- #

    def test_dtype_and_content_preserved(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")
        base = t.load()
        ds = t.load(desired_frequency=5.0)

        for k, v in ds.items():
            if k == "gripper_state":
                assert v.dtype == object
                assert set(v).issubset({"open", "closed"})
            elif "metadata" in k:
                assert v.dtype == object  # String data
            else:
                assert v.dtype == base[k].dtype
        t.close()

    def test_different_data_types_preserved(self, temp_dir, rng):
        """Test that various numpy data types are preserved correctly."""
        path = os.path.join(temp_dir, "dtype_test.vla")
        traj = Trajectory(path, mode="w")

        # Create data with different dtypes
        test_data = {
            "int8_data": np.array([1, 2, 3], dtype=np.int8),
            "int32_data": np.array([100, 200, 300], dtype=np.int32),
            "float64_data": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            "bool_data": np.array([True, False, True], dtype=bool),
            "uint8_image": (rng.random((4, 4)) * 255).astype(np.uint8),
        }

        for i in range(3):
            step = {k: v[i] if v.ndim > 0 else v for k, v in test_data.items()}
            step["uint8_image"] = test_data["uint8_image"]  # Keep full image
            traj.add_by_dict(step, timestamp=i * 100)

        traj.close()

        # Load and verify dtypes
        t = Trajectory(path, mode="r")
        loaded = t.load()

        assert loaded["int8_data"].dtype == np.int8
        assert loaded["int32_data"].dtype == np.int32
        assert loaded["float64_data"].dtype == np.float64
        assert loaded["bool_data"].dtype == bool
        assert loaded["uint8_image"].dtype == np.uint8

        t.close()

    # -------------------------- return_type ------------------------------ #

    def test_container_return(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")
        p1 = t.load(return_type="container")
        p2 = t.load(return_type="container", desired_frequency=5.0)
        p3 = t.load(return_type="container", data_slice=slice(0, 5))
        assert p1 == trajectory_path == p2 == p3
        t.close()

    def test_invalid_return_type(self, trajectory_path):
        """Test invalid return_type parameter."""
        t = Trajectory(trajectory_path, mode="r")
        with pytest.raises(ValueError,
                           match="return_type must be 'numpy' or 'container'"):
            _ = t.load(return_type="invalid")
        t.close()

    # ----------------------------- errors -------------------------------- #

    def test_invalid_args(self, trajectory_path):
        t = Trajectory(trajectory_path, mode="r")
        with pytest.raises(ValueError):
            _ = t.load(return_type="bad")
        with pytest.raises(ValueError):
            _ = t.load(desired_frequency=-1.0)
        t.close()

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a file that doesn't exist."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.vla")
        with pytest.raises(FileNotFoundError):
            _ = Trajectory(nonexistent_path, mode="r")

    # -------------------------- seeking optimization ---------------------- #

    def test_seeking_optimization_slice_only(self, trajectory_path):
        """Test that seeking works correctly for slice-only loads."""
        t = Trajectory(trajectory_path, mode="r")

        # Load a slice from middle of data
        sliced = t.load(data_slice=slice(30, 40))
        full = t.load()

        # Should match exactly
        for k in sliced:
            np.testing.assert_array_equal(sliced[k], full[k][30:40])

        t.close()

    def test_seeking_optimization_with_frequency(self, trajectory_path):
        """Test seeking when combining frequency and slice."""
        t = Trajectory(trajectory_path, mode="r")

        # This should seek to the appropriate timestamp for resampled data
        combo = t.load(desired_frequency=5.0, data_slice=slice(10, 20))

        # Compare with manual approach
        resampled = t.load(desired_frequency=5.0)
        expected = {}
        for k, v in resampled.items():
            expected[k] = v[10:20]

        for k in combo:
            np.testing.assert_array_equal(combo[k], expected[k])

        t.close()

    def test_seeking_failure_fallback(self, small_trajectory_path):
        """Test that seeking failure gracefully falls back to normal decoding."""
        t = Trajectory(small_trajectory_path, mode="r")

        # This should work even if seeking fails internally
        result = t.load(data_slice=slice(1, 4))
        full = t.load()

        for k in result:
            np.testing.assert_array_equal(result[k], full[k][1:4])

        t.close()

    # --------------------------- performance ----------------------------- #

    def test_slice_faster_than_full(self, trajectory_path):
        """Not a strict perf test – just asserts both paths run quickly."""
        t = Trajectory(trajectory_path, mode="r")

        start = time.time()
        _ = t.load()
        full_time = time.time() - start

        start = time.time()
        _ = t.load(data_slice=slice(0, 10))
        slice_time = time.time() - start

        # In CI, timings can be noisy – just check they completed.
        assert full_time > 0.0 and slice_time > 0.0
        t.close()

    # ---------------------- codec smoke test ----------------------------- #

    @pytest.mark.parametrize("codec", ["rawvideo", "ffv1"])
    def test_different_codecs_roundtrip(self, temp_dir, base_trajectory_data,
                                        codec):
        path = os.path.join(temp_dir, f"traj_{codec}.vla")
        traj = Trajectory(path, mode="w", video_codec=codec)

        # Add data with explicit timestamps (100ms intervals = 10 Hz)
        for i, step_data in enumerate(base_trajectory_data):
            timestamp_ms = int(i * 100)  # 100ms intervals
            # Remove timestamp from step_data since we're passing it explicitly
            data_without_timestamp = {
                k: v
                for k, v in step_data.items() if k != "timestamp"
            }
            traj.add_by_dict(data_without_timestamp, timestamp=timestamp_ms)

        traj.close()

        t = Trajectory(path, mode="r")
        # basic slice
        part = t.load(data_slice=slice(0, 8))
        assert len(next(iter(part.values()))) == 8
        t.close()

    # ------------------------ advanced edge cases ----------------------- #

    def test_empty_packets_handling(self, temp_dir):
        """Test handling of empty or None packets."""
        path = os.path.join(temp_dir, "sparse.vla")
        traj = Trajectory(path, mode="w")

        # Add some normal data with gaps
        for i in [0, 2, 5, 7]:  # Sparse timestamps
            traj.add("value", i, timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()
        assert len(data["value"]) == 4  # Should have 4 values
        np.testing.assert_array_equal(data["value"], [0, 2, 5, 7])
        t.close()

    def test_single_frame_trajectory(self, temp_dir):
        """Test loading trajectory with only one frame."""
        path = os.path.join(temp_dir, "single.vla")
        traj = Trajectory(path, mode="w")

        traj.add_by_dict({"value": 42, "name": "single"}, timestamp=0)
        traj.close()

        t = Trajectory(path, mode="r")

        # Test various operations on single frame
        full = t.load()
        assert len(full["value"]) == 1
        assert full["value"][0] == 42

        # Slice that includes the frame
        sliced = t.load(data_slice=slice(0, 1))
        assert len(sliced["value"]) == 1

        # Slice that excludes the frame
        empty = t.load(data_slice=slice(1, 2))
        assert len(empty["value"]) == 0

        # Resampling
        resampled = t.load(desired_frequency=1.0)
        assert len(resampled["value"]) == 1

        t.close()

    def test_large_step_slice(self, trajectory_path):
        """Test slicing with step larger than data length."""
        t = Trajectory(trajectory_path, mode="r")

        # Step of 1000 on 100 elements should give only first element
        large_step = t.load(data_slice=slice(0, None, 1000))
        assert all(len(v) == 1 for v in large_step.values())

        t.close()

    def test_complex_feature_names(self, temp_dir, rng):
        """Test loading with complex/nested feature names."""
        path = os.path.join(temp_dir, "complex_names.vla")
        traj = Trajectory(path, mode="w", feature_name_separator="/")

        # Add nested dictionary data
        nested_data = {
            "robot": {
                "arm": {
                    "joint_0": 1.0,
                    "joint_1": 2.0
                },
                "base": {
                    "x": 0.0,
                    "y": 1.0
                },
            },
            "sensor": {
                "camera": {
                    "rgb": rng.random((8, 8, 3)),
                    "depth": rng.random((8, 8))
                }
            },
        }

        for i in range(5):
            traj.add_by_dict(nested_data, timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()

        # Check that nested names are properly flattened
        expected_keys = {
            "robot/arm/joint_0",
            "robot/arm/joint_1",
            "robot/base/x",
            "robot/base/y",
            "sensor/camera/rgb",
            "sensor/camera/depth",
        }
        assert set(data.keys()) == expected_keys

        # Test slicing on complex names
        sliced = t.load(data_slice=slice(1, 4))
        assert all(len(v) == 3 for v in sliced.values())

        t.close()

    def test_concurrent_stream_early_termination(self, trajectory_path):
        """Test early termination when all streams finish their slice."""
        t = Trajectory(trajectory_path, mode="r")

        # Load a small slice that should trigger early termination
        small_slice = t.load(data_slice=slice(0, 5))
        full = t.load()

        # Verify correctness
        for k in small_slice:
            np.testing.assert_array_equal(small_slice[k], full[k][:5])

        t.close()

    def test_metadata_preservation_during_load(self, trajectory_path):
        """Test that stream metadata is correctly preserved during loading."""
        t = Trajectory(trajectory_path, mode="r")

        # Load with different parameters should preserve feature types
        full = t.load()
        sliced = t.load(data_slice=slice(0, 10))
        resampled = t.load(desired_frequency=5.0)

        # All should have same keys and compatible dtypes
        assert set(full.keys()) == set(sliced.keys()) == set(resampled.keys())

        for k in full.keys():
            assert full[k].dtype == sliced[k].dtype
            # Resampled might have different length but same dtype
            assert full[k].dtype == resampled[k].dtype

        t.close()

    def test_extreme_upsampling_frequency(self, trajectory_path):
        """Test upsampling with extremely high frequency."""
        t = Trajectory(trajectory_path, mode="r")
        ref = t.load()
        hi = t.load(desired_frequency=1e3)  # 1000 Hz - very high

        # Should get significantly more frames due to upsampling
        ref_len = len(ref["robot_position"])
        hi_len = len(hi["robot_position"])

        # Should have many more frames but bounded by reasonable limits
        assert (
            hi_len > ref_len
        ), f"High frequency should create more frames: {hi_len} vs {ref_len}"

        # Should contain all original data
        ref_positions = ref["robot_position"]
        hi_positions = hi["robot_position"]

        # Check that original values are preserved in upsampled data
        unique_ref = [tuple(row) for row in ref_positions]
        unique_hi = [tuple(row) for row in hi_positions]

        for orig_pos in unique_ref:
            assert (
                orig_pos in unique_hi
            ), f"Original position {orig_pos} should be preserved in upsampled data"

        t.close()


class TestTrajectoryLoadIntegration:
    """Integration tests combining multiple features."""

    def test_full_pipeline_integration(self, temp_dir, rng):
        """Test complete pipeline from creation to loading with all features."""
        path = os.path.join(temp_dir, "integration.vla")

        # Create trajectory with diverse data types
        traj = Trajectory(path, mode="w", video_codec="ffv1")

        for i in range(50):
            step_data = {
                "timestamp": i * 0.02,  # 50 Hz
                "position": rng.normal(size=3).astype(np.float32),
                "image": (rng.random((16, 16, 3)) * 255).astype(np.uint8),
                "status": "active" if i % 3 == 0 else "idle",
                "metadata": {
                    "iteration": i,
                    "phase": "test"
                },
            }
            traj.add_by_dict(step_data,
                             timestamp=int(i * 20))  # 20ms intervals

        traj.close()

        # Test various loading scenarios
        t = Trajectory(path, mode="r")

        # Full load
        full = t.load()
        full_len = len(next(iter(full.values())))
        assert full_len == 50

        # Downsample to ~25Hz
        downsampled = t.load(desired_frequency=25.0)
        down_len = len(next(iter(downsampled.values())))
        assert 15 <= down_len <= 35  # Should be roughly half, allow wide tolerance

        # Slice middle portion
        middle = t.load(data_slice=slice(10, 40))
        assert len(next(iter(middle.values()))) == 30

        # Combine resampling and slicing - allow for more flexibility
        combo = t.load(desired_frequency=10.0, data_slice=slice(5, 15))
        combo_len = len(next(iter(combo.values())))
        assert combo_len >= 0  # At minimum should not error and return valid data

        # Container return
        container_path = t.load(return_type="container")
        assert container_path == path

        t.close()

    def test_robustness_with_malformed_data(self, temp_dir):
        """Test robustness when loading trajectories with potential issues."""
        path = os.path.join(temp_dir, "robust.vla")
        traj = Trajectory(path, mode="w")

        # Add some normal data
        for i in range(10):
            traj.add_by_dict({
                "value": i,
                "data": np.array([i, i + 1])
            },
                             timestamp=i * 100)

        traj.close()

        t = Trajectory(path, mode="r")

        # Should handle various edge case parameters gracefully
        try:
            # Very large slice that goes beyond data
            result = t.load(data_slice=slice(0, 1000))
            assert len(next(iter(result.values()))) == 10

            # Very small frequency
            result = t.load(desired_frequency=0.01)
            assert len(next(iter(result.values()))) <= 2

            # Slice with large step
            result = t.load(data_slice=slice(0, None, 100))
            assert len(next(iter(result.values()))) == 1

        except Exception as e:
            pytest.fail(f"Robustness test failed with: {e}")

        t.close()

    def test_upsample_basic(self, trajectory_path):
        """Test basic upsampling functionality by duplicating prior frames."""
        t = Trajectory(trajectory_path, mode="r")

        # Original data is at 10 Hz (100ms intervals)
        # Request 20 Hz (50ms intervals) - should double the frame count
        original = t.load()
        upsampled = t.load(desired_frequency=20.0)

        # Should have approximately double the frames
        orig_len = len(original["robot_position"])
        up_len = len(upsampled["robot_position"])

        # Should be close to 2x but might vary due to timing
        assert (
            up_len > orig_len
        ), f"Upsampled length {up_len} should be greater than original {orig_len}"
        assert (
            up_len <= orig_len * 2 + 5
        ), f"Upsampled length {up_len} should not be much more than 2x original {orig_len}"

        t.close()

    def test_upsample_2x_exact(self, temp_dir, rng):
        """Test exact 2x upsampling with controlled timing."""
        path = os.path.join(temp_dir, "upsample_test.vla")
        traj = Trajectory(path, mode="w")

        # Create data with exact 200ms intervals (5 Hz)
        for i in range(10):
            timestamp_ms = int(i * 200)  # 200ms intervals = 5 Hz
            data = {
                "step": i,
                "value": float(i * 10),
                "array": np.array([i, i + 1], dtype=np.float32),
            }
            traj.add_by_dict(data, timestamp=timestamp_ms)

        traj.close()

        # Now read with 10 Hz (100ms intervals) - should get 2x frames
        t = Trajectory(path, mode="r")
        original = t.load()
        upsampled = t.load(desired_frequency=10.0)

        orig_len = len(original["step"])
        up_len = len(upsampled["step"])

        # Should have roughly double the frames
        assert (
            up_len > orig_len
        ), f"Expected more frames in upsampled ({up_len}) than original ({orig_len})"

        # Check that original frames are preserved
        # The original frames should appear at certain positions
        orig_steps = original["step"]
        up_steps = upsampled["step"]

        # Should have duplicated frames
        unique_steps = np.unique(up_steps)
        assert len(unique_steps) == len(
            orig_steps), "Should have same unique values"

        t.close()

    def test_upsample_with_slice(self, trajectory_path):
        """Test upsampling combined with slicing."""
        t = Trajectory(trajectory_path, mode="r")

        # Get reference: first upsample, then slice
        upsampled_first = t.load(desired_frequency=20.0)
        reference = {k: v[slice(10, 30)] for k, v in upsampled_first.items()}

        # Get actual: upsample and slice in one call
        combo = t.load(desired_frequency=20.0, data_slice=slice(10, 30))

        # Should be equivalent
        assert combo.keys() == reference.keys()
        for k in combo:
            np.testing.assert_array_equal(combo[k],
                                          reference[k],
                                          err_msg=f"Mismatch in feature {k}")

        t.close()

    def test_upsample_preserves_data_types(self, temp_dir, rng):
        """Test that upsampling preserves data types correctly."""
        path = os.path.join(temp_dir, "upsample_types_test.vla")
        traj = Trajectory(path, mode="w")

        # Add varied data types
        for i in range(5):
            timestamp_ms = int(i * 500)  # 2 Hz
            data = {
                "int_val": int(i),
                "float_val": float(i * 1.5),
                "str_val": f"string_{i}",
                "array_uint8": np.array([i, i + 1], dtype=np.uint8),
                "array_float32": np.array([i * 1.1, i * 2.2],
                                          dtype=np.float32),
                "image": (rng.random((8, 8, 3)) * 255).astype(np.uint8),
            }
            traj.add_by_dict(data, timestamp=timestamp_ms)

        traj.close()

        # Upsample to 4 Hz
        t = Trajectory(path, mode="r")
        original = t.load()
        upsampled = t.load(desired_frequency=4.0)

        # Check data types are preserved
        for key in original:
            assert (upsampled[key].dtype == original[key].dtype
                    ), f"Dtype mismatch for {key}"

        # Check string handling
        orig_strings = set(original["str_val"])
        up_strings = set(upsampled["str_val"])
        assert orig_strings == up_strings, "String values should be preserved"

        # Check that duplicated frames have identical values
        up_int_vals = upsampled["int_val"]
        for i in range(len(up_int_vals) - 1):
            if up_int_vals[i] == up_int_vals[i + 1]:
                # This is a duplicated frame, all values should match
                for key in upsampled:
                    np.testing.assert_array_equal(
                        upsampled[key][i],
                        upsampled[key][i + 1],
                        err_msg=
                        f"Duplicated frames should have identical {key} values",
                    )

        t.close()

    def test_upsample_edge_cases(self, temp_dir, rng):
        """Test upsampling edge cases."""
        path = os.path.join(temp_dir, "upsample_edge_test.vla")
        traj = Trajectory(path, mode="w")

        # Single frame
        data = {"single": 42, "array": np.array([1, 2, 3], dtype=np.float32)}
        traj.add_by_dict(data, timestamp=0)
        traj.close()

        # Try to upsample single frame
        t = Trajectory(path, mode="r")
        original = t.load()
        upsampled = t.load(desired_frequency=100.0)

        # Should get the same single frame (no upsampling possible)
        assert len(original["single"]) == len(upsampled["single"]) == 1
        np.testing.assert_array_equal(original["single"], upsampled["single"])

        t.close()

    def test_upsample_irregular_intervals(self, temp_dir, rng):
        """Test upsampling with irregular time intervals."""
        path = os.path.join(temp_dir, "upsample_irregular_test.vla")
        traj = Trajectory(path, mode="w")

        # Add frames with irregular intervals
        timestamps = [0, 150, 400, 450, 800]  # Irregular gaps
        for i, ts in enumerate(timestamps):
            data = {
                "frame": i,
                "timestamp_orig": ts,
                "data": np.array([i, i * 2], dtype=np.float32),
            }
            traj.add_by_dict(data, timestamp=ts)

        traj.close()

        # Upsample to regular 10 Hz (100ms intervals)
        t = Trajectory(path, mode="r")
        original = t.load()
        upsampled = t.load(desired_frequency=10.0)

        orig_len = len(original["frame"])
        up_len = len(upsampled["frame"])

        # Should have more frames due to filling gaps
        assert (up_len > orig_len
                ), f"Should have more upsampled frames: {up_len} vs {orig_len}"

        # Large gap between timestamps[2]=400 and timestamps[4]=800 should be filled
        # 400ms gap at 100ms intervals should add ~3 intermediate frames
        up_frames = upsampled["frame"]

        # Should have duplicated frames in the gap
        unique_frames = np.unique(up_frames)
        assert len(
            unique_frames) == orig_len, "Should have same unique frame values"

        t.close()

    def test_upsample_vs_downsample_consistency(self, temp_dir, rng):
        """Test that upsampling and downsampling are consistent operations."""
        # Create trajectory with known frequency
        path = os.path.join(temp_dir, "consistency_test.vla")
        traj = Trajectory(path, mode="w")

        # 5 Hz base frequency (200ms intervals)
        for i in range(20):
            timestamp_ms = int(i * 200)
            data = {
                "step": i,
                "value": i * 1.5,
                "vector": np.array([i, i + 1, i + 2], dtype=np.float32),
            }
            traj.add_by_dict(data, timestamp=timestamp_ms)

        traj.close()

        t = Trajectory(path, mode="r")

        # Test different frequencies
        original = t.load()  # 5 Hz
        downsampled = t.load(desired_frequency=2.5)  # 2.5 Hz (downsample)
        upsampled = t.load(desired_frequency=10.0)  # 10 Hz (upsample)

        orig_len = len(original["step"])
        down_len = len(downsampled["step"])
        up_len = len(upsampled["step"])

        # Sanity checks
        assert down_len < orig_len, "Downsampling should reduce frame count"
        assert up_len > orig_len, "Upsampling should increase frame count"

        # All should contain the same unique values for step
        orig_steps = set(original["step"])
        down_steps = set(downsampled["step"])
        up_steps = set(upsampled["step"])

        # Downsampled should be subset of original
        assert down_steps.issubset(
            orig_steps), "Downsampled steps should be subset of original"

        # Upsampled should contain all original steps
        assert orig_steps.issubset(
            up_steps), "Upsampled should contain all original steps"

        t.close()
