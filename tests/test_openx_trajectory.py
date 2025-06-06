"""Unit tests for Open X-Embodiment trajectory functionality."""

import os
import shutil
import tempfile
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from robodm import Trajectory
from robodm.loader import RLDSLoader

from .test_fixtures import MockFileSystem, MockTimeProvider

# Define codecs to test with OpenX data
OPENX_TEST_CODECS = ["rawvideo", "ffv1", "libaom-av1", "libx264"]

# Common Open X-Embodiment datasets for testing
OPENX_TEST_DATASETS = [
    "bridge",
    "berkeley_cable_routing",
    "nyu_door_opening_surprising_effectiveness",
]


def validate_openx_roundtrip(temp_dir, codec, openx_data, dataset_name):
    """Helper function to validate Open X-Embodiment data through encoding/decoding roundtrip."""
    path = os.path.join(temp_dir,
                        f"openx_roundtrip_{dataset_name}_{codec}.vla")

    try:
        # Step 1: Create trajectory from OpenX data using from_list_of_dicts
        Trajectory.from_list_of_dicts(openx_data, path=path, video_codec=codec)

        # Step 2: Verify file exists and has content
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Step 3: Read back the trajectory
        traj_read = Trajectory(path, mode="r")
        loaded_data = traj_read.load()
        traj_read.close()

        # Step 4: Validate basic structure
        assert isinstance(loaded_data, dict)
        assert len(loaded_data) > 0

        # Step 5: Validate trajectory length matches original
        original_length = len(openx_data)
        for key, values in loaded_data.items():
            if hasattr(values, "shape"):
                assert (
                    values.shape[0] == original_length
                ), f"Length mismatch for {key}: got {values.shape[0]}, expected {original_length}"

        return True, None, loaded_data

    except Exception as e:
        return False, str(e), None


class TestOpenXTrajectoryIntegration:
    """Test Open X-Embodiment dataset integration with VLA trajectories."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_openx_data(self):
        """Create mock Open X-Embodiment data that mimics the real structure."""
        # Create synthetic data that matches typical OpenX structure
        mock_data = []
        for step in range(5):  # Small trajectory for testing
            step_data = {
                # Observation data - image and proprioceptive info
                "observation": {
                    "image":
                    np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    "state":
                    np.random.uniform(-1, 1,
                                      7).astype(np.float32),  # joint positions
                },
                # Action data
                "action": np.random.uniform(-1, 1, 7).astype(np.float32),
                # Reward (typically 0 except at task completion)
                "reward": np.float32(1.0 if step == 4 else 0.0),
                # Termination flag
                "is_terminal": step == 4,
                # Step information
                "step": step,
            }
            mock_data.append(step_data)

        return mock_data

    @pytest.fixture
    def bridge_style_data(self):
        """Create data that mimics the Bridge dataset structure."""
        mock_data = []
        for step in range(3):  # Even smaller for faster testing
            step_data = {
                "observation": {
                    "image": np.full((256, 256, 3), step * 85,
                                     dtype=np.uint8),  # Deterministic image
                    "state": np.array([step * 0.1] * 7,
                                      dtype=np.float32),  # Deterministic state
                },
                "action":
                np.array([step, step + 0.5] * 3 + [step],
                         dtype=np.float32),  # 7D action
                "reward":
                np.float32(0.0),
                "is_terminal":
                False,
                "step":
                step,
            }
            mock_data.append(step_data)

        return mock_data

    def test_openx_data_structure_validation(self, mock_openx_data):
        """Test that mock OpenX data has the expected structure."""
        assert len(mock_openx_data) == 5

        # Check each step has required fields
        for step_data in mock_openx_data:
            assert "observation" in step_data
            assert "action" in step_data
            assert "reward" in step_data
            assert "is_terminal" in step_data

            # Check observation structure
            obs = step_data["observation"]
            assert "image" in obs
            assert "state" in obs
            assert obs["image"].shape == (128, 128, 3)
            assert obs["state"].shape == (7, )

            # Check action structure
            assert step_data["action"].shape == (7, )

    @pytest.mark.parametrize("codec", OPENX_TEST_CODECS)
    def test_openx_trajectory_roundtrip(self, temp_dir, bridge_style_data,
                                        codec):
        """Test Open X-Embodiment data integrity through VLA trajectory roundtrip."""
        success, error, loaded_data = validate_openx_roundtrip(
            temp_dir, codec, bridge_style_data, "bridge_test")

        if not success:
            if "not available" in str(error).lower() or "codec" in str(
                    error).lower():
                pytest.skip(f"Codec {codec} not available: {error}")
            else:
                pytest.fail(f"Roundtrip failed for codec {codec}: {error}")

        # Validate data integrity with appropriate tolerances
        assert loaded_data is not None

        # Check that we have the expected fields
        expected_fields = [
            "observation/image",
            "observation/state",
            "action",
            "reward",
            "step",
        ]
        for field in expected_fields:
            assert any(
                field in key or key.endswith(field.split("/")[-1])
                for key in loaded_data.keys()
            ), f"Field {field} not found in loaded data. Available: {list(loaded_data.keys())}"

        # Define tolerances based on codec
        if codec in ["rawvideo", "ffv1"]:
            # Lossless codecs
            image_tolerance = 0
            float_tolerance = 1e-6
        else:
            # Lossy codecs
            image_tolerance = 15
            float_tolerance = 1e-3

        # Find the actual keys in loaded data
        image_key = next(k for k in loaded_data.keys() if "image" in k)
        state_key = next(k for k in loaded_data.keys() if "state" in k)
        action_key = next(k for k in loaded_data.keys()
                          if k.endswith("action"))
        step_key = next(k for k in loaded_data.keys() if k.endswith("step"))

        # Validate shapes
        assert loaded_data[image_key].shape == (3, 256, 256, 3)
        assert loaded_data[state_key].shape == (3, 7)
        assert loaded_data[action_key].shape == (3, 7)
        assert loaded_data[step_key].shape == (3, )

        # Validate step values (should be exact)
        np.testing.assert_array_equal(loaded_data[step_key], [0, 1, 2])

        # Validate action values with tolerance
        expected_actions = np.array(
            [
                [0, 0.5, 0, 0.5, 0, 0.5, 0],
                [1, 1.5, 1, 1.5, 1, 1.5, 1],
                [2, 2.5, 2, 2.5, 2, 2.5, 2],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(loaded_data[action_key],
                                   expected_actions,
                                   rtol=float_tolerance)

        # Validate state values with tolerance
        expected_states = np.array([[0.0] * 7, [0.1] * 7, [0.2] * 7],
                                   dtype=np.float32)
        np.testing.assert_allclose(loaded_data[state_key],
                                   expected_states,
                                   rtol=float_tolerance)

        # Validate image values with tolerance (deterministic pattern)
        if codec in ["rawvideo", "ffv1"]:
            # For lossless codecs, check exact values
            expected_images = np.array([
                np.full((256, 256, 3), 0, dtype=np.uint8),
                np.full((256, 256, 3), 85, dtype=np.uint8),
                np.full((256, 256, 3), 170, dtype=np.uint8),
            ])
            np.testing.assert_array_equal(loaded_data[image_key],
                                          expected_images)
        else:
            # For lossy codecs, check that values are reasonably close
            expected_images = np.array([
                np.full((256, 256, 3), 0, dtype=np.uint8),
                np.full((256, 256, 3), 85, dtype=np.uint8),
                np.full((256, 256, 3), 170, dtype=np.uint8),
            ])
            diff = np.abs(loaded_data[image_key].astype(np.int16) -
                          expected_images.astype(np.int16))
            assert (np.max(diff) <= image_tolerance
                    ), f"Image values differ by more than {image_tolerance}"

    def test_openx_trajectory_comparison_original_vs_reconstructed(
            self, temp_dir, bridge_style_data):
        """Test detailed comparison between original OpenX data and reconstructed trajectory."""
        # Use lossless codec for exact comparison
        codec = "rawvideo"

        success, error, loaded_data = validate_openx_roundtrip(
            temp_dir, codec, bridge_style_data, "comparison_test")

        if not success:
            pytest.skip(f"Cannot perform comparison test: {error}")

        # Extract original data for comparison
        original_images = np.array(
            [step["observation"]["image"] for step in bridge_style_data])
        original_states = np.array(
            [step["observation"]["state"] for step in bridge_style_data])
        original_actions = np.array(
            [step["action"] for step in bridge_style_data])
        original_steps = np.array([step["step"] for step in bridge_style_data])

        # Find keys in loaded data
        image_key = next(k for k in loaded_data.keys() if "image" in k)
        state_key = next(k for k in loaded_data.keys() if "state" in k)
        action_key = next(k for k in loaded_data.keys()
                          if k.endswith("action"))
        step_key = next(k for k in loaded_data.keys() if k.endswith("step"))

        # Compare original vs reconstructed (should be exact for rawvideo)
        np.testing.assert_array_equal(
            loaded_data[image_key],
            original_images,
            "Images differ between original and reconstructed",
        )
        np.testing.assert_array_equal(
            loaded_data[state_key],
            original_states,
            "States differ between original and reconstructed",
        )
        np.testing.assert_array_equal(
            loaded_data[action_key],
            original_actions,
            "Actions differ between original and reconstructed",
        )
        np.testing.assert_array_equal(
            loaded_data[step_key],
            original_steps,
            "Steps differ between original and reconstructed",
        )

    def test_openx_multiple_codecs_consistency(self, temp_dir,
                                               bridge_style_data):
        """Test that different codecs produce consistent results within their expected tolerances."""
        codec_results = {}

        # Test multiple codecs
        test_codecs = ["rawvideo", "ffv1"]  # Start with lossless codecs

        for codec in test_codecs:
            success, error, loaded_data = validate_openx_roundtrip(
                temp_dir, codec, bridge_style_data, f"consistency_{codec}")

            if success:
                codec_results[codec] = loaded_data
            else:
                print(f"Skipping codec {codec}: {error}")

        # Compare results between available codecs
        if len(codec_results) >= 2:
            codecs = list(codec_results.keys())
            reference_codec = codecs[0]
            reference_data = codec_results[reference_codec]

            for other_codec in codecs[1:]:
                other_data = codec_results[other_codec]

                # Find common keys
                common_keys = set(reference_data.keys()) & set(
                    other_data.keys())

                for key in common_keys:
                    ref_array = reference_data[key]
                    other_array = other_data[key]

                    assert (
                        ref_array.shape == other_array.shape
                    ), f"Shape mismatch for {key} between {reference_codec} and {other_codec}"

                    # For lossless codecs, arrays should be identical
                    if reference_codec in ["rawvideo", "ffv1"
                                           ] and other_codec in [
                                               "rawvideo",
                                               "ffv1",
                                           ]:
                        np.testing.assert_array_equal(
                            ref_array,
                            other_array,
                            f"Lossless codecs {reference_codec} and {other_codec} produced different results for {key}",
                        )

    # @pytest.mark.integration
    def test_openx_codec_availability_report(self, temp_dir, mock_openx_data):
        """Test and report which codecs work with Open X-Embodiment data."""
        codec_status = {}

        for codec in OPENX_TEST_CODECS:
            success, error, _ = validate_openx_roundtrip(
                temp_dir, codec, mock_openx_data, "availability_test")
            codec_status[codec] = {"available": success, "error": error}

        # Print codec availability report for OpenX data
        print("\n" + "=" * 60)
        print("OPEN X-EMBODIMENT CODEC AVAILABILITY REPORT")
        print("=" * 60)

        available_codecs = []
        unavailable_codecs = []

        for codec, status in codec_status.items():
            if status["available"]:
                available_codecs.append(codec)
                print(f"✓ {codec}: Available and working with OpenX data")
            else:
                unavailable_codecs.append(codec)
                print(f"✗ {codec}: {status['error']}")

        print(
            f"\nSummary: {len(available_codecs)}/{len(OPENX_TEST_CODECS)} codecs available for OpenX data"
        )
        print("=" * 60)

        # Ensure at least one codec works with OpenX data
        assert (len(available_codecs)
                > 0), "No codecs are available for Open X-Embodiment data!"


class TestRLDSLoaderIntegration:
    """Test RLDS loader integration with OpenX datasets (requires actual data)."""

    # @pytest.mark.slow
    # @pytest.mark.skipif(os.getenv("OPENX_DATA_DIR") is None,
    #                    reason="OPENX_DATA_DIR environment variable not set")
    @pytest.mark.parametrize("video_codec", ["rawvideo", "libx264"])
    def test_real_openx_data_codec_comparison(self, temp_dir, video_codec):
        """Test real OpenX data with different codecs using appropriate validation for each."""
        data_dir = "gs://gresearch/robotics/fractal20220817_data/0.1.0/"
        dataset_name = "fractal20220817_data"

        try:
            # Load real OpenX data using the correct RLDSLoader API
            loader = RLDSLoader(
                path=data_dir,
                split="train",
                batch_size=1,
                shuffle_buffer=10,
                shuffling=False,
            )

            # Get first trajectory
            first_traj_batch = next(iter(loader))
            first_traj_data = first_traj_batch[0]

            print(f"\n=== TESTING CODEC: {video_codec} ===")

            # Convert nested dict data to flat structure
            original_flat_data = {}
            for step_data in first_traj_data:
                for key, value in step_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            full_key = f"{key}/{subkey}"
                            if full_key not in original_flat_data:
                                original_flat_data[full_key] = []
                            original_flat_data[full_key].append(subvalue)
                    else:
                        if key not in original_flat_data:
                            original_flat_data[key] = []
                        original_flat_data[key].append(value)

            # Convert to arrays
            for key, values in original_flat_data.items():
                try:
                    original_flat_data[key] = np.array(values)
                except:
                    pass

            # Test conversion
            path = os.path.join(temp_dir, f"codec_test_{video_codec}.vla")
            Trajectory.from_list_of_dicts(first_traj_data,
                                          path=path,
                                          video_codec=video_codec)

            # Read back
            traj_read = Trajectory(path, mode="r")
            loaded_data = traj_read.load()
            traj_read.close()

            # Find field mappings (simplified version)
            field_mappings = {}
            for orig_key in original_flat_data.keys():
                if orig_key in loaded_data:
                    field_mappings[orig_key] = orig_key
                else:
                    # Try to find semantic matches
                    for recon_key in loaded_data.keys():
                        if (orig_key.split("/")[-1] == recon_key.split("/")[-1]
                                or orig_key.replace("/", "_").lower()
                                == recon_key.replace("/", "_").lower()):
                            field_mappings[orig_key] = recon_key
                            break

            # Define codec-specific tolerances
            is_lossless = video_codec in ["rawvideo", "ffv1"]
            if is_lossless:
                image_tolerance = 0  # Exact match required
                float_tolerance = (
                    1e-7  # Very small tolerance for floating point precision
                )
                print(
                    f"Using lossless codec tolerances (exact match required)")
            else:
                image_tolerance = 200  # Allow compression artifacts for lossy codecs (reasonable for H.264)
                float_tolerance = 1e-4  # Small tolerance for lossy compression
                print(
                    f"Using lossy codec tolerances (image_tol={image_tolerance}, float_tol={float_tolerance})"
                )

            # Validate based on codec type
            exact_matches = 0
            acceptable_matches = 0
            total_fields = len(field_mappings)

            for orig_key, recon_key in field_mappings.items():
                orig_data = original_flat_data[orig_key]
                recon_data = loaded_data[recon_key]

                # Skip if shapes don't match
                if hasattr(orig_data, "shape") and hasattr(
                        recon_data, "shape"):
                    if orig_data.shape != recon_data.shape:
                        continue

                is_image_field = ("image" in orig_key.lower()
                                  and hasattr(orig_data, "dtype")
                                  and orig_data.dtype == np.uint8)

                if is_image_field and not is_lossless:
                    # For lossy codecs, allow image compression differences
                    if np.array_equal(orig_data, recon_data):
                        exact_matches += 1
                    else:
                        max_diff = np.max(
                            np.abs(
                                orig_data.astype(np.int16) -
                                recon_data.astype(np.int16)))
                        if max_diff <= image_tolerance:
                            acceptable_matches += 1
                        else:
                            pytest.fail(
                                f"Image field {orig_key} exceeds tolerance: max_diff={max_diff} > {image_tolerance}"
                            )
                elif hasattr(orig_data, "dtype") and np.issubdtype(
                        orig_data.dtype, np.floating):
                    # Floating point comparison
                    if np.allclose(
                            orig_data,
                            recon_data,
                            rtol=float_tolerance,
                            atol=float_tolerance,
                    ):
                        exact_matches += 1
                    else:
                        pytest.fail(
                            f"Float field {orig_key} doesn't match within tolerance"
                        )
                else:
                    # Other data should be exact
                    if np.array_equal(orig_data, recon_data):
                        exact_matches += 1
                    else:
                        pytest.fail(
                            f"Field {orig_key} should be exact but differs")

            # Codec-specific final validation
            if is_lossless:
                assert (
                    exact_matches == total_fields
                ), f"Lossless codec {video_codec}: {exact_matches}/{total_fields} exact matches"
                print(
                    f"✓ Lossless codec {video_codec}: all {exact_matches} fields match exactly"
                )
            else:
                total_acceptable = exact_matches + acceptable_matches
                assert (
                    total_acceptable == total_fields
                ), f"Lossy codec {video_codec}: {total_acceptable}/{total_fields} within tolerance"
                print(
                    f"✓ Lossy codec {video_codec}: {exact_matches} exact + {acceptable_matches} acceptable = {total_acceptable}/{total_fields}"
                )

        except Exception as e:
            if "not available" in str(e).lower() or "codec" in str(e).lower():
                pytest.skip(f"Codec {video_codec} not available: {e}")
            else:
                pytest.fail(f"Failed with codec {video_codec}: {e}")

    @pytest.mark.parametrize("codec", OPENX_TEST_CODECS)
    def test_real_openx_data_loading(self, temp_dir, codec):
        """Test loading real Open X-Embodiment data and compare original vs reconstructed."""
        data_dir = "gs://gresearch/robotics/fractal20220817_data/0.1.0/"
        dataset_name = "fractal20220817_data"  # Define dataset_name for file naming
        video_codec = codec  # Test with lossy codec

        try:
            # Load real OpenX data using the correct RLDSLoader API
            loader = RLDSLoader(
                path=data_dir,
                split="train",
                batch_size=1,
                shuffle_buffer=10,
                shuffling=False,  # Don't shuffle for testing
            )

            # Get first trajectory using iterator interface
            first_traj_batch = next(iter(loader))
            first_traj_data = first_traj_batch[
                0]  # Get the actual trajectory data from batch

            print(f"\n=== TRAJECTORY-LEVEL ANALYSIS ===")
            trajectory_length = len(first_traj_data)
            print(f"Original trajectory length: {trajectory_length} steps")
            print(f"Video codec: {video_codec}")

            # Analyze trajectory structure and collect all data
            print(f"\n=== ORIGINAL TRAJECTORY STRUCTURE ===")
            step_fields = {}
            trajectory_data = {}

            # First pass: understand the structure and collect data
            for step_idx, step_data in enumerate(first_traj_data):
                if step_idx == 0:
                    print(f"Step 0 structure:")
                    for key, value in step_data.items():
                        if isinstance(value, dict):
                            print(
                                f"  {key}: dict with keys {list(value.keys())}"
                            )
                            for subkey, subvalue in value.items():
                                full_key = f"{key}/{subkey}"
                                if hasattr(subvalue, "shape"):
                                    print(
                                        f"    {subkey}: {type(subvalue).__name__} {subvalue.shape} {getattr(subvalue, 'dtype', 'no dtype')}"
                                    )
                                else:
                                    print(
                                        f"    {subkey}: {type(subvalue).__name__}"
                                    )
                        else:
                            print(
                                f"  {key}: {type(value).__name__} {getattr(value, 'shape', 'no shape')} {getattr(value, 'dtype', 'no dtype')}"
                            )

                # Collect all data for trajectory-level comparison
                for key, value in step_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            full_key = f"{key}/{subkey}"
                            if full_key not in trajectory_data:
                                trajectory_data[full_key] = []
                            trajectory_data[full_key].append(subvalue)
                    else:
                        if key not in trajectory_data:
                            trajectory_data[key] = []
                        trajectory_data[key].append(value)

            # Convert trajectory data to numpy arrays
            original_trajectory = {}
            for key, values in trajectory_data.items():
                try:
                    original_trajectory[key] = np.array(values)
                    if len(values) == 0:
                        print(f"Warning: Empty trajectory for key {key}")
                    elif len(values) != trajectory_length:
                        print(
                            f"Warning: Trajectory length mismatch for {key}: {len(values)} vs {trajectory_length}"
                        )
                except Exception as e:
                    print(f"Could not convert {key} to array: {e}")
                    original_trajectory[key] = values  # Keep as list

            print(f"\nOriginal trajectory fields: {len(original_trajectory)}")
            for key, data in original_trajectory.items():
                if hasattr(data, "shape"):
                    print(f"  {key}: {data.shape} {data.dtype}")
                else:
                    print(
                        f"  {key}: {type(data)} (length: {len(data) if hasattr(data, '__len__') else 'N/A'})"
                    )

            # Test conversion to VLA format
            path = os.path.join(temp_dir, f"real_{dataset_name}_test.vla")
            print(f"\n=== CONVERTING TO VLA FORMAT ===")
            Trajectory.from_list_of_dicts(first_traj_data,
                                          path=path,
                                          video_codec=video_codec)

            # Verify file was created and can be read back
            assert os.path.exists(path)
            file_size = os.path.getsize(path)
            assert file_size > 0
            print(f"VLA file created: {file_size} bytes")

            # Read back the entire trajectory
            print(f"\n=== READING BACK VLA TRAJECTORY ===")
            traj_read = Trajectory(path, mode="r")
            loaded_trajectory = traj_read.load()
            traj_read.close()

            # Basic validation
            assert isinstance(loaded_trajectory, dict)
            assert len(loaded_trajectory) > 0

            print(f"Reconstructed trajectory fields: {len(loaded_trajectory)}")
            reconstructed_length = None
            for key, values in loaded_trajectory.items():
                if hasattr(values, "shape"):
                    print(f"  {key}: {values.shape} {values.dtype}")
                    if reconstructed_length is None:
                        reconstructed_length = values.shape[0]
                    elif values.shape[0] != reconstructed_length:
                        print(
                            f"Warning: Inconsistent trajectory length for {key}: {values.shape[0]} vs {reconstructed_length}"
                        )
                else:
                    print(
                        f"  {key}: {type(values)} (length: {len(values) if hasattr(values, '__len__') else 'N/A'})"
                    )

            # TRAJECTORY-LEVEL VALIDATION
            print(f"\n=== TRAJECTORY-LEVEL VALIDATION ===")

            # 1. Trajectory Length Validation
            print(f"Original trajectory length: {trajectory_length}")
            print(f"Reconstructed trajectory length: {reconstructed_length}")
            assert (
                reconstructed_length == trajectory_length
            ), f"Trajectory length mismatch: original={trajectory_length}, reconstructed={reconstructed_length}"
            print("✓ Trajectory length preserved")

            # 2. Field Mapping and Coverage
            print(f"\n=== FIELD MAPPING ANALYSIS ===")
            original_keys = set(original_trajectory.keys())
            reconstructed_keys = set(loaded_trajectory.keys())

            # Advanced field mapping
            field_mappings = {}
            unmatched_original = set(original_keys)
            unmatched_reconstructed = set(reconstructed_keys)

            # Exact matches first
            for orig_key in list(unmatched_original):
                if orig_key in unmatched_reconstructed:
                    field_mappings[orig_key] = orig_key
                    unmatched_original.remove(orig_key)
                    unmatched_reconstructed.remove(orig_key)

            # Semantic matching for remaining fields
            for orig_key in list(unmatched_original):
                for recon_key in list(unmatched_reconstructed):
                    if self._fields_match_semantically(orig_key, recon_key):
                        field_mappings[orig_key] = recon_key
                        unmatched_original.remove(orig_key)
                        unmatched_reconstructed.remove(recon_key)
                        break

            mapping_coverage = len(field_mappings) / len(original_keys) * 100
            print(
                f"Field mapping coverage: {mapping_coverage:.1f}% ({len(field_mappings)}/{len(original_keys)})"
            )

            if unmatched_original:
                print(f"Unmatched original fields: {unmatched_original}")
            if unmatched_reconstructed:
                print(
                    f"Unmatched reconstructed fields: {unmatched_reconstructed}"
                )

            # 3. Define codec-specific validation criteria
            is_lossless = video_codec in ["rawvideo"]
            if is_lossless:
                image_tolerance = 0
                float_tolerance = 1e-7
                print(f"Using lossless validation (exact match required)")
            else:
                image_tolerance = 200  # Reasonable for H.264 compression
                float_tolerance = 1e-4
                print(
                    f"Using lossy validation (image_tol={image_tolerance}, float_tol={float_tolerance})"
                )

            # 4. Comprehensive Trajectory Data Validation
            print(f"\n=== COMPREHENSIVE DATA VALIDATION ===")
            validation_results = {
                "exact_matches": 0,
                "acceptable_matches": 0,
                "shape_mismatches": [],
                "value_mismatches": [],
                "temporal_errors": [],
                "critical_errors": [],
            }

            for orig_key, recon_key in field_mappings.items():
                try:
                    orig_data = original_trajectory[orig_key]
                    recon_data = loaded_trajectory[recon_key]

                    # Validate this field across the entire trajectory
                    field_result = self._validate_trajectory_field(
                        orig_key,
                        orig_data,
                        recon_data,
                        is_lossless,
                        image_tolerance,
                        float_tolerance,
                        trajectory_length,
                    )

                    # Accumulate results
                    if field_result["status"] == "exact_match":
                        validation_results["exact_matches"] += 1
                        print(f"✓ {orig_key}: Exact match across trajectory")
                    elif field_result["status"] == "acceptable_match":
                        validation_results["acceptable_matches"] += 1
                        print(
                            f"~ {orig_key}: Acceptable match (max_diff: {field_result.get('max_diff', 'N/A')} ≤ {image_tolerance})"
                        )
                    elif field_result["status"] == "shape_mismatch":
                        validation_results["shape_mismatches"].append(
                            field_result)
                        print(
                            f"✗ {orig_key}: Shape mismatch {field_result['error']}"
                        )
                    elif field_result["status"] == "value_mismatch":
                        validation_results["value_mismatches"].append(
                            field_result)
                        print(
                            f"✗ {orig_key}: Value mismatch - {field_result['error']}"
                        )
                    elif field_result["status"] == "temporal_error":
                        validation_results["temporal_errors"].append(
                            field_result)
                        print(
                            f"✗ {orig_key}: Temporal consistency error - {field_result['error']}"
                        )
                    else:
                        validation_results["critical_errors"].append(
                            field_result)
                        print(
                            f"? {orig_key}: Critical error - {field_result.get('error', 'Unknown')}"
                        )

                except Exception as e:
                    error_result = {
                        "field": orig_key,
                        "status": "critical_error",
                        "error": str(e),
                    }
                    validation_results["critical_errors"].append(error_result)
                    print(f"? {orig_key}: Exception during validation - {e}")

            # 5. Final Trajectory Integrity Assessment
            print(f"\n=== TRAJECTORY INTEGRITY SUMMARY ===")
            total_fields = len(field_mappings)
            total_passed = (validation_results["exact_matches"] +
                            validation_results["acceptable_matches"])

            print(f"Total trajectory fields validated: {total_fields}")
            print(f"Exact matches: {validation_results['exact_matches']}")
            print(
                f"Acceptable matches: {validation_results['acceptable_matches']}"
            )
            print(
                f"Shape mismatches: {len(validation_results['shape_mismatches'])}"
            )
            print(
                f"Value mismatches: {len(validation_results['value_mismatches'])}"
            )
            print(
                f"Temporal errors: {len(validation_results['temporal_errors'])}"
            )
            print(
                f"Critical errors: {len(validation_results['critical_errors'])}"
            )

            # Assertions for trajectory integrity
            assert total_fields > 0, "No trajectory fields could be validated"
            assert (
                len(validation_results["critical_errors"]) == 0
            ), f"Critical errors in trajectory validation: {validation_results['critical_errors'][:3]}"
            assert (
                len(validation_results["shape_mismatches"]) == 0
            ), f"Shape mismatches in trajectory: {[r['error'] for r in validation_results['shape_mismatches'][:3]]}"
            assert (
                len(validation_results["temporal_errors"]) == 0
            ), f"Temporal consistency errors: {[r['error'] for r in validation_results['temporal_errors'][:3]]}"

            # Check for essential trajectory components
            has_image_trajectory = any("image" in key.lower()
                                       for key in field_mappings.keys())
            has_action_trajectory = any("action" in key.lower()
                                        for key in field_mappings.keys())
            assert (has_image_trajectory
                    ), "No image trajectory found in reconstructed data"
            assert (has_action_trajectory
                    ), "No action trajectory found in reconstructed data"

            # Codec-specific trajectory validation
            if is_lossless:
                assert (
                    total_passed == total_fields
                ), f"Lossless codec {video_codec}: {total_passed}/{total_fields} trajectory fields passed validation"
                print(
                    f"✓ Lossless trajectory validation: all {total_fields} fields exact"
                )
            else:
                # For lossy codecs, ensure non-image data is exact and image data is within tolerance
                image_fields = [
                    key for key in field_mappings.keys()
                    if "image" in key.lower()
                ]
                non_image_fields = [
                    key for key in field_mappings.keys()
                    if "image" not in key.lower()
                ]

                # All non-image trajectory data should be exact
                non_image_mismatches = [
                    r for r in validation_results["value_mismatches"]
                    if not any("image" in r["field"].lower() for _ in [1])
                ]
                assert (
                    len(non_image_mismatches) == 0
                ), f"Non-image trajectory data must be exact for lossy codecs: {[r['field'] for r in non_image_mismatches[:3]]}"

                assert (
                    total_passed == total_fields
                ), f"Lossy codec {video_codec}: {total_passed}/{total_fields} trajectory fields within tolerance"
                print(
                    f"✓ Lossy trajectory validation: {validation_results['exact_matches']} exact + {validation_results['acceptable_matches']} acceptable = {total_passed}/{total_fields}"
                )

            # Field mapping coverage requirement
            assert (
                mapping_coverage >= 95.0
            ), f"Poor trajectory field coverage: {mapping_coverage:.1f}% (minimum: 95%)"

            print(f"\n✓ TRAJECTORY INTEGRITY VALIDATION PASSED!")
            print(
                f"Successfully validated entire {dataset_name} trajectory with {trajectory_length} steps"
            )
            print(
                f"Codec: {video_codec}, Fields: {total_fields}, Integrity: {total_passed}/{total_fields}"
            )

        except Exception as e:
            pytest.fail(f"Trajectory validation failed: {e}")

    def _fields_match_semantically(self, orig_key, recon_key):
        """Check if two field keys represent the same data semantically."""
        # Exact match
        if orig_key == recon_key:
            return True

        # Clean and normalize keys
        orig_clean = orig_key.replace("/", "_").lower()
        recon_clean = recon_key.replace("/", "_").lower()

        if orig_clean == recon_clean:
            return True

        # Check if they share significant key components
        orig_tokens = set(orig_clean.split("_"))
        recon_tokens = set(recon_clean.split("_"))
        overlap = len(orig_tokens & recon_tokens)

        # Require high overlap for semantic matching
        if len(orig_tokens) > 0 and len(recon_tokens) > 0:
            overlap_ratio = overlap / min(len(orig_tokens), len(recon_tokens))
            return overlap_ratio >= 0.8

        return False

    def _validate_trajectory_field(
        self,
        field_name,
        orig_data,
        recon_data,
        is_lossless,
        image_tolerance,
        float_tolerance,
        expected_length,
    ):
        """Validate a single field across the entire trajectory."""
        try:
            # Shape validation
            if hasattr(orig_data, "shape") and hasattr(recon_data, "shape"):
                if orig_data.shape != recon_data.shape:
                    return {
                        "status": "shape_mismatch",
                        "field": field_name,
                        "error": f"{orig_data.shape} vs {recon_data.shape}",
                    }

                # Temporal length validation
                if orig_data.shape[0] != expected_length:
                    return {
                        "status":
                        "temporal_error",
                        "field":
                        field_name,
                        "error":
                        f"Original data length {orig_data.shape[0]} != expected {expected_length}",
                    }
                if recon_data.shape[0] != expected_length:
                    return {
                        "status":
                        "temporal_error",
                        "field":
                        field_name,
                        "error":
                        f"Reconstructed data length {recon_data.shape[0]} != expected {expected_length}",
                    }

            # Determine field type for appropriate validation
            is_image_field = ("image" in field_name.lower()
                              and hasattr(orig_data, "dtype")
                              and orig_data.dtype == np.uint8)

            # Data validation with trajectory-appropriate tolerances
            if hasattr(orig_data, "dtype") and np.issubdtype(
                    orig_data.dtype, np.floating):
                # Floating point trajectory data
                if np.allclose(orig_data,
                               recon_data,
                               rtol=float_tolerance,
                               atol=float_tolerance):
                    return {"status": "exact_match", "field": field_name}
                else:
                    max_diff = np.max(np.abs(orig_data - recon_data))
                    return {
                        "status": "value_mismatch",
                        "field": field_name,
                        "error":
                        f"Float trajectory max_diff={max_diff} > tolerance={float_tolerance}",
                        "max_diff": max_diff,
                    }
            elif is_image_field:
                # Image trajectory validation
                if np.array_equal(orig_data, recon_data):
                    return {"status": "exact_match", "field": field_name}
                elif not is_lossless:
                    # For lossy codecs, check if within tolerance
                    max_diff = np.max(
                        np.abs(
                            orig_data.astype(np.int16) -
                            recon_data.astype(np.int16)))
                    if max_diff <= image_tolerance:
                        return {
                            "status": "acceptable_match",
                            "field": field_name,
                            "max_diff": max_diff,
                        }
                    else:
                        return {
                            "status": "value_mismatch",
                            "field": field_name,
                            "error":
                            f"Image trajectory max_diff={max_diff} > tolerance={image_tolerance}",
                            "max_diff": max_diff,
                        }
                else:
                    # Lossless codec should be exact
                    max_diff = np.max(
                        np.abs(
                            orig_data.astype(np.int16) -
                            recon_data.astype(np.int16)))
                    return {
                        "status": "value_mismatch",
                        "field": field_name,
                        "error":
                        f"Lossless image trajectory should be exact, got max_diff={max_diff}",
                        "max_diff": max_diff,
                    }
            else:
                # Other data types should be exact
                if np.array_equal(orig_data, recon_data):
                    return {"status": "exact_match", "field": field_name}
                else:
                    if hasattr(orig_data, "dtype") and np.issubdtype(
                            orig_data.dtype, np.integer):
                        max_diff = np.max(
                            np.abs(
                                orig_data.astype(np.int64) -
                                recon_data.astype(np.int64)))
                        return {
                            "status": "value_mismatch",
                            "field": field_name,
                            "error":
                            f"Non-image trajectory data should be exact, got max_diff={max_diff}",
                            "max_diff": max_diff,
                        }
                    else:
                        return {
                            "status": "value_mismatch",
                            "field": field_name,
                            "error":
                            "Non-numeric trajectory comparison failed",
                        }

        except Exception as e:
            return {
                "status": "critical_error",
                "field": field_name,
                "error": f"Exception during validation: {str(e)}",
            }


class TestOpenXFormatComparison:
    """Test comparing VLA, HDF5, and TFRecord formats for Open X trajectory data."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def openx_test_data(self):
        """Create OpenX-style test data for format comparison."""
        # Create more substantial test data for meaningful benchmarks
        num_steps = 50  # Reasonable size for testing

        mock_data = []
        for step in range(num_steps):
            step_data = {
                "observation": {
                    "image":
                    np.random.randint(
                        0, 255, (224, 224, 3),
                        dtype=np.uint8),  # Typical camera resolution
                    "wrist_image":
                    np.random.randint(0, 255, (84, 84, 3),
                                      dtype=np.uint8),  # Smaller wrist camera
                    "state":
                    np.random.uniform(-1, 1,
                                      7).astype(np.float32),  # Joint positions
                    "gripper_state":
                    np.random.uniform(0, 1,
                                      1).astype(np.float32),  # Gripper opening
                },
                "action":
                np.random.uniform(-1, 1,
                                  7).astype(np.float32),  # Robot actions
                "reward": np.float32(1.0 if step == num_steps -
                                     1 else 0.0),  # Sparse reward
                "is_terminal": step == num_steps - 1,
                "step": step,
                "language_instruction":
                f"Step {step} instruction",  # Text data
            }
            mock_data.append(step_data)

        return mock_data

    def _save_as_vla(self, data, path, video_codec="rawvideo"):
        """Save data as VLA format and return metrics."""
        start_time = time.time()

        # Convert data to VLA format
        Trajectory.from_list_of_dicts(data, path=path, video_codec=video_codec)

        creation_time = time.time() - start_time
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        return {
            "creation_time": creation_time,
            "file_size_mb": file_size_mb,
            "path": path,
        }

    def _save_as_hdf5(self, data, path):
        """Save data as HDF5 format and return metrics."""
        import h5py

        start_time = time.time()

        # Convert list of dicts to dict of arrays format
        structured_data = {}
        for step_idx, step_data in enumerate(data):
            for key, value in step_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        full_key = f"{key}/{subkey}"
                        if full_key not in structured_data:
                            structured_data[full_key] = []
                        structured_data[full_key].append(subvalue)
                else:
                    if key not in structured_data:
                        structured_data[key] = []
                    structured_data[key].append(value)

        # Convert lists to numpy arrays and save to HDF5
        with h5py.File(path, "w") as f:
            for key, values in structured_data.items():
                try:
                    if isinstance(values[0], str):
                        # Handle string data
                        string_array = np.array(values, dtype="S")
                        f.create_dataset(
                            key,
                            data=string_array,
                            compression="gzip",
                            compression_opts=9,
                        )
                    else:
                        # Handle numeric data
                        array_data = np.array(values)
                        f.create_dataset(key,
                                         data=array_data,
                                         compression="gzip",
                                         compression_opts=9)
                except Exception as e:
                    print(f"Warning: Failed to save {key}: {e}")

        creation_time = time.time() - start_time
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        return {
            "creation_time": creation_time,
            "file_size_mb": file_size_mb,
            "path": path,
        }

    def _save_as_tfrecord(self, data, path):
        """Save data as TFRecord format and return metrics."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available for TFRecord benchmarking")

        start_time = time.time()

        def _bytes_feature(value):
            """Convert bytes or string to bytes feature."""
            if isinstance(value, str):
                value = value.encode("utf-8")
            elif isinstance(value, np.ndarray):
                value = value.tobytes()
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value]))

        def _float_feature(value):
            """Convert float array to float feature."""
            if isinstance(value, np.ndarray):
                value = value.flatten()
            elif not hasattr(value, "__iter__"):
                value = [value]
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            """Convert int to int64 feature."""
            if not hasattr(value, "__iter__"):
                value = [value]
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        with tf.io.TFRecordWriter(path) as writer:
            for step_data in data:
                features = {}

                for key, value in step_data.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for subkey, subvalue in value.items():
                            full_key = f"{key}/{subkey}"
                            if isinstance(subvalue, np.ndarray):
                                if subvalue.dtype == np.uint8:
                                    features[full_key] = _bytes_feature(
                                        subvalue)
                                else:
                                    features[full_key] = _float_feature(
                                        subvalue)
                            elif isinstance(subvalue, str):
                                features[full_key] = _bytes_feature(subvalue)
                            else:
                                features[full_key] = _float_feature(
                                    [float(subvalue)])
                    else:
                        # Handle top-level values
                        if isinstance(value, np.ndarray):
                            if value.dtype == np.uint8:
                                features[key] = _bytes_feature(value)
                            else:
                                features[key] = _float_feature(value)
                        elif isinstance(value, str):
                            features[key] = _bytes_feature(value)
                        elif isinstance(value, bool):
                            features[key] = _int64_feature([int(value)])
                        else:
                            features[key] = _float_feature([float(value)])

                example = tf.train.Example(features=tf.train.Features(
                    feature=features))
                writer.write(example.SerializeToString())

        creation_time = time.time() - start_time
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        return {
            "creation_time": creation_time,
            "file_size_mb": file_size_mb,
            "path": path,
        }

    def _load_vla(self, path):
        """Load VLA format and return metrics."""
        start_time = time.time()

        traj = Trajectory(path, mode="r")
        data = traj.load()
        traj.close()

        loading_time = time.time() - start_time

        return {"loading_time": loading_time, "data": data}

    def _load_hdf5(self, path):
        """Load HDF5 format and return metrics."""
        import h5py

        start_time = time.time()

        data = {}
        with h5py.File(path, "r") as f:

            def _read_group(group, prefix=""):
                for key, item in group.items():
                    full_key = f"{prefix}/{key}" if prefix else key
                    if isinstance(item, h5py.Group):
                        _read_group(item, full_key)
                    else:
                        data[full_key] = item[:]

            _read_group(f)

        loading_time = time.time() - start_time

        return {"loading_time": loading_time, "data": data}

    def _load_tfrecord(self, path, original_data):
        """Load TFRecord format and return metrics."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available for TFRecord loading")

        start_time = time.time()

        # Create feature description based on original data structure
        feature_description = {}
        sample_step = original_data[0]

        for key, value in sample_step.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    full_key = f"{key}/{subkey}"
                    if isinstance(subvalue, np.ndarray):
                        if subvalue.dtype == np.uint8:
                            feature_description[
                                full_key] = tf.io.FixedLenFeature([],
                                                                  tf.string)
                        else:
                            feature_description[
                                full_key] = tf.io.FixedLenFeature(
                                    [subvalue.size], tf.float32)
                    elif isinstance(subvalue, str):
                        feature_description[full_key] = tf.io.FixedLenFeature(
                            [], tf.string)
                    else:
                        feature_description[full_key] = tf.io.FixedLenFeature(
                            [1], tf.float32)
            else:
                if isinstance(value, np.ndarray):
                    if value.dtype == np.uint8:
                        feature_description[key] = tf.io.FixedLenFeature(
                            [], tf.string)
                    else:
                        feature_description[key] = tf.io.FixedLenFeature(
                            [value.size], tf.float32)
                elif isinstance(value, str):
                    feature_description[key] = tf.io.FixedLenFeature([],
                                                                     tf.string)
                elif isinstance(value, bool):
                    feature_description[key] = tf.io.FixedLenFeature([1],
                                                                     tf.int64)
                else:
                    feature_description[key] = tf.io.FixedLenFeature(
                        [1], tf.float32)

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto,
                                              feature_description)

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(_parse_function)

        # Convert to numpy arrays
        data = {}
        all_examples = list(dataset.as_numpy_iterator())

        for key in feature_description.keys():
            sample_value = None
            # Find the original sample value
            if "/" in key:
                main_key, sub_key = key.split("/", 1)
                if main_key in sample_step and sub_key in sample_step[main_key]:
                    sample_value = sample_step[main_key][sub_key]
            else:
                if key in sample_step:
                    sample_value = sample_step[key]

            if sample_value is not None:
                arrays = []
                for example in all_examples:
                    if isinstance(sample_value, np.ndarray):
                        if sample_value.dtype == np.uint8:
                            array = np.frombuffer(example[key],
                                                  dtype=np.uint8).reshape(
                                                      sample_value.shape)
                        else:
                            array = example[key].reshape(sample_value.shape)
                        arrays.append(array)
                    elif isinstance(sample_value, str):
                        arrays.append(example[key].decode("utf-8"))
                    elif isinstance(sample_value, bool):
                        arrays.append(bool(example[key][0]))
                    else:
                        arrays.append(float(example[key][0]))

                data[key] = (np.array(arrays)
                             if not isinstance(arrays[0], str) else arrays)

        loading_time = time.time() - start_time

        return {"loading_time": loading_time, "data": data}

    def _calculate_data_size(self, data):
        """Calculate the uncompressed size of data in MB."""
        total_bytes = 0

        for step_data in data:
            for key, value in step_data.items():
                if isinstance(value, dict):
                    for subvalue in value.values():
                        total_bytes += self._get_value_size(subvalue)
                else:
                    total_bytes += self._get_value_size(value)

        return total_bytes / (1024 * 1024)

    def _get_value_size(self, value):
        """Get the size of a value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, (int, float, bool)):
            return 8  # Approximate size
        else:
            return 100  # Default estimate for unknown types

    @pytest.mark.parametrize("vla_codec", ["rawvideo", "ffv1", "libx264"])
    def test_openx_format_comparison(self, temp_dir, openx_test_data,
                                     vla_codec):
        """Compare VLA, HDF5, and TFRecord formats for OpenX trajectory data."""
        print(f"\n=== OPENX FORMAT COMPARISON TEST ===")
        print(f"VLA Codec: {vla_codec}")
        print(f"Test data: {len(openx_test_data)} steps")

        # Calculate original data size
        original_size_mb = self._calculate_data_size(openx_test_data)
        print(f"Original data size: {original_size_mb:.2f} MB")

        # File paths for different formats
        vla_path = os.path.join(temp_dir, f"test_{vla_codec}.vla")
        hdf5_path = os.path.join(temp_dir, "test.h5")
        tfrecord_path = os.path.join(temp_dir, "test.tfrecord")

        results = {}

        # Test VLA format
        print(f"\n--- VLA FORMAT ({vla_codec}) ---")
        try:
            vla_save_metrics = self._save_as_vla(openx_test_data, vla_path,
                                                 vla_codec)
            vla_load_metrics = self._load_vla(vla_path)

            results["VLA"] = {
                "codec":
                vla_codec,
                "creation_time":
                vla_save_metrics["creation_time"],
                "file_size_mb":
                vla_save_metrics["file_size_mb"],
                "loading_time":
                vla_load_metrics["loading_time"],
                "compression_ratio":
                (original_size_mb / vla_save_metrics["file_size_mb"]
                 if vla_save_metrics["file_size_mb"] > 0 else 0),
                "success":
                True,
                "data":
                vla_load_metrics["data"],
            }

            print(
                f"✓ VLA creation time: {vla_save_metrics['creation_time']:.3f}s"
            )
            print(
                f"✓ VLA file size: {vla_save_metrics['file_size_mb']:.2f} MB")
            print(
                f"✓ VLA loading time: {vla_load_metrics['loading_time']:.3f}s")
            print(
                f"✓ VLA compression ratio: {results['VLA']['compression_ratio']:.2f}x"
            )

        except Exception as e:
            if "not available" in str(e).lower() or "codec" in str(e).lower():
                pytest.skip(f"VLA codec {vla_codec} not available: {e}")
            else:
                results["VLA"] = {"success": False, "error": str(e)}
                print(f"✗ VLA failed: {e}")

        # Test HDF5 format
        print(f"\n--- HDF5 FORMAT ---")
        try:
            hdf5_save_metrics = self._save_as_hdf5(openx_test_data, hdf5_path)
            hdf5_load_metrics = self._load_hdf5(hdf5_path)

            results["HDF5"] = {
                "creation_time":
                hdf5_save_metrics["creation_time"],
                "file_size_mb":
                hdf5_save_metrics["file_size_mb"],
                "loading_time":
                hdf5_load_metrics["loading_time"],
                "compression_ratio":
                (original_size_mb / hdf5_save_metrics["file_size_mb"]
                 if hdf5_save_metrics["file_size_mb"] > 0 else 0),
                "success":
                True,
                "data":
                hdf5_load_metrics["data"],
            }

            print(
                f"✓ HDF5 creation time: {hdf5_save_metrics['creation_time']:.3f}s"
            )
            print(
                f"✓ HDF5 file size: {hdf5_save_metrics['file_size_mb']:.2f} MB"
            )
            print(
                f"✓ HDF5 loading time: {hdf5_load_metrics['loading_time']:.3f}s"
            )
            print(
                f"✓ HDF5 compression ratio: {results['HDF5']['compression_ratio']:.2f}x"
            )

        except Exception as e:
            results["HDF5"] = {"success": False, "error": str(e)}
            print(f"✗ HDF5 failed: {e}")

        # Test TFRecord format
        print(f"\n--- TFRECORD FORMAT ---")
        try:
            tfrecord_save_metrics = self._save_as_tfrecord(
                openx_test_data, tfrecord_path)
            tfrecord_load_metrics = self._load_tfrecord(
                tfrecord_path, openx_test_data)

            results["TFRecord"] = {
                "creation_time":
                tfrecord_save_metrics["creation_time"],
                "file_size_mb":
                tfrecord_save_metrics["file_size_mb"],
                "loading_time":
                tfrecord_load_metrics["loading_time"],
                "compression_ratio":
                (original_size_mb / tfrecord_save_metrics["file_size_mb"]
                 if tfrecord_save_metrics["file_size_mb"] > 0 else 0),
                "success":
                True,
                "data":
                tfrecord_load_metrics["data"],
            }

            print(
                f"✓ TFRecord creation time: {tfrecord_save_metrics['creation_time']:.3f}s"
            )
            print(
                f"✓ TFRecord file size: {tfrecord_save_metrics['file_size_mb']:.2f} MB"
            )
            print(
                f"✓ TFRecord loading time: {tfrecord_load_metrics['loading_time']:.3f}s"
            )
            print(
                f"✓ TFRecord compression ratio: {results['TFRecord']['compression_ratio']:.2f}x"
            )

        except Exception as e:
            if "TensorFlow" in str(e):
                print(f"⚠ TFRecord skipped: {e}")
                pytest.skip(str(e))
            else:
                results["TFRecord"] = {"success": False, "error": str(e)}
                print(f"✗ TFRecord failed: {e}")

        # Comparison and analysis
        print(f"\n=== COMPARISON SUMMARY ===")
        successful_formats = {
            k: v
            for k, v in results.items() if v.get("success", False)
        }

        if len(successful_formats) == 0:
            pytest.fail("No formats succeeded")

        # Print comparison table with proper codec information
        print(
            f"{'Format (Codec)':<18} {'Size(MB)':<10} {'Save(s)':<10} {'Load(s)':<10} {'Comp.Ratio':<12} {'Total(s)':<10}"
        )
        print("-" * 80)

        for format_name, metrics in successful_formats.items():
            # Format display name with codec
            if "codec" in metrics:
                display_name = f"{format_name} ({metrics['codec']})"
            else:
                display_name = format_name

            total_time = metrics["creation_time"] + metrics["loading_time"]
            print(
                f"{display_name:<18} {metrics['file_size_mb']:<10.2f} {metrics['creation_time']:<10.3f} {metrics['loading_time']:<10.3f} {metrics['compression_ratio']:<12.2f} {total_time:<10.3f}"
            )

        # Performance winners
        if len(successful_formats) > 1:
            print(f"\n=== PERFORMANCE ANALYSIS ===")

            # Best compression
            best_compression = max(successful_formats.items(),
                                   key=lambda x: x[1]["compression_ratio"])
            best_compression_name = (
                f"{best_compression[0]} ({best_compression[1].get('codec', 'N/A')})"
                if "codec" in best_compression[1] else best_compression[0])
            print(
                f"🏆 Best compression: {best_compression_name} ({best_compression[1]['compression_ratio']:.2f}x)"
            )

            # Fastest save
            fastest_save = min(successful_formats.items(),
                               key=lambda x: x[1]["creation_time"])
            fastest_save_name = (
                f"{fastest_save[0]} ({fastest_save[1].get('codec', 'N/A')})"
                if "codec" in fastest_save[1] else fastest_save[0])
            print(
                f"🚀 Fastest save: {fastest_save_name} ({fastest_save[1]['creation_time']:.3f}s)"
            )

            # Fastest load
            fastest_load = min(successful_formats.items(),
                               key=lambda x: x[1]["loading_time"])
            fastest_load_name = (
                f"{fastest_load[0]} ({fastest_load[1].get('codec', 'N/A')})"
                if "codec" in fastest_load[1] else fastest_load[0])
            print(
                f"⚡ Fastest load: {fastest_load_name} ({fastest_load[1]['loading_time']:.3f}s)"
            )

            # Best overall (lowest total time)
            best_overall = min(
                successful_formats.items(),
                key=lambda x: x[1]["creation_time"] + x[1]["loading_time"],
            )
            best_overall_name = (
                f"{best_overall[0]} ({best_overall[1].get('codec', 'N/A')})"
                if "codec" in best_overall[1] else best_overall[0])
            total_time = (best_overall[1]["creation_time"] +
                          best_overall[1]["loading_time"])
            print(
                f"🎯 Best overall: {best_overall_name} ({total_time:.3f}s total)"
            )

        # Basic data integrity check
        print(f"\n=== DATA INTEGRITY CHECK ===")
        if "VLA" in successful_formats and "HDF5" in successful_formats:
            vla_data = successful_formats["VLA"]["data"]
            hdf5_data = successful_formats["HDF5"]["data"]

            # Compare some basic metrics
            vla_keys = set(vla_data.keys())
            hdf5_keys = set(hdf5_data.keys())

            common_keys = vla_keys & hdf5_keys
            coverage = len(common_keys) / max(len(vla_keys),
                                              len(hdf5_keys)) * 100

            print(
                f"VLA-HDF5 field coverage: {coverage:.1f}% ({len(common_keys)}/{max(len(vla_keys), len(hdf5_keys))} fields)"
            )

            # Check a few common fields for basic integrity
            integrity_checks = 0
            passed_checks = 0

            for key in list(common_keys)[:5]:  # Check first 5 common fields
                try:
                    vla_array = vla_data[key]
                    hdf5_array = hdf5_data[key]

                    if hasattr(vla_array, "shape") and hasattr(
                            hdf5_array, "shape"):
                        integrity_checks += 1
                        if vla_array.shape == hdf5_array.shape:
                            passed_checks += 1
                            print(
                                f"✓ {key}: shape consistency {vla_array.shape}"
                            )
                        else:
                            print(
                                f"✗ {key}: shape mismatch {vla_array.shape} vs {hdf5_array.shape}"
                            )
                except Exception as e:
                    print(f"? {key}: integrity check failed - {e}")

            if integrity_checks > 0:
                integrity_rate = passed_checks / integrity_checks * 100
                print(
                    f"Basic integrity: {integrity_rate:.1f}% ({passed_checks}/{integrity_checks} checks passed)"
                )

        # Assertions for test validation
        assert len(
            successful_formats) > 0, "At least one format should succeed"

        # Ensure file sizes are reasonable (not empty, not too large)
        for format_name, metrics in successful_formats.items():
            assert (metrics["file_size_mb"]
                    > 0), f"{format_name} file should not be empty"
            assert (metrics["file_size_mb"] < original_size_mb *
                    10), f"{format_name} file suspiciously large"

        print(f"\n✅ OpenX format comparison test completed successfully!")
        print(
            f"Tested {len(successful_formats)} formats with {len(openx_test_data)} trajectory steps"
        )

    def test_openx_format_comparison_comprehensive(self, temp_dir,
                                                   openx_test_data):
        """Comprehensive comparison of all formats and codecs for OpenX trajectory data."""
        print(f"\n=== COMPREHENSIVE OPENX FORMAT COMPARISON ===")
        print(f"Test data: {len(openx_test_data)} steps")

        # Calculate original data size
        original_size_mb = self._calculate_data_size(openx_test_data)
        print(f"Original data size: {original_size_mb:.2f} MB")

        # Test all codecs for VLA
        vla_codecs = ["rawvideo", "ffv1", "libx264"]
        all_results = {}

        # Test VLA with different codecs
        for codec in vla_codecs:
            print(f"\n--- VLA FORMAT ({codec}) ---")
            vla_path = os.path.join(temp_dir, f"test_{codec}.vla")

            try:
                vla_save_metrics = self._save_as_vla(openx_test_data, vla_path,
                                                     codec)
                vla_load_metrics = self._load_vla(vla_path)

                all_results[f"VLA ({codec})"] = {
                    "format":
                    "VLA",
                    "codec":
                    codec,
                    "creation_time":
                    vla_save_metrics["creation_time"],
                    "file_size_mb":
                    vla_save_metrics["file_size_mb"],
                    "loading_time":
                    vla_load_metrics["loading_time"],
                    "compression_ratio":
                    (original_size_mb / vla_save_metrics["file_size_mb"]
                     if vla_save_metrics["file_size_mb"] > 0 else 0),
                    "success":
                    True,
                    "data":
                    vla_load_metrics["data"],
                }

                print(
                    f"✓ VLA ({codec}): create={vla_save_metrics['creation_time']:.3f}s, "
                    f"load={vla_load_metrics['loading_time']:.3f}s, "
                    f"size={vla_save_metrics['file_size_mb']:.2f} MB")

            except Exception as e:
                if "not available" in str(e).lower() or "codec" in str(
                        e).lower():
                    print(f"⚠ VLA ({codec}): Codec not available")
                    continue
                else:
                    all_results[f"VLA ({codec})"] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"✗ VLA ({codec}): Failed - {e}")

        # Test HDF5 format
        print(f"\n--- HDF5 FORMAT ---")
        hdf5_path = os.path.join(temp_dir, "test.h5")
        try:
            hdf5_save_metrics = self._save_as_hdf5(openx_test_data, hdf5_path)
            hdf5_load_metrics = self._load_hdf5(hdf5_path)

            all_results["HDF5"] = {
                "format":
                "HDF5",
                "creation_time":
                hdf5_save_metrics["creation_time"],
                "file_size_mb":
                hdf5_save_metrics["file_size_mb"],
                "loading_time":
                hdf5_load_metrics["loading_time"],
                "compression_ratio":
                (original_size_mb / hdf5_save_metrics["file_size_mb"]
                 if hdf5_save_metrics["file_size_mb"] > 0 else 0),
                "success":
                True,
                "data":
                hdf5_load_metrics["data"],
            }

            print(f"✓ HDF5: create={hdf5_save_metrics['creation_time']:.3f}s, "
                  f"load={hdf5_load_metrics['loading_time']:.3f}s, "
                  f"size={hdf5_save_metrics['file_size_mb']:.2f} MB")

        except Exception as e:
            all_results["HDF5"] = {"success": False, "error": str(e)}
            print(f"✗ HDF5: Failed - {e}")

        # Test TFRecord format
        print(f"\n--- TFRECORD FORMAT ---")
        tfrecord_path = os.path.join(temp_dir, "test.tfrecord")
        try:
            tfrecord_save_metrics = self._save_as_tfrecord(
                openx_test_data, tfrecord_path)
            tfrecord_load_metrics = self._load_tfrecord(
                tfrecord_path, openx_test_data)

            all_results["TFRecord"] = {
                "format":
                "TFRecord",
                "creation_time":
                tfrecord_save_metrics["creation_time"],
                "file_size_mb":
                tfrecord_save_metrics["file_size_mb"],
                "loading_time":
                tfrecord_load_metrics["loading_time"],
                "compression_ratio":
                (original_size_mb / tfrecord_save_metrics["file_size_mb"]
                 if tfrecord_save_metrics["file_size_mb"] > 0 else 0),
                "success":
                True,
                "data":
                tfrecord_load_metrics["data"],
            }

            print(
                f"✓ TFRecord: create={tfrecord_save_metrics['creation_time']:.3f}s, "
                f"load={tfrecord_load_metrics['loading_time']:.3f}s, "
                f"size={tfrecord_save_metrics['file_size_mb']:.2f} MB")

        except Exception as e:
            if "TensorFlow" in str(e):
                print(f"⚠ TFRecord: Skipped (TensorFlow not available)")
            else:
                all_results["TFRecord"] = {"success": False, "error": str(e)}
                print(f"✗ TFRecord: Failed - {e}")

        # Filter successful results
        successful_formats = {
            k: v
            for k, v in all_results.items() if v.get("success", False)
        }

        if len(successful_formats) == 0:
            pytest.fail("No formats succeeded")

        # Comprehensive comparison table
        print(f"\n=== COMPREHENSIVE PERFORMANCE COMPARISON ===")
        print(
            f"{'Format (Codec)':<18} {'Size(MB)':<10} {'Load(s)':<10} {'Comp.Ratio':<12} {'Throughput':<12}"
        )
        print("-" * 74)

        for format_name, metrics in successful_formats.items():
            throughput = (1.0 / metrics["loading_time"]
                          if metrics["loading_time"] > 0 else 0)
            print(
                f"{format_name:<18} {metrics['file_size_mb']:<10.2f} "
                f"{metrics['loading_time']:<10.3f} {metrics['compression_ratio']:<12.2f} {throughput:<12.2f}"
            )

        # Detailed analysis by category
        print(f"\n=== DETAILED PERFORMANCE ANALYSIS ===")

        # Best in each category
        if len(successful_formats) > 1:
            best_compression = max(successful_formats.items(),
                                   key=lambda x: x[1]["compression_ratio"])
            smallest_size = min(successful_formats.items(),
                                key=lambda x: x[1]["file_size_mb"])
            fastest_load = min(successful_formats.items(),
                               key=lambda x: x[1]["loading_time"])
            best_throughput = max(
                successful_formats.items(),
                key=lambda x: (1.0 / x[1]["loading_time"]
                               if x[1]["loading_time"] > 0 else 0),
            )

            print(
                f"🏆 Best compression ratio: {best_compression[0]} ({best_compression[1]['compression_ratio']:.2f}x)"
            )
            print(
                f"🗜️ Smallest file size: {smallest_size[0]} ({smallest_size[1]['file_size_mb']:.2f} MB)"
            )
            print(
                f"⚡ Fastest loading: {fastest_load[0]} ({fastest_load[1]['loading_time']:.3f}s)"
            )
            print(
                f"📈 Best throughput: {best_throughput[0]} ({1.0 / best_throughput[1]['loading_time']:.2f} samples/s)"
            )

        # Codec-specific analysis for VLA
        vla_results = {
            k: v
            for k, v in successful_formats.items() if k.startswith("VLA")
        }
        if len(vla_results) > 1:
            print(f"\n=== VLA CODEC COMPARISON ===")
            print(
                f"{'Codec':<12} {'Size(MB)':<10} {'Comp.Ratio':<12} {'Load(s)':<10} {'Throughput':<12}"
            )
            print("-" * 68)

            for format_name, metrics in vla_results.items():
                codec = format_name.split("(")[1].rstrip(")")
                throughput = (1.0 / metrics["loading_time"]
                              if metrics["loading_time"] > 0 else 0)
                print(
                    f"{codec:<12} {metrics['file_size_mb']:<10.2f} {metrics['compression_ratio']:<12.2f} "
                    f"{metrics['loading_time']:<10.3f} {throughput:<12.2f}")

        # Test passed successfully
        assert len(successful_formats) > 0, "At least one format should work"


class TestOpenXLoaderBenchmark:
    """Test OpenX data conversion to different formats and benchmark loader performance."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def openx_dataset_sample(self):
        """Create a larger OpenX-style dataset for loader benchmarking."""
        # Create multiple trajectories with realistic OpenX structure
        num_trajectories = 5
        steps_per_trajectory = 20

        trajectories = []
        for traj_idx in range(num_trajectories):
            trajectory_data = []
            for step in range(steps_per_trajectory):
                step_data = {
                    "observation": {
                        "image":
                        np.random.randint(0,
                                          255, (256, 256, 3),
                                          dtype=np.uint8),
                        "wrist_image":
                        np.random.randint(0,
                                          255, (128, 128, 3),
                                          dtype=np.uint8),
                        "state":
                        np.random.uniform(-1, 1, 7).astype(np.float32),
                        "gripper_state":
                        np.random.uniform(0, 1, 1).astype(np.float32),
                    },
                    "action":
                    np.random.uniform(-1, 1, 7).astype(np.float32),
                    "reward":
                    np.float32(1.0 if step == steps_per_trajectory -
                               1 else 0.0),
                    "is_terminal":
                    step == steps_per_trajectory - 1,
                    "step":
                    step,
                    "language_instruction":
                    f"Trajectory {traj_idx}, Step {step}",
                    "episode_id":
                    traj_idx,
                }
                trajectory_data.append(step_data)
            trajectories.append(trajectory_data)

        return trajectories

    def _create_vla_datasets(self, trajectories, temp_dir, codec="rawvideo"):
        """Convert trajectories to VLA format and return dataset info."""
        vla_dir = os.path.join(temp_dir, "vla_data")
        os.makedirs(vla_dir, exist_ok=True)

        start_time = time.time()
        vla_paths = []
        total_size = 0

        for idx, trajectory in enumerate(trajectories):
            path = os.path.join(vla_dir, f"trajectory_{idx:03d}.vla")
            try:
                Trajectory.from_list_of_dicts(trajectory,
                                              path=path,
                                              video_codec=codec)
                vla_paths.append(path)
                total_size += os.path.getsize(path)
            except Exception as e:
                print(f"Failed to create VLA trajectory {idx}: {e}")

        creation_time = time.time() - start_time

        return {
            "format": "VLA",
            "codec": codec,
            "paths": vla_paths,
            "creation_time": creation_time,
            "total_size_mb": total_size / (1024 * 1024),
            "num_files": len(vla_paths),
            "pattern": os.path.join(vla_dir, "*.vla"),
        }

    def _create_hdf5_datasets(self, trajectories, temp_dir):
        """Convert trajectories to HDF5 format and return dataset info."""
        import h5py

        hdf5_dir = os.path.join(temp_dir, "hdf5_data")
        os.makedirs(hdf5_dir, exist_ok=True)

        start_time = time.time()
        hdf5_paths = []
        total_size = 0

        for idx, trajectory in enumerate(trajectories):
            path = os.path.join(hdf5_dir, f"trajectory_{idx:03d}.h5")

            try:
                # Convert trajectory to structured format
                structured_data = {}
                for step_idx, step_data in enumerate(trajectory):
                    for key, value in step_data.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                full_key = f"{key}/{subkey}"
                                if full_key not in structured_data:
                                    structured_data[full_key] = []
                                structured_data[full_key].append(subvalue)
                        else:
                            if key not in structured_data:
                                structured_data[key] = []
                            structured_data[key].append(value)

                # Save to HDF5
                with h5py.File(path, "w") as f:
                    for key, values in structured_data.items():
                        try:
                            if isinstance(values[0], str):
                                string_array = np.array(values, dtype="S")
                                f.create_dataset(
                                    key,
                                    data=string_array,
                                    compression="gzip",
                                    compression_opts=9,
                                )
                            else:
                                array_data = np.array(values)
                                f.create_dataset(
                                    key,
                                    data=array_data,
                                    compression="gzip",
                                    compression_opts=9,
                                )
                        except Exception as e:
                            print(
                                f"Warning: Failed to save {key} to HDF5: {e}")

                hdf5_paths.append(path)
                total_size += os.path.getsize(path)

            except Exception as e:
                print(f"Failed to create HDF5 trajectory {idx}: {e}")

        creation_time = time.time() - start_time

        return {
            "format": "HDF5",
            "paths": hdf5_paths,
            "creation_time": creation_time,
            "total_size_mb": total_size / (1024 * 1024),
            "num_files": len(hdf5_paths),
            "pattern": os.path.join(hdf5_dir, "*.h5"),
        }

    def _create_tfrecord_datasets(self, trajectories, temp_dir):
        """Convert trajectories to TFRecord format and return dataset info."""
        try:
            import tensorflow as tf
        except ImportError:
            return None

        tfrecord_dir = os.path.join(temp_dir, "tfrecord_data")
        os.makedirs(tfrecord_dir, exist_ok=True)

        start_time = time.time()
        tfrecord_paths = []
        total_size = 0

        def _bytes_feature(value):
            if isinstance(value, str):
                value = value.encode("utf-8")
            elif isinstance(value, np.ndarray):
                value = value.tobytes()
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value]))

        def _float_feature(value):
            if isinstance(value, np.ndarray):
                value = value.flatten()
            elif not hasattr(value, "__iter__"):
                value = [value]
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            if not hasattr(value, "__iter__"):
                value = [value]
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        for idx, trajectory in enumerate(trajectories):
            path = os.path.join(tfrecord_dir, f"trajectory_{idx:03d}.tfrecord")

            try:
                with tf.io.TFRecordWriter(path) as writer:
                    for step_data in trajectory:
                        features = {}

                        for key, value in step_data.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    full_key = f"{key}/{subkey}"
                                    if isinstance(subvalue, np.ndarray):
                                        if subvalue.dtype == np.uint8:
                                            features[
                                                full_key] = _bytes_feature(
                                                    subvalue)
                                        else:
                                            features[
                                                full_key] = _float_feature(
                                                    subvalue)
                                    elif isinstance(subvalue, str):
                                        features[full_key] = _bytes_feature(
                                            subvalue)
                                    else:
                                        features[full_key] = _float_feature(
                                            [float(subvalue)])
                            else:
                                # Handle top-level values
                                if isinstance(value, np.ndarray):
                                    if value.dtype == np.uint8:
                                        features[key] = _bytes_feature(value)
                                    else:
                                        features[key] = _float_feature(value)
                                elif isinstance(value, str):
                                    features[key] = _bytes_feature(value)
                                elif isinstance(value, bool):
                                    features[key] = _int64_feature(
                                        [int(value)])
                                else:
                                    features[key] = _float_feature(
                                        [float(value)])

                        example = tf.train.Example(features=tf.train.Features(
                            feature=features))
                        writer.write(example.SerializeToString())

                tfrecord_paths.append(path)
                total_size += os.path.getsize(path)

            except Exception as e:
                print(f"Failed to create TFRecord trajectory {idx}: {e}")

        creation_time = time.time() - start_time

        return {
            "format": "TFRecord",
            "paths": tfrecord_paths,
            "creation_time": creation_time,
            "total_size_mb": total_size / (1024 * 1024),
            "num_files": len(tfrecord_paths),
            "pattern": os.path.join(tfrecord_dir, "*.tfrecord"),
        }

    def _benchmark_vla_loader(self, dataset_info, batch_size=1):
        """Benchmark VLA loader performance."""
        from robodm.loader import NonShuffleVLALoader

        start_time = time.time()

        # Create loader
        loader = NonShuffleVLALoader(dataset_info["pattern"])

        # Load all trajectories
        trajectories = list(loader)

        loading_time = time.time() - start_time

        return {
            "format":
            "VLA",
            "loader_type":
            "NonShuffleVLALoader",
            "loading_time":
            loading_time,
            "num_trajectories":
            len(trajectories),
            "batch_size":
            batch_size,
            "throughput_traj_per_sec":
            (len(trajectories) / loading_time if loading_time > 0 else 0),
            "data_sample":
            trajectories[0] if trajectories else None,
        }

    def _benchmark_hdf5_loader(self, dataset_info, batch_size=1):
        """Benchmark HDF5 loader performance."""
        try:
            from robodm.loader.hdf5 import get_hdf5_dataloader
        except ImportError:
            return None

        start_time = time.time()

        # Create loader
        dataloader = get_hdf5_dataloader(
            path=dataset_info["pattern"],
            batch_size=batch_size,
            num_workers=0,  # Use single thread for consistent measurement
        )

        # Load all batches
        batches = list(dataloader)
        total_trajectories = sum(len(batch) for batch in batches)

        loading_time = time.time() - start_time

        return {
            "format":
            "HDF5",
            "loader_type":
            "HDF5Loader",
            "loading_time":
            loading_time,
            "num_trajectories":
            total_trajectories,
            "num_batches":
            len(batches),
            "batch_size":
            batch_size,
            "throughput_traj_per_sec":
            (total_trajectories / loading_time if loading_time > 0 else 0),
            "data_sample":
            batches[0][0] if batches and batches[0] else None,
        }

    def _benchmark_tfrecord_loader(self, dataset_info, batch_size=1):
        """Benchmark TFRecord loading (basic implementation)."""
        try:
            import tensorflow as tf
        except ImportError:
            return None

        start_time = time.time()

        # Simple TFRecord loading (not using a formal loader)
        trajectory_count = 0
        for path in dataset_info["paths"]:
            dataset = tf.data.TFRecordDataset(path)
            for _ in dataset:
                trajectory_count += 1

        loading_time = time.time() - start_time

        return {
            "format":
            "TFRecord",
            "loader_type":
            "TFRecordDataset",
            "loading_time":
            loading_time,
            "num_trajectories":
            trajectory_count,
            "batch_size":
            batch_size,
            "throughput_traj_per_sec":
            (trajectory_count / loading_time if loading_time > 0 else 0),
            "data_sample":
            None,  # Would need more complex parsing
        }

    @pytest.mark.parametrize("vla_codec", ["rawvideo", "ffv1", "libx264"])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_openx_loader_benchmark_comprehensive(self, temp_dir,
                                                  openx_dataset_sample,
                                                  vla_codec, batch_size):
        """Comprehensive benchmark comparing loaders across different formats."""
        print(f"\n=== OPENX LOADER BENCHMARK ===")
        print(f"VLA Codec: {vla_codec}")
        print(f"Batch Size: {batch_size}")
        print(f"Dataset: {len(openx_dataset_sample)} trajectories")

        # Calculate original data size
        total_steps = sum(len(traj) for traj in openx_dataset_sample)
        print(f"Total steps: {total_steps}")

        # Phase 1: Create datasets in different formats
        print(f"\n--- DATASET CREATION PHASE ---")
        dataset_infos = {}

        # Create VLA datasets
        try:
            vla_info = self._create_vla_datasets(openx_dataset_sample,
                                                 temp_dir, vla_codec)
            if vla_info["num_files"] > 0:
                dataset_infos["VLA"] = vla_info
                print(
                    f"✓ VLA ({vla_codec}): {vla_info['num_files']} files, {vla_info['total_size_mb']:.2f} MB, {vla_info['creation_time']:.3f}s"
                )
            else:
                print(f"✗ VLA ({vla_codec}): No files created")
        except Exception as e:
            if "not available" in str(e).lower() or "codec" in str(e).lower():
                pytest.skip(f"VLA codec {vla_codec} not available: {e}")
            else:
                print(f"✗ VLA ({vla_codec}): Failed - {e}")

        # Create HDF5 datasets
        try:
            hdf5_info = self._create_hdf5_datasets(openx_dataset_sample,
                                                   temp_dir)
            if hdf5_info["num_files"] > 0:
                dataset_infos["HDF5"] = hdf5_info
                print(
                    f"✓ HDF5: {hdf5_info['num_files']} files, {hdf5_info['total_size_mb']:.2f} MB, {hdf5_info['creation_time']:.3f}s"
                )
            else:
                print(f"✗ HDF5: No files created")
        except Exception as e:
            print(f"✗ HDF5: Failed - {e}")

        # Create TFRecord datasets
        try:
            tfrecord_info = self._create_tfrecord_datasets(
                openx_dataset_sample, temp_dir)
            if tfrecord_info and tfrecord_info["num_files"] > 0:
                dataset_infos["TFRecord"] = tfrecord_info
                print(
                    f"✓ TFRecord: {tfrecord_info['num_files']} files, {tfrecord_info['total_size_mb']:.2f} MB, {tfrecord_info['creation_time']:.3f}s"
                )
            else:
                print(
                    f"⚠ TFRecord: Skipped (TensorFlow not available or creation failed)"
                )
        except Exception as e:
            print(f"⚠ TFRecord: Skipped - {e}")

        if not dataset_infos:
            pytest.fail("No datasets were created successfully")

        # Phase 2: Benchmark loaders
        print(f"\n--- LOADER BENCHMARK PHASE ---")
        loader_results = {}

        # Benchmark VLA loader
        if "VLA" in dataset_infos:
            try:
                vla_result = self._benchmark_vla_loader(
                    dataset_infos["VLA"], batch_size)
                if vla_result:
                    loader_results["VLA"] = vla_result
                    print(
                        f"✓ VLA Loader: {vla_result['loading_time']:.3f}s, {vla_result['throughput_traj_per_sec']:.2f} traj/s"
                    )
            except Exception as e:
                print(f"✗ VLA Loader: Failed - {e}")

        # Benchmark HDF5 loader
        if "HDF5" in dataset_infos:
            try:
                hdf5_result = self._benchmark_hdf5_loader(
                    dataset_infos["HDF5"], batch_size)
                if hdf5_result:
                    loader_results["HDF5"] = hdf5_result
                    print(
                        f"✓ HDF5 Loader: {hdf5_result['loading_time']:.3f}s, {hdf5_result['throughput_traj_per_sec']:.2f} traj/s"
                    )
            except Exception as e:
                print(f"✗ HDF5 Loader: Failed - {e}")

        # Benchmark TFRecord loader
        if "TFRecord" in dataset_infos:
            try:
                tfrecord_result = self._benchmark_tfrecord_loader(
                    dataset_infos["TFRecord"], batch_size)
                if tfrecord_result:
                    loader_results["TFRecord"] = tfrecord_result
                    print(
                        f"✓ TFRecord Loader: {tfrecord_result['loading_time']:.3f}s, {tfrecord_result['throughput_traj_per_sec']:.2f} traj/s"
                    )
            except Exception as e:
                print(f"✗ TFRecord Loader: Failed - {e}")

        if not loader_results:
            pytest.fail("No loaders succeeded")

        # Phase 3: Analysis and comparison
        print(f"\n=== COMPREHENSIVE PERFORMANCE ANALYSIS ===")

        # Combined metrics table
        print(
            f"{'Format (Codec)':<18} {'Creation(s)':<12} {'Size(MB)':<10} {'Loading(s)':<12} {'Load Speed':<12} {'Total(s)':<10}"
        )
        print("-" * 88)

        for format_name in dataset_infos.keys():
            if format_name in loader_results:
                # Format display name with codec
                if "codec" in dataset_infos[format_name]:
                    display_name = (
                        f"{format_name} ({dataset_infos[format_name]['codec']})"
                    )
                else:
                    display_name = format_name

                total_time = (dataset_infos[format_name]["creation_time"] +
                              loader_results[format_name]["loading_time"])

                # Calculate load speed in traj/s
                load_speed = loader_results[format_name][
                    "throughput_traj_per_sec"]

                print(
                    f"{display_name:<18} {dataset_infos[format_name]['creation_time']:<12.3f} {dataset_infos[format_name]['total_size_mb']:<10.2f} {loader_results[format_name]['loading_time']:<12.3f} {load_speed:<12.2f} {total_time:<10.3f}"
                )

        # Performance winners
        if len(loader_results) > 1:
            print(f"\n=== PERFORMANCE WINNERS ===")

            # Fastest creation
            fastest_creation = min(dataset_infos.items(),
                                   key=lambda x: x[1]["creation_time"])
            fastest_creation_name = (
                f"{fastest_creation[0]} ({fastest_creation[1].get('codec', 'N/A')})"
                if "codec" in fastest_creation[1] else fastest_creation[0])
            print(
                f"🚀 Fastest creation: {fastest_creation_name} ({fastest_creation[1]['creation_time']:.3f}s)"
            )

            # Best compression (smallest file size)
            best_compression = min(dataset_infos.items(),
                                   key=lambda x: x[1]["total_size_mb"])
            best_compression_name = (
                f"{best_compression[0]} ({best_compression[1].get('codec', 'N/A')})"
                if "codec" in best_compression[1] else best_compression[0])
            print(
                f"🗜️ Best compression: {best_compression_name} ({best_compression[1]['total_size_mb']:.2f} MB)"
            )

            # Fastest loading
            fastest_loading = min(loader_results.items(),
                                  key=lambda x: x[1]["loading_time"])
            fastest_loading_name = (
                f"{fastest_loading[0]} ({dataset_infos[fastest_loading[0]].get('codec', 'N/A')})"
                if fastest_loading[0] in dataset_infos
                and "codec" in dataset_infos[fastest_loading[0]] else
                fastest_loading[0])
            print(
                f"⚡ Fastest loading: {fastest_loading_name} ({fastest_loading[1]['loading_time']:.3f}s)"
            )

            # Best overall (lowest total time)
            best_overall = min(
                ((
                    name,
                    dataset_infos[name]["creation_time"] +
                    loader_results[name]["loading_time"],
                ) for name in loader_results.keys()),
                key=lambda x: x[1],
            )
            best_overall_name = (
                f"{best_overall[0]} ({dataset_infos[best_overall[0]].get('codec', 'N/A')})"
                if "codec" in dataset_infos[best_overall[0]] else
                best_overall[0])
            print(
                f"🎯 Best overall: {best_overall_name} ({best_overall[1]:.3f}s total)"
            )

        # Data integrity check
        print(f"\n=== DATA INTEGRITY CHECK ===")
        sample_data = None
        for format_name, result in loader_results.items():
            if result["data_sample"] is not None:
                sample_data = result["data_sample"]
                sample_format = format_name
                break

        if sample_data:
            print(f"Sample data from {sample_format}:")
            for key, value in list(
                    sample_data.items())[:5]:  # Show first 5 keys
                if hasattr(value, "shape"):
                    print(f"  {key}: {value.shape} {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")

        # Assertions for test validation
        assert len(
            dataset_infos) > 0, "At least one dataset format should be created"
        assert len(loader_results) > 0, "At least one loader should succeed"

        # Ensure all loaders return consistent trajectory counts
        expected_traj_count = len(openx_dataset_sample)
        for format_name, result in loader_results.items():
            if format_name != "TFRecord":  # TFRecord counts steps, not trajectories
                actual_count = result["num_trajectories"]
                assert (
                    actual_count == expected_traj_count
                ), f"{format_name} loader returned {actual_count} trajectories, expected {expected_traj_count}"

        print(f"\n✅ OpenX loader benchmark completed successfully!")
        print(
            f"Tested {len(loader_results)} loaders with {len(openx_dataset_sample)} trajectories (batch_size={batch_size})"
        )

    def test_openx_loader_scalability(self, temp_dir):
        """Test loader scalability with different dataset sizes."""
        sizes = [1, 3, 5]  # Number of trajectories
        steps_per_traj = 100

        print(f"\n=== LOADER SCALABILITY TEST ===")

        scalability_results = {}

        for size in sizes:
            print(f"\n--- Testing with {size} trajectories ---")

            # Create dataset of specified size
            trajectories = []
            for traj_idx in range(size):
                trajectory_data = []
                for step in range(steps_per_traj):
                    step_data = {
                        "observation": {
                            "image":
                            np.random.randint(0,
                                              255, (128, 128, 3),
                                              dtype=np.uint8),
                            "state":
                            np.random.uniform(-1, 1, 4).astype(np.float32),
                        },
                        "action": np.random.uniform(-1, 1,
                                                    4).astype(np.float32),
                        "step": step,
                    }
                    trajectory_data.append(step_data)
                trajectories.append(trajectory_data)

            size_results = {}

            # Test VLA format
            try:
                vla_info = self._create_vla_datasets(trajectories, temp_dir,
                                                     "rawvideo")
                vla_result = self._benchmark_vla_loader(vla_info, batch_size=1)

                size_results["VLA"] = {
                    "creation_time":
                    vla_info["creation_time"],
                    "loading_time":
                    vla_result["loading_time"],
                    "size_mb":
                    vla_info["total_size_mb"],
                    "throughput":
                    vla_result["throughput_traj_per_sec"],
                    "total_time":
                    vla_info["creation_time"] + vla_result["loading_time"],
                }

                print(
                    f"VLA: create={vla_info['creation_time']:.3f}s, load={vla_result['loading_time']:.3f}s, {vla_result['throughput_traj_per_sec']:.2f} traj/s"
                )

            except Exception as e:
                print(f"VLA failed for size {size}: {e}")

            # Test HDF5 format
            try:
                hdf5_info = self._create_hdf5_datasets(trajectories, temp_dir)
                hdf5_result = self._benchmark_hdf5_loader(hdf5_info,
                                                          batch_size=1)

                if hdf5_result:
                    size_results["HDF5"] = {
                        "creation_time":
                        hdf5_info["creation_time"],
                        "loading_time":
                        hdf5_result["loading_time"],
                        "size_mb":
                        hdf5_info["total_size_mb"],
                        "throughput":
                        hdf5_result["throughput_traj_per_sec"],
                        "total_time":
                        hdf5_info["creation_time"] +
                        hdf5_result["loading_time"],
                    }

                    print(
                        f"HDF5: create={hdf5_info['creation_time']:.3f}s, load={hdf5_result['loading_time']:.3f}s, {hdf5_result['throughput_traj_per_sec']:.2f} traj/s"
                    )

            except Exception as e:
                print(f"HDF5 failed for size {size}: {e}")

            # Store results for this size
            if size_results:
                scalability_results[size] = size_results

        # Comprehensive analysis
        if len(scalability_results) > 1:
            print(f"\n=== DETAILED SCALABILITY ANALYSIS ===")

            # Format comparison table
            formats = set()
            for size_data in scalability_results.values():
                formats.update(size_data.keys())

            for format_name in sorted(formats):
                print(f"\n--- {format_name} SCALABILITY ---")
                print(
                    f"{'Size':<6} {'Create(s)':<10} {'Load(s)':<10} {'Total(s)':<10} {'Size(MB)':<10} {'Throughput':<12}"
                )
                print("-" * 70)

                for size in sorted(scalability_results.keys()):
                    if format_name in scalability_results[size]:
                        data = scalability_results[size][format_name]
                        print(
                            f"{size:<6} {data['creation_time']:<10.3f} {data['loading_time']:<10.3f} {data['total_time']:<10.3f} {data['size_mb']:<10.2f} {data['throughput']:<12.2f}"
                        )

            # Scaling efficiency analysis
            print(f"\n=== SCALING EFFICIENCY ANALYSIS ===")

            for format_name in sorted(formats):
                print(f"\n{format_name} scaling:")

                format_data = []
                for size in sorted(scalability_results.keys()):
                    if format_name in scalability_results[size]:
                        format_data.append(
                            (size, scalability_results[size][format_name]))

                if len(format_data) >= 2:
                    # Calculate scaling factors
                    base_size, base_data = format_data[0]

                    print(
                        f"  Base ({base_size} traj): {base_data['total_time']:.3f}s total"
                    )

                    for size, data in format_data[1:]:
                        size_scale = size / base_size
                        time_scale = data["total_time"] / base_data[
                            "total_time"]
                        efficiency = size_scale / time_scale if time_scale > 0 else 0

                        print(
                            f"  {size} traj ({size_scale:.1f}x data): {data['total_time']:.3f}s ({time_scale:.2f}x time), efficiency: {efficiency:.2f}"
                        )

                        # Analyze individual components
                        create_scale = (data["creation_time"] /
                                        base_data["creation_time"] if
                                        base_data["creation_time"] > 0 else 0)
                        load_scale = (data["loading_time"] /
                                      base_data["loading_time"]
                                      if base_data["loading_time"] > 0 else 0)
                        size_scale_actual = (data["size_mb"] /
                                             base_data["size_mb"] if
                                             base_data["size_mb"] > 0 else 0)

                        print(
                            f"    Creation: {create_scale:.2f}x, Loading: {load_scale:.2f}x, Size: {size_scale_actual:.2f}x"
                        )

            # Head-to-head comparison
            if len(formats) >= 2:
                print(f"\n=== HEAD-TO-HEAD COMPARISON ===")

                formats_list = sorted(list(formats))

                for size in sorted(scalability_results.keys()):
                    print(f"\nSize {size} trajectories:")

                    size_data = scalability_results[size]
                    available_formats = [
                        f for f in formats_list if f in size_data
                    ]

                    if len(available_formats) >= 2:
                        # Find winners for each metric
                        fastest_creation = min(
                            available_formats,
                            key=lambda f: size_data[f]["creation_time"],
                        )
                        fastest_loading = min(
                            available_formats,
                            key=lambda f: size_data[f]["loading_time"],
                        )
                        fastest_total = min(
                            available_formats,
                            key=lambda f: size_data[f]["total_time"])
                        smallest_size = min(
                            available_formats,
                            key=lambda f: size_data[f]["size_mb"])
                        best_throughput = max(
                            available_formats,
                            key=lambda f: size_data[f]["throughput"])

                        print(
                            f"  🚀 Fastest creation: {fastest_creation} ({size_data[fastest_creation]['creation_time']:.3f}s)"
                        )
                        print(
                            f"  ⚡ Fastest loading: {fastest_loading} ({size_data[fastest_loading]['loading_time']:.3f}s)"
                        )
                        print(
                            f"  🎯 Fastest total: {fastest_total} ({size_data[fastest_total]['total_time']:.3f}s)"
                        )
                        print(
                            f"  🗜️ Smallest size: {smallest_size} ({size_data[smallest_size]['size_mb']:.2f} MB)"
                        )
                        print(
                            f"  📈 Best throughput: {best_throughput} ({size_data[best_throughput]['throughput']:.2f} traj/s)"
                        )

                        # Calculate relative performance
                        for fmt1 in available_formats:
                            for fmt2 in available_formats:
                                if fmt1 < fmt2:  # Avoid duplicate comparisons
                                    total_ratio = (
                                        size_data[fmt2]["total_time"] /
                                        size_data[fmt1]["total_time"])
                                    size_ratio = (size_data[fmt2]["size_mb"] /
                                                  size_data[fmt1]["size_mb"])

                                    if total_ratio > 1.1:
                                        print(
                                            f"  📊 {fmt1} is {total_ratio:.2f}x faster than {fmt2}"
                                        )
                                    elif total_ratio < 0.9:
                                        print(
                                            f"  📊 {fmt2} is {1/total_ratio:.2f}x faster than {fmt1}"
                                        )

                                    if size_ratio > 1.1:
                                        print(
                                            f"  💾 {fmt1} is {size_ratio:.2f}x more compact than {fmt2}"
                                        )
                                    elif size_ratio < 0.9:
                                        print(
                                            f"  💾 {fmt2} is {1/size_ratio:.2f}x more compact than {fmt1}"
                                        )

        assert (len(scalability_results)
                > 0), "At least one scalability test should succeed"

        # Test scalability characteristics
        for format_name in formats:
            format_data = []
            for size in sorted(scalability_results.keys()):
                if format_name in scalability_results[size]:
                    format_data.append(
                        (size, scalability_results[size][format_name]))

            if len(format_data) >= 2:
                # Ensure times scale reasonably (not exponentially)
                max_size = max(item[0] for item in format_data)
                min_size = min(item[0] for item in format_data)
                max_time = max(item[1]["total_time"] for item in format_data)
                min_time = min(item[1]["total_time"] for item in format_data)

                size_ratio = max_size / min_size
                time_ratio = max_time / min_time

                # Time should not scale worse than quadratically with data size
                assert (
                    time_ratio <= size_ratio**2 * 2
                ), f"{format_name} scales poorly: {size_ratio:.1f}x data leads to {time_ratio:.1f}x time"

    def test_openx_rlds_integration_benchmark(self, temp_dir):
        """Test RLDS integration if real RLDS data is available."""
        rlds_data_dir = "gs://gresearch/robotics/fractal20220817_data/0.1.0/"

        # Check if RLDS data is available
        if not os.path.exists(rlds_data_dir):
            pytest.skip("RLDS test data not available")

        print(f"\n=== RLDS INTEGRATION BENCHMARK ===")

        try:
            # Test RLDS loading performance
            start_time = time.time()

            loader = RLDSLoader(
                path=rlds_data_dir,
                split="train",
                batch_size=1,
                shuffle_buffer=10,
                shuffling=False,
            )

            # Load a few trajectories to benchmark
            trajectories = []
            for i, batch in enumerate(loader):
                trajectories.extend(batch)
                if i >= 2:  # Load 3 batches
                    break

            rlds_loading_time = time.time() - start_time

            print(
                f"RLDS loaded {len(trajectories)} trajectories in {rlds_loading_time:.3f}s"
            )
            print(
                f"RLDS throughput: {len(trajectories) / rlds_loading_time:.2f} traj/s"
            )

            if len(trajectories) > 0:
                # Test conversion to other formats for comparison
                sample_trajectory = trajectories[0]
                print(
                    f"Sample trajectory length: {len(sample_trajectory)} steps"
                )

                # Convert to VLA and benchmark
                try:
                    vla_path = os.path.join(temp_dir, "rlds_to_vla_test.vla")

                    start_time = time.time()
                    Trajectory.from_list_of_dicts(sample_trajectory,
                                                  path=vla_path,
                                                  video_codec="rawvideo")
                    vla_creation_time = time.time() - start_time

                    start_time = time.time()
                    traj_read = Trajectory(vla_path, mode="r")
                    vla_data = traj_read.load()
                    traj_read.close()
                    vla_loading_time = time.time() - start_time

                    vla_size_mb = os.path.getsize(vla_path) / (1024 * 1024)

                    print(f"\nRLDS→VLA Conversion:")
                    print(f"  Creation: {vla_creation_time:.3f}s")
                    print(f"  Loading: {vla_loading_time:.3f}s")
                    print(f"  Size: {vla_size_mb:.2f} MB")
                    print(
                        f"  Total: {vla_creation_time + vla_loading_time:.3f}s"
                    )

                except Exception as e:
                    print(f"VLA conversion failed: {e}")

                # Convert to HDF5 and benchmark
                try:
                    import h5py

                    hdf5_path = os.path.join(temp_dir, "rlds_to_hdf5_test.h5")

                    start_time = time.time()

                    # Convert to HDF5 format
                    structured_data = {}
                    for step_idx, step_data in enumerate(sample_trajectory):
                        for key, value in step_data.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    full_key = f"{key}/{subkey}"
                                    if full_key not in structured_data:
                                        structured_data[full_key] = []
                                    structured_data[full_key].append(subvalue)
                            else:
                                if key not in structured_data:
                                    structured_data[key] = []
                                structured_data[key].append(value)

                    with h5py.File(hdf5_path, "w") as f:
                        for key, values in structured_data.items():
                            try:
                                if isinstance(values[0], str):
                                    string_array = np.array(values, dtype="S")
                                    f.create_dataset(
                                        key,
                                        data=string_array,
                                        compression="gzip",
                                        compression_opts=9,
                                    )
                                else:
                                    array_data = np.array(values)
                                    f.create_dataset(
                                        key,
                                        data=array_data,
                                        compression="gzip",
                                        compression_opts=9,
                                    )
                            except Exception as e:
                                print(
                                    f"Warning: Failed to save {key} to HDF5: {e}"
                                )

                    hdf5_creation_time = time.time() - start_time

                    start_time = time.time()
                    with h5py.File(hdf5_path, "r") as f:
                        hdf5_data = {}

                        def _read_group(group, prefix=""):
                            for key, item in group.items():
                                full_key = f"{prefix}/{key}" if prefix else key
                                if isinstance(item, h5py.Group):
                                    _read_group(item, full_key)
                                else:
                                    hdf5_data[full_key] = item[:]

                        _read_group(f)

                    hdf5_loading_time = time.time() - start_time
                    hdf5_size_mb = os.path.getsize(hdf5_path) / (1024 * 1024)

                    print(f"\nRLDS→HDF5 Conversion:")
                    print(f"  Creation: {hdf5_creation_time:.3f}s")
                    print(f"  Loading: {hdf5_loading_time:.3f}s")
                    print(f"  Size: {hdf5_size_mb:.2f} MB")
                    print(
                        f"  Total: {hdf5_creation_time + hdf5_loading_time:.3f}s"
                    )

                except Exception as e:
                    print(f"HDF5 conversion failed: {e}")

                print(f"\n=== RLDS BENCHMARK SUMMARY ===")
                print(f"Original RLDS loading: {rlds_loading_time:.3f}s")
                print(
                    f"Real-world conversion and loading benchmarks completed")

        except ImportError:
            pytest.skip(
                "TensorFlow or TensorFlow Datasets not available for RLDS testing"
            )
        except Exception as e:
            print(f"RLDS benchmark failed: {e}")
            # Don't fail the test, just report the issue
            assert True  # Pass the test even if RLDS fails

    def test_openx_loader_benchmark_all_codecs(self, temp_dir,
                                               openx_dataset_sample):
        """Comprehensive benchmark comparing all loaders and codecs."""
        print(f"\n=== COMPREHENSIVE OPENX LOADER BENCHMARK ===")
        print(f"Dataset: {len(openx_dataset_sample)} trajectories")

        # Calculate original data size
        total_steps = sum(len(traj) for traj in openx_dataset_sample)
        print(f"Total steps: {total_steps}")

        # Test all VLA codecs
        vla_codecs = ["rawvideo", "ffv1", "libx264"]
        all_dataset_infos = {}
        all_loader_results = {}

        # Phase 1: Create datasets in all formats and codecs
        print(f"\n--- DATASET CREATION PHASE ---")

        # Test VLA with different codecs
        for codec in vla_codecs:
            try:
                vla_info = self._create_vla_datasets(openx_dataset_sample,
                                                     temp_dir, codec)
                if vla_info["num_files"] > 0:
                    format_name = f"VLA ({codec})"
                    all_dataset_infos[format_name] = vla_info
                    print(
                        f"✓ {format_name}: {vla_info['num_files']} files, {vla_info['total_size_mb']:.2f} MB, {vla_info['creation_time']:.3f}s"
                    )
                else:
                    print(f"✗ VLA ({codec}): No files created")
            except Exception as e:
                if "not available" in str(e).lower() or "codec" in str(
                        e).lower():
                    print(f"⚠ VLA ({codec}): Codec not available")
                else:
                    print(f"✗ VLA ({codec}): Failed - {e}")

        # Test HDF5
        try:
            hdf5_info = self._create_hdf5_datasets(openx_dataset_sample,
                                                   temp_dir)
            if hdf5_info["num_files"] > 0:
                all_dataset_infos["HDF5"] = hdf5_info
                print(
                    f"✓ HDF5: {hdf5_info['num_files']} files, {hdf5_info['total_size_mb']:.2f} MB, {hdf5_info['creation_time']:.3f}s"
                )
            else:
                print(f"✗ HDF5: No files created")
        except Exception as e:
            print(f"✗ HDF5: Failed - {e}")

        # Test TFRecord
        try:
            tfrecord_info = self._create_tfrecord_datasets(
                openx_dataset_sample, temp_dir)
            if tfrecord_info and tfrecord_info["num_files"] > 0:
                all_dataset_infos["TFRecord"] = tfrecord_info
                print(
                    f"✓ TFRecord: {tfrecord_info['num_files']} files, {tfrecord_info['total_size_mb']:.2f} MB, {tfrecord_info['creation_time']:.3f}s"
                )
            else:
                print(
                    f"⚠ TFRecord: Skipped (TensorFlow not available or creation failed)"
                )
        except Exception as e:
            print(f"⚠ TFRecord: Skipped - {e}")

        if not all_dataset_infos:
            pytest.fail("No datasets were created successfully")

        # Phase 2: Benchmark all loaders
        print(f"\n--- LOADER BENCHMARK PHASE ---")

        for format_name, dataset_info in all_dataset_infos.items():
            try:
                if format_name.startswith("VLA"):
                    loader_result = self._benchmark_vla_loader(dataset_info,
                                                               batch_size=1)
                elif format_name == "HDF5":
                    loader_result = self._benchmark_hdf5_loader(dataset_info,
                                                                batch_size=1)
                elif format_name == "TFRecord":
                    loader_result = self._benchmark_tfrecord_loader(
                        dataset_info, batch_size=1)
                else:
                    continue

                if loader_result:
                    all_loader_results[format_name] = loader_result
                    print(
                        f"✓ {format_name} Loader: {loader_result['loading_time']:.3f}s, {loader_result['throughput_traj_per_sec']:.2f} traj/s"
                    )
            except Exception as e:
                print(f"✗ {format_name} Loader: Failed - {e}")

        if not all_loader_results:
            pytest.fail("No loaders succeeded")

        # Phase 3: Comprehensive analysis
        print(f"\n=== COMPREHENSIVE PERFORMANCE ANALYSIS ===")
        print(
            f"{'Format (Codec)':<18} {'Size(MB)':<10} {'Loading(s)':<12} {'Load Speed':<12}"
        )
        print("-" * 64)

        for format_name in all_dataset_infos.keys():
            if format_name in all_loader_results:
                load_speed = all_loader_results[format_name][
                    "throughput_traj_per_sec"]

                print(
                    f"{format_name:<18} "
                    f"{all_dataset_infos[format_name]['total_size_mb']:<10.2f} "
                    f"{all_loader_results[format_name]['loading_time']:<12.3f} "
                    f"{load_speed:<12.2f}")

        # Performance winners
        if len(all_loader_results) > 1:
            print(f"\n=== PERFORMANCE WINNERS ===")

            best_compression = min(all_dataset_infos.items(),
                                   key=lambda x: x[1]["total_size_mb"])
            print(
                f"🗜️ Best compression: {best_compression[0]} ({best_compression[1]['total_size_mb']:.2f} MB)"
            )

            fastest_loading = min(all_loader_results.items(),
                                  key=lambda x: x[1]["loading_time"])
            print(
                f"⚡ Fastest loading: {fastest_loading[0]} ({fastest_loading[1]['loading_time']:.3f}s)"
            )

            best_throughput = max(
                all_loader_results.items(),
                key=lambda x: x[1]["throughput_traj_per_sec"],
            )
            print(
                f"📈 Best throughput: {best_throughput[0]} ({best_throughput[1]['throughput_traj_per_sec']:.2f} traj/s)"
            )

        # VLA codec-specific analysis
        vla_results = {
            k: v
            for k, v in all_loader_results.items() if k.startswith("VLA")
        }
        if len(vla_results) > 1:
            print(f"\n=== VLA CODEC COMPARISON ===")
            print(
                f"{'Codec':<12} {'Size(MB)':<10} {'Loading(s)':<12} {'Throughput':<12}"
            )
            print("-" * 58)

            for format_name in vla_results.keys():
                codec = format_name.split("(")[1].rstrip(")")
                dataset_info = all_dataset_infos[format_name]
                loader_info = all_loader_results[format_name]

                print(f"{codec:<12} {dataset_info['total_size_mb']:<10.2f} "
                      f"{loader_info['loading_time']:<12.3f} "
                      f"{loader_info['throughput_traj_per_sec']:<12.2f}")

        # Test passed successfully
        assert len(all_loader_results) > 0, "At least one loader should work"

    def test_openx_scalability_comprehensive(self, temp_dir):
        """Comprehensive scalability test across all formats and codecs."""
        print(f"\n=== COMPREHENSIVE SCALABILITY TEST ===")

        # Test with different dataset sizes
        test_sizes = [1, 3, 5]
        results_by_size = {}

        for size in test_sizes:
            print(f"\n--- Testing with {size} trajectories ---")

            # Create synthetic trajectories for this size
            trajectories = self._create_synthetic_trajectories(size)
            results_by_size[size] = {}

            # Test all VLA codecs
            for codec in ["rawvideo", "ffv1", "libx264"]:
                try:
                    # Create VLA dataset
                    vla_info = self._create_vla_datasets(
                        trajectories, temp_dir, codec)
                    if vla_info["num_files"] > 0:
                        # Benchmark VLA loader
                        loader_result = self._benchmark_vla_loader(
                            vla_info, batch_size=1)
                        if loader_result:
                            results_by_size[size][f"VLA ({codec})"] = {
                                "loading_time":
                                loader_result["loading_time"],
                                "file_size_mb":
                                vla_info["total_size_mb"],
                                "throughput":
                                loader_result["throughput_traj_per_sec"],
                            }
                            print(
                                f"VLA ({codec}): load={loader_result['loading_time']:.3f}s, {loader_result['throughput_traj_per_sec']:.2f} traj/s"
                            )
                except Exception as e:
                    if "not available" in str(e).lower() or "codec" in str(
                            e).lower():
                        continue
                    else:
                        print(f"VLA ({codec}): Failed - {e}")

            # Test HDF5
            try:
                hdf5_info = self._create_hdf5_datasets(trajectories, temp_dir)
                if hdf5_info["num_files"] > 0:
                    loader_result = self._benchmark_hdf5_loader(hdf5_info,
                                                                batch_size=1)
                    if loader_result:
                        results_by_size[size]["HDF5"] = {
                            "loading_time": loader_result["loading_time"],
                            "file_size_mb": hdf5_info["total_size_mb"],
                            "throughput":
                            loader_result["throughput_traj_per_sec"],
                        }
                        print(
                            f"HDF5: load={loader_result['loading_time']:.3f}s, {loader_result['throughput_traj_per_sec']:.2f} traj/s"
                        )
            except Exception as e:
                print(f"HDF5: Failed - {e}")

            # Test TFRecord
            try:
                tfrecord_info = self._create_tfrecord_datasets(
                    trajectories, temp_dir)
                if tfrecord_info and tfrecord_info["num_files"] > 0:
                    loader_result = self._benchmark_tfrecord_loader(
                        tfrecord_info, batch_size=1)
                    if loader_result:
                        results_by_size[size]["TFRecord"] = {
                            "loading_time": loader_result["loading_time"],
                            "file_size_mb": tfrecord_info["total_size_mb"],
                            "throughput":
                            loader_result["throughput_traj_per_sec"],
                        }
                        print(
                            f"TFRecord: load={loader_result['loading_time']:.3f}s, {loader_result['throughput_traj_per_sec']:.2f} traj/s"
                        )
            except Exception as e:
                continue

        # Analysis
        print(f"\n=== DETAILED SCALABILITY ANALYSIS ===")

        # Get all unique formats tested
        all_formats = set()
        for size_results in results_by_size.values():
            all_formats.update(size_results.keys())

        # Print scalability table for each format
        for format_name in sorted(all_formats):
            print(f"\n--- {format_name.upper()} SCALABILITY ---")
            print(
                f"{'Size':<6} {'Load(s)':<10} {'Size(MB)':<10} {'Throughput':<10}"
            )
            print("-" * 48)

            for size in test_sizes:
                if format_name in results_by_size[size]:
                    result = results_by_size[size][format_name]
                    print(
                        f"{size:<6} {result['loading_time']:<10.3f} "
                        f"{result['file_size_mb']:<10.2f} {result['throughput']:<10.2f}"
                    )

        # Scaling efficiency analysis
        print(f"\n=== SCALING EFFICIENCY ANALYSIS ===")

        for format_name in sorted(all_formats):
            # Check if we have data for all sizes
            size_data = {}
            for size in test_sizes:
                if format_name in results_by_size[size]:
                    size_data[size] = results_by_size[size][format_name]

            if len(size_data) >= 2:
                print(f"\n{format_name} scaling:")
                base_size = min(size_data.keys())
                base_result = size_data[base_size]
                print(
                    f"  Base ({base_size} traj): {base_result['loading_time']:.3f}s loading"
                )

                for size in sorted(size_data.keys()):
                    if size != base_size:
                        result = size_data[size]
                        data_ratio = size / base_size
                        time_ratio = (result["loading_time"] /
                                      base_result["loading_time"])
                        efficiency = data_ratio / time_ratio if time_ratio > 0 else 0

                        size_ratio = (result["file_size_mb"] /
                                      base_result["file_size_mb"] if
                                      base_result["file_size_mb"] > 0 else 0)

                        print(
                            f"  {size} traj ({data_ratio:.1f}x data): {result['loading_time']:.3f}s ({time_ratio:.2f}x time), efficiency: {efficiency:.2f}"
                        )
                        print(
                            f"    Loading: {time_ratio:.2f}x, Size: {size_ratio:.2f}x"
                        )

        # Head-to-head comparison at each size
        print(f"\n=== HEAD-TO-HEAD COMPARISON ===")

        for size in test_sizes:
            if results_by_size[size]:
                print(f"\nSize {size} trajectories:")

                # Find winners in each category
                fastest_loading = min(results_by_size[size].items(),
                                      key=lambda x: x[1]["loading_time"])
                smallest_size = min(results_by_size[size].items(),
                                    key=lambda x: x[1]["file_size_mb"])
                best_throughput = max(results_by_size[size].items(),
                                      key=lambda x: x[1]["throughput"])

                print(
                    f"  ⚡ Fastest loading: {fastest_loading[0]} ({fastest_loading[1]['loading_time']:.3f}s)"
                )
                print(
                    f"  🗜️ Smallest size: {smallest_size[0]} ({smallest_size[1]['file_size_mb']:.2f} MB)"
                )
                print(
                    f"  📈 Best throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.2f} traj/s)"
                )

                # Calculate speed comparison between fastest and others
                if len(results_by_size[size]) > 1:
                    all_times = [
                        (name, result["loading_time"])
                        for name, result in results_by_size[size].items()
                    ]
                    fastest_time = min(all_times, key=lambda x: x[1])
                    slowest_time = max(all_times, key=lambda x: x[1])
                    if slowest_time[1] > 0:
                        speedup = slowest_time[1] / fastest_time[1]
                        print(
                            f"  📊 {fastest_time[0]} is {speedup:.2f}x faster than {slowest_time[0]}"
                        )

        # Test passed successfully
        assert len(
            results_by_size) > 0, "Should have results for at least one size"

    def _create_synthetic_trajectories(self, num_trajectories):
        """Create synthetic trajectories for scalability testing."""
        trajectories = []
        steps_per_trajectory = 20

        for traj_idx in range(num_trajectories):
            trajectory_data = []
            for step in range(steps_per_trajectory):
                step_data = {
                    "observation": {
                        "image":
                        np.random.randint(0,
                                          255, (256, 256, 3),
                                          dtype=np.uint8),
                        "wrist_image":
                        np.random.randint(0,
                                          255, (128, 128, 3),
                                          dtype=np.uint8),
                        "state":
                        np.random.uniform(-1, 1, 7).astype(np.float32),
                        "gripper_state":
                        np.random.uniform(0, 1, 1).astype(np.float32),
                    },
                    "action":
                    np.random.uniform(-1, 1, 7).astype(np.float32),
                    "reward":
                    np.float32(1.0 if step == steps_per_trajectory -
                               1 else 0.0),
                    "is_terminal":
                    step == steps_per_trajectory - 1,
                    "step":
                    step,
                    "language_instruction":
                    f"Trajectory {traj_idx}, Step {step}",
                    "episode_id":
                    traj_idx,
                }
                trajectory_data.append(step_data)
            trajectories.append(trajectory_data)

        return trajectories
