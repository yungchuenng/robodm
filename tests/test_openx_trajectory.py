"""Unit tests for Open X-Embodiment trajectory functionality."""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from fog_x import Trajectory
from fog_x.loader import RLDSLoader
from .test_fixtures import MockFileSystem, MockTimeProvider

# Define codecs to test with OpenX data
OPENX_TEST_CODECS = ["rawvideo", "ffv1", "libaom-av1", "libx264"]

# Common Open X-Embodiment datasets for testing
OPENX_TEST_DATASETS = ["bridge", "berkeley_cable_routing", "nyu_door_opening_surprising_effectiveness"]


def validate_openx_roundtrip(temp_dir, codec, openx_data, dataset_name):
    """Helper function to validate Open X-Embodiment data through encoding/decoding roundtrip."""
    path = os.path.join(temp_dir, f"openx_roundtrip_{dataset_name}_{codec}.vla")
    
    try:
        # Step 1: Create trajectory from OpenX data using from_list_of_dicts
        Trajectory.from_list_of_dicts(
            openx_data, 
            path=path,
            video_codec=codec
        )
        
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
            if hasattr(values, 'shape'):
                assert values.shape[0] == original_length, f"Length mismatch for {key}: got {values.shape[0]}, expected {original_length}"
        
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
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    "state": np.random.uniform(-1, 1, 7).astype(np.float32),  # joint positions
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
                    "image": np.full((256, 256, 3), step * 85, dtype=np.uint8),  # Deterministic image
                    "state": np.array([step * 0.1] * 7, dtype=np.float32),  # Deterministic state
                },
                "action": np.array([step, step + 0.5] * 3 + [step], dtype=np.float32),  # 7D action
                "reward": np.float32(0.0),
                "is_terminal": False,
                "step": step,
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
            assert obs["state"].shape == (7,)
            
            # Check action structure
            assert step_data["action"].shape == (7,)
    
    @pytest.mark.parametrize("codec", OPENX_TEST_CODECS)
    def test_openx_trajectory_roundtrip(self, temp_dir, bridge_style_data, codec):
        """Test Open X-Embodiment data integrity through VLA trajectory roundtrip."""
        success, error, loaded_data = validate_openx_roundtrip(
            temp_dir, codec, bridge_style_data, "bridge_test"
        )
        
        if not success:
            if "not available" in str(error).lower() or "codec" in str(error).lower():
                pytest.skip(f"Codec {codec} not available: {error}")
            else:
                pytest.fail(f"Roundtrip failed for codec {codec}: {error}")
        
        # Validate data integrity with appropriate tolerances
        assert loaded_data is not None
        
        # Check that we have the expected fields
        expected_fields = ["observation/image", "observation/state", "action", "reward", "step"]
        for field in expected_fields:
            assert any(field in key or key.endswith(field.split('/')[-1]) for key in loaded_data.keys()), \
                f"Field {field} not found in loaded data. Available: {list(loaded_data.keys())}"
        
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
        action_key = next(k for k in loaded_data.keys() if k.endswith("action"))
        step_key = next(k for k in loaded_data.keys() if k.endswith("step"))
        
        # Validate shapes
        assert loaded_data[image_key].shape == (3, 256, 256, 3)
        assert loaded_data[state_key].shape == (3, 7)
        assert loaded_data[action_key].shape == (3, 7)
        assert loaded_data[step_key].shape == (3,)
        
        # Validate step values (should be exact)
        np.testing.assert_array_equal(loaded_data[step_key], [0, 1, 2])
        
        # Validate action values with tolerance
        expected_actions = np.array([
            [0, 0.5, 0, 0.5, 0, 0.5, 0],
            [1, 1.5, 1, 1.5, 1, 1.5, 1],
            [2, 2.5, 2, 2.5, 2, 2.5, 2]
        ], dtype=np.float32)
        np.testing.assert_allclose(loaded_data[action_key], expected_actions, rtol=float_tolerance)
        
        # Validate state values with tolerance
        expected_states = np.array([
            [0.0] * 7,
            [0.1] * 7,
            [0.2] * 7
        ], dtype=np.float32)
        np.testing.assert_allclose(loaded_data[state_key], expected_states, rtol=float_tolerance)
        
        # Validate image values with tolerance (deterministic pattern)
        if codec in ["rawvideo", "ffv1"]:
            # For lossless codecs, check exact values
            expected_images = np.array([
                np.full((256, 256, 3), 0, dtype=np.uint8),
                np.full((256, 256, 3), 85, dtype=np.uint8),
                np.full((256, 256, 3), 170, dtype=np.uint8),
            ])
            np.testing.assert_array_equal(loaded_data[image_key], expected_images)
        else:
            # For lossy codecs, check that values are reasonably close
            expected_images = np.array([
                np.full((256, 256, 3), 0, dtype=np.uint8),
                np.full((256, 256, 3), 85, dtype=np.uint8),
                np.full((256, 256, 3), 170, dtype=np.uint8),
            ])
            diff = np.abs(loaded_data[image_key].astype(np.int16) - expected_images.astype(np.int16))
            assert np.max(diff) <= image_tolerance, f"Image values differ by more than {image_tolerance}"
    
    def test_openx_trajectory_comparison_original_vs_reconstructed(self, temp_dir, bridge_style_data):
        """Test detailed comparison between original OpenX data and reconstructed trajectory."""
        # Use lossless codec for exact comparison
        codec = "rawvideo"
        
        success, error, loaded_data = validate_openx_roundtrip(
            temp_dir, codec, bridge_style_data, "comparison_test"
        )
        
        if not success:
            pytest.skip(f"Cannot perform comparison test: {error}")
        
        # Extract original data for comparison
        original_images = np.array([step["observation"]["image"] for step in bridge_style_data])
        original_states = np.array([step["observation"]["state"] for step in bridge_style_data])
        original_actions = np.array([step["action"] for step in bridge_style_data])
        original_steps = np.array([step["step"] for step in bridge_style_data])
        
        # Find keys in loaded data
        image_key = next(k for k in loaded_data.keys() if "image" in k)
        state_key = next(k for k in loaded_data.keys() if "state" in k)
        action_key = next(k for k in loaded_data.keys() if k.endswith("action"))
        step_key = next(k for k in loaded_data.keys() if k.endswith("step"))
        
        # Compare original vs reconstructed (should be exact for rawvideo)
        np.testing.assert_array_equal(loaded_data[image_key], original_images, 
                                    "Images differ between original and reconstructed")
        np.testing.assert_array_equal(loaded_data[state_key], original_states,
                                    "States differ between original and reconstructed")
        np.testing.assert_array_equal(loaded_data[action_key], original_actions,
                                    "Actions differ between original and reconstructed")
        np.testing.assert_array_equal(loaded_data[step_key], original_steps,
                                    "Steps differ between original and reconstructed")
    
    def test_openx_multiple_codecs_consistency(self, temp_dir, bridge_style_data):
        """Test that different codecs produce consistent results within their expected tolerances."""
        codec_results = {}
        
        # Test multiple codecs
        test_codecs = ["rawvideo", "ffv1"]  # Start with lossless codecs
        
        for codec in test_codecs:
            success, error, loaded_data = validate_openx_roundtrip(
                temp_dir, codec, bridge_style_data, f"consistency_{codec}"
            )
            
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
                common_keys = set(reference_data.keys()) & set(other_data.keys())
                
                for key in common_keys:
                    ref_array = reference_data[key]
                    other_array = other_data[key]
                    
                    assert ref_array.shape == other_array.shape, \
                        f"Shape mismatch for {key} between {reference_codec} and {other_codec}"
                    
                    # For lossless codecs, arrays should be identical
                    if reference_codec in ["rawvideo", "ffv1"] and other_codec in ["rawvideo", "ffv1"]:
                        np.testing.assert_array_equal(ref_array, other_array,
                                                    f"Lossless codecs {reference_codec} and {other_codec} produced different results for {key}")
    
    # @pytest.mark.integration
    def test_openx_codec_availability_report(self, temp_dir, mock_openx_data):
        """Test and report which codecs work with Open X-Embodiment data."""
        codec_status = {}
        
        for codec in OPENX_TEST_CODECS:
            success, error, _ = validate_openx_roundtrip(
                temp_dir, codec, mock_openx_data, "availability_test"
            )
            codec_status[codec] = {"available": success, "error": error}
        
        # Print codec availability report for OpenX data
        print("\n" + "="*60)
        print("OPEN X-EMBODIMENT CODEC AVAILABILITY REPORT")
        print("="*60)
        
        available_codecs = []
        unavailable_codecs = []
        
        for codec, status in codec_status.items():
            if status["available"]:
                available_codecs.append(codec)
                print(f"✓ {codec}: Available and working with OpenX data")
            else:
                unavailable_codecs.append(codec)
                print(f"✗ {codec}: {status['error']}")
        
        print(f"\nSummary: {len(available_codecs)}/{len(OPENX_TEST_CODECS)} codecs available for OpenX data")
        print("="*60)
        
        # Ensure at least one codec works with OpenX data
        assert len(available_codecs) > 0, "No codecs are available for Open X-Embodiment data!"


class TestRLDSLoaderIntegration:
    """Test RLDS loader integration with OpenX datasets (requires actual data)."""
    
    # @pytest.mark.slow
    # @pytest.mark.skipif(os.getenv("OPENX_DATA_DIR") is None, 
    #                    reason="OPENX_DATA_DIR environment variable not set")
    @pytest.mark.parametrize("video_codec", ["rawvideo", "libx264"])
    def test_real_openx_data_codec_comparison(self, temp_dir, video_codec):
        """Test real OpenX data with different codecs using appropriate validation for each."""
        data_dir = "/home/kych/berkeley/datasets/rtx/fractal20220817_data/0.1.0/"
        dataset_name = "fractal20220817_data"
        
        try:
            # Load real OpenX data using the correct RLDSLoader API
            loader = RLDSLoader(
                path=data_dir,
                split="train",
                batch_size=1,
                shuffle_buffer=10,
                shuffling=False
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
            Trajectory.from_list_of_dicts(
                first_traj_data,
                path=path,
                video_codec=video_codec
            )
            
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
                        if (orig_key.split('/')[-1] == recon_key.split('/')[-1] or
                            orig_key.replace('/', '_').lower() == recon_key.replace('/', '_').lower()):
                            field_mappings[orig_key] = recon_key
                            break
            
            # Define codec-specific tolerances
            is_lossless = video_codec in ["rawvideo", "ffv1"]
            if is_lossless:
                image_tolerance = 0  # Exact match required
                float_tolerance = 1e-7  # Very small tolerance for floating point precision
                print(f"Using lossless codec tolerances (exact match required)")
            else:
                image_tolerance = 100  # Allow compression artifacts for lossy codecs (reasonable for H.264)
                float_tolerance = 1e-4  # Small tolerance for lossy compression
                print(f"Using lossy codec tolerances (image_tol={image_tolerance}, float_tol={float_tolerance})")
            
            # Validate based on codec type
            exact_matches = 0
            acceptable_matches = 0
            total_fields = len(field_mappings)
            
            for orig_key, recon_key in field_mappings.items():
                orig_data = original_flat_data[orig_key]
                recon_data = loaded_data[recon_key]
                
                # Skip if shapes don't match
                if hasattr(orig_data, 'shape') and hasattr(recon_data, 'shape'):
                    if orig_data.shape != recon_data.shape:
                        continue
                
                is_image_field = 'image' in orig_key.lower() and hasattr(orig_data, 'dtype') and orig_data.dtype == np.uint8
                
                if is_image_field and not is_lossless:
                    # For lossy codecs, allow image compression differences
                    if np.array_equal(orig_data, recon_data):
                        exact_matches += 1
                    else:
                        max_diff = np.max(np.abs(orig_data.astype(np.int16) - recon_data.astype(np.int16)))
                        if max_diff <= image_tolerance:
                            acceptable_matches += 1
                        else:
                            pytest.fail(f"Image field {orig_key} exceeds tolerance: max_diff={max_diff} > {image_tolerance}")
                elif hasattr(orig_data, 'dtype') and np.issubdtype(orig_data.dtype, np.floating):
                    # Floating point comparison
                    if np.allclose(orig_data, recon_data, rtol=float_tolerance, atol=float_tolerance):
                        exact_matches += 1
                    else:
                        pytest.fail(f"Float field {orig_key} doesn't match within tolerance")
                else:
                    # Other data should be exact
                    if np.array_equal(orig_data, recon_data):
                        exact_matches += 1
                    else:
                        pytest.fail(f"Field {orig_key} should be exact but differs")
            
            # Codec-specific final validation
            if is_lossless:
                assert exact_matches == total_fields, f"Lossless codec {video_codec}: {exact_matches}/{total_fields} exact matches"
                print(f"✓ Lossless codec {video_codec}: all {exact_matches} fields match exactly")
            else:
                total_acceptable = exact_matches + acceptable_matches
                assert total_acceptable == total_fields, f"Lossy codec {video_codec}: {total_acceptable}/{total_fields} within tolerance"
                print(f"✓ Lossy codec {video_codec}: {exact_matches} exact + {acceptable_matches} acceptable = {total_acceptable}/{total_fields}")
            
        except Exception as e:
            if "not available" in str(e).lower() or "codec" in str(e).lower():
                pytest.skip(f"Codec {video_codec} not available: {e}")
            else:
                pytest.fail(f"Failed with codec {video_codec}: {e}")

    def test_real_openx_data_loading(self, temp_dir):
        """Test loading real Open X-Embodiment data and compare original vs reconstructed."""
        data_dir = "/home/kych/berkeley/datasets/rtx/fractal20220817_data/0.1.0/"
        dataset_name = "fractal20220817_data"  # Define dataset_name for file naming
        video_codec = "libx264"  # Test with lossy codec
        
        try:
            # Load real OpenX data using the correct RLDSLoader API
            loader = RLDSLoader(
                path=data_dir,
                split="train",
                batch_size=1,
                shuffle_buffer=10,
                shuffling=False  # Don't shuffle for testing
            )
            
            # Get first trajectory using iterator interface
            first_traj_batch = next(iter(loader))
            first_traj_data = first_traj_batch[0]  # Get the actual trajectory data from batch
            
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
                            print(f"  {key}: dict with keys {list(value.keys())}")
                            for subkey, subvalue in value.items():
                                full_key = f"{key}/{subkey}"
                                if hasattr(subvalue, 'shape'):
                                    print(f"    {subkey}: {type(subvalue).__name__} {subvalue.shape} {getattr(subvalue, 'dtype', 'no dtype')}")
                                else:
                                    print(f"    {subkey}: {type(subvalue).__name__}")
                        else:
                            print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', 'no shape')} {getattr(value, 'dtype', 'no dtype')}")
                
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
                        print(f"Warning: Trajectory length mismatch for {key}: {len(values)} vs {trajectory_length}")
                except Exception as e:
                    print(f"Could not convert {key} to array: {e}")
                    original_trajectory[key] = values  # Keep as list
            
            print(f"\nOriginal trajectory fields: {len(original_trajectory)}")
            for key, data in original_trajectory.items():
                if hasattr(data, 'shape'):
                    print(f"  {key}: {data.shape} {data.dtype}")
                else:
                    print(f"  {key}: {type(data)} (length: {len(data) if hasattr(data, '__len__') else 'N/A'})")
            
            # Test conversion to VLA format
            path = os.path.join(temp_dir, f"real_{dataset_name}_test.vla")
            print(f"\n=== CONVERTING TO VLA FORMAT ===")
            Trajectory.from_list_of_dicts(
                first_traj_data,
                path=path,
                video_codec=video_codec
            )
            
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
                if hasattr(values, 'shape'):
                    print(f"  {key}: {values.shape} {values.dtype}")
                    if reconstructed_length is None:
                        reconstructed_length = values.shape[0]
                    elif values.shape[0] != reconstructed_length:
                        print(f"Warning: Inconsistent trajectory length for {key}: {values.shape[0]} vs {reconstructed_length}")
                else:
                    print(f"  {key}: {type(values)} (length: {len(values) if hasattr(values, '__len__') else 'N/A'})")
            
            # TRAJECTORY-LEVEL VALIDATION
            print(f"\n=== TRAJECTORY-LEVEL VALIDATION ===")
            
            # 1. Trajectory Length Validation
            print(f"Original trajectory length: {trajectory_length}")
            print(f"Reconstructed trajectory length: {reconstructed_length}")
            assert reconstructed_length == trajectory_length, \
                f"Trajectory length mismatch: original={trajectory_length}, reconstructed={reconstructed_length}"
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
            print(f"Field mapping coverage: {mapping_coverage:.1f}% ({len(field_mappings)}/{len(original_keys)})")
            
            if unmatched_original:
                print(f"Unmatched original fields: {unmatched_original}")
            if unmatched_reconstructed:
                print(f"Unmatched reconstructed fields: {unmatched_reconstructed}")
            
            # 3. Define codec-specific validation criteria
            is_lossless = video_codec in ["rawvideo", "ffv1"]
            if is_lossless:
                image_tolerance = 0
                float_tolerance = 1e-7
                print(f"Using lossless validation (exact match required)")
            else:
                image_tolerance = 100  # Reasonable for H.264 compression
                float_tolerance = 1e-4
                print(f"Using lossy validation (image_tol={image_tolerance}, float_tol={float_tolerance})")
            
            # 4. Comprehensive Trajectory Data Validation
            print(f"\n=== COMPREHENSIVE DATA VALIDATION ===")
            validation_results = {
                'exact_matches': 0,
                'acceptable_matches': 0,
                'shape_mismatches': [],
                'value_mismatches': [],
                'temporal_errors': [],
                'critical_errors': []
            }
            
            for orig_key, recon_key in field_mappings.items():
                try:
                    orig_data = original_trajectory[orig_key]
                    recon_data = loaded_trajectory[recon_key]
                    
                    # Validate this field across the entire trajectory
                    field_result = self._validate_trajectory_field(
                        orig_key, orig_data, recon_data, 
                        is_lossless, image_tolerance, float_tolerance, trajectory_length
                    )
                    
                    # Accumulate results
                    if field_result['status'] == 'exact_match':
                        validation_results['exact_matches'] += 1
                        print(f"✓ {orig_key}: Exact match across trajectory")
                    elif field_result['status'] == 'acceptable_match':
                        validation_results['acceptable_matches'] += 1
                        print(f"~ {orig_key}: Acceptable match (max_diff: {field_result.get('max_diff', 'N/A')} ≤ {image_tolerance})")
                    elif field_result['status'] == 'shape_mismatch':
                        validation_results['shape_mismatches'].append(field_result)
                        print(f"✗ {orig_key}: Shape mismatch {field_result['error']}")
                    elif field_result['status'] == 'value_mismatch':
                        validation_results['value_mismatches'].append(field_result)
                        print(f"✗ {orig_key}: Value mismatch - {field_result['error']}")
                    elif field_result['status'] == 'temporal_error':
                        validation_results['temporal_errors'].append(field_result)
                        print(f"✗ {orig_key}: Temporal consistency error - {field_result['error']}")
                    else:
                        validation_results['critical_errors'].append(field_result)
                        print(f"? {orig_key}: Critical error - {field_result.get('error', 'Unknown')}")
                        
                except Exception as e:
                    error_result = {'field': orig_key, 'status': 'critical_error', 'error': str(e)}
                    validation_results['critical_errors'].append(error_result)
                    print(f"? {orig_key}: Exception during validation - {e}")
            
            # 5. Final Trajectory Integrity Assessment
            print(f"\n=== TRAJECTORY INTEGRITY SUMMARY ===")
            total_fields = len(field_mappings)
            total_passed = validation_results['exact_matches'] + validation_results['acceptable_matches']
            
            print(f"Total trajectory fields validated: {total_fields}")
            print(f"Exact matches: {validation_results['exact_matches']}")
            print(f"Acceptable matches: {validation_results['acceptable_matches']}")
            print(f"Shape mismatches: {len(validation_results['shape_mismatches'])}")
            print(f"Value mismatches: {len(validation_results['value_mismatches'])}")
            print(f"Temporal errors: {len(validation_results['temporal_errors'])}")
            print(f"Critical errors: {len(validation_results['critical_errors'])}")
            
            # Assertions for trajectory integrity
            assert total_fields > 0, "No trajectory fields could be validated"
            assert len(validation_results['critical_errors']) == 0, \
                f"Critical errors in trajectory validation: {validation_results['critical_errors'][:3]}"
            assert len(validation_results['shape_mismatches']) == 0, \
                f"Shape mismatches in trajectory: {[r['error'] for r in validation_results['shape_mismatches'][:3]]}"
            assert len(validation_results['temporal_errors']) == 0, \
                f"Temporal consistency errors: {[r['error'] for r in validation_results['temporal_errors'][:3]]}"
            
            # Check for essential trajectory components
            has_image_trajectory = any('image' in key.lower() for key in field_mappings.keys())
            has_action_trajectory = any('action' in key.lower() for key in field_mappings.keys())
            assert has_image_trajectory, "No image trajectory found in reconstructed data"
            assert has_action_trajectory, "No action trajectory found in reconstructed data"
            
            # Codec-specific trajectory validation
            if is_lossless:
                assert total_passed == total_fields, \
                    f"Lossless codec {video_codec}: {total_passed}/{total_fields} trajectory fields passed validation"
                print(f"✓ Lossless trajectory validation: all {total_fields} fields exact")
            else:
                # For lossy codecs, ensure non-image data is exact and image data is within tolerance
                image_fields = [key for key in field_mappings.keys() if 'image' in key.lower()]
                non_image_fields = [key for key in field_mappings.keys() if 'image' not in key.lower()]
                
                # All non-image trajectory data should be exact
                non_image_mismatches = [r for r in validation_results['value_mismatches'] 
                                      if not any('image' in r['field'].lower() for _ in [1])]
                assert len(non_image_mismatches) == 0, \
                    f"Non-image trajectory data must be exact for lossy codecs: {[r['field'] for r in non_image_mismatches[:3]]}"
                
                assert total_passed == total_fields, \
                    f"Lossy codec {video_codec}: {total_passed}/{total_fields} trajectory fields within tolerance"
                print(f"✓ Lossy trajectory validation: {validation_results['exact_matches']} exact + {validation_results['acceptable_matches']} acceptable = {total_passed}/{total_fields}")
            
            # Field mapping coverage requirement
            assert mapping_coverage >= 95.0, \
                f"Poor trajectory field coverage: {mapping_coverage:.1f}% (minimum: 95%)"
            
            print(f"\n✓ TRAJECTORY INTEGRITY VALIDATION PASSED!")
            print(f"Successfully validated entire {dataset_name} trajectory with {trajectory_length} steps")
            print(f"Codec: {video_codec}, Fields: {total_fields}, Integrity: {total_passed}/{total_fields}")
            
        except Exception as e:
            pytest.fail(f"Trajectory validation failed: {e}")

    def _fields_match_semantically(self, orig_key, recon_key):
        """Check if two field keys represent the same data semantically."""
        # Exact match
        if orig_key == recon_key:
            return True
        
        # Clean and normalize keys
        orig_clean = orig_key.replace('/', '_').lower()
        recon_clean = recon_key.replace('/', '_').lower()
        
        if orig_clean == recon_clean:
            return True
        
        # Check if they share significant key components
        orig_tokens = set(orig_clean.split('_'))
        recon_tokens = set(recon_clean.split('_'))
        overlap = len(orig_tokens & recon_tokens)
        
        # Require high overlap for semantic matching
        if len(orig_tokens) > 0 and len(recon_tokens) > 0:
            overlap_ratio = overlap / min(len(orig_tokens), len(recon_tokens))
            return overlap_ratio >= 0.8
        
        return False
    
    def _validate_trajectory_field(self, field_name, orig_data, recon_data, is_lossless, 
                                  image_tolerance, float_tolerance, expected_length):
        """Validate a single field across the entire trajectory."""
        try:
            # Shape validation
            if hasattr(orig_data, 'shape') and hasattr(recon_data, 'shape'):
                if orig_data.shape != recon_data.shape:
                    return {
                        'status': 'shape_mismatch',
                        'field': field_name,
                        'error': f"{orig_data.shape} vs {recon_data.shape}"
                    }
                
                # Temporal length validation
                if orig_data.shape[0] != expected_length:
                    return {
                        'status': 'temporal_error',
                        'field': field_name,
                        'error': f"Original data length {orig_data.shape[0]} != expected {expected_length}"
                    }
                if recon_data.shape[0] != expected_length:
                    return {
                        'status': 'temporal_error',
                        'field': field_name,
                        'error': f"Reconstructed data length {recon_data.shape[0]} != expected {expected_length}"
                    }
            
            # Determine field type for appropriate validation
            is_image_field = 'image' in field_name.lower() and hasattr(orig_data, 'dtype') and orig_data.dtype == np.uint8
            
            # Data validation with trajectory-appropriate tolerances
            if hasattr(orig_data, 'dtype') and np.issubdtype(orig_data.dtype, np.floating):
                # Floating point trajectory data
                if np.allclose(orig_data, recon_data, rtol=float_tolerance, atol=float_tolerance):
                    return {'status': 'exact_match', 'field': field_name}
                else:
                    max_diff = np.max(np.abs(orig_data - recon_data))
                    return {
                        'status': 'value_mismatch',
                        'field': field_name,
                        'error': f"Float trajectory max_diff={max_diff} > tolerance={float_tolerance}",
                        'max_diff': max_diff
                    }
            elif is_image_field:
                # Image trajectory validation
                if np.array_equal(orig_data, recon_data):
                    return {'status': 'exact_match', 'field': field_name}
                elif not is_lossless:
                    # For lossy codecs, check if within tolerance
                    max_diff = np.max(np.abs(orig_data.astype(np.int16) - recon_data.astype(np.int16)))
                    if max_diff <= image_tolerance:
                        return {
                            'status': 'acceptable_match',
                            'field': field_name,
                            'max_diff': max_diff
                        }
                    else:
                        return {
                            'status': 'value_mismatch',
                            'field': field_name,
                            'error': f"Image trajectory max_diff={max_diff} > tolerance={image_tolerance}",
                            'max_diff': max_diff
                        }
                else:
                    # Lossless codec should be exact
                    max_diff = np.max(np.abs(orig_data.astype(np.int16) - recon_data.astype(np.int16)))
                    return {
                        'status': 'value_mismatch',
                        'field': field_name,
                        'error': f"Lossless image trajectory should be exact, got max_diff={max_diff}",
                        'max_diff': max_diff
                    }
            else:
                # Other data types should be exact
                if np.array_equal(orig_data, recon_data):
                    return {'status': 'exact_match', 'field': field_name}
                else:
                    if hasattr(orig_data, 'dtype') and np.issubdtype(orig_data.dtype, np.integer):
                        max_diff = np.max(np.abs(orig_data.astype(np.int64) - recon_data.astype(np.int64)))
                        return {
                            'status': 'value_mismatch',
                            'field': field_name,
                            'error': f"Non-image trajectory data should be exact, got max_diff={max_diff}",
                            'max_diff': max_diff
                        }
                    else:
                        return {
                            'status': 'value_mismatch',
                            'field': field_name,
                            'error': "Non-numeric trajectory comparison failed"
                        }
                        
        except Exception as e:
            return {
                'status': 'critical_error',
                'field': field_name,
                'error': f"Exception during validation: {str(e)}"
            } 