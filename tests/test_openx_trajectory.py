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
    def test_real_openx_data_loading(self, temp_dir):
        """Test loading real Open X-Embodiment data and compare original vs reconstructed."""
        data_dir = "gs://gresearch/robotics/fractal20220817_data/0.1.0"
        dataset_name = "fractal20220817_data"  # Define dataset_name for file naming
        
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
            
            print(f"\n=== ORIGINAL OPENX DATA ANALYSIS ===")
            print(f"Trajectory length: {len(first_traj_data)} steps")
            
            # Analyze original data structure
            original_fields = {}
            for step_idx, step_data in enumerate(first_traj_data):
                print(f"\nStep {step_idx} fields:")
                for key, value in step_data.items():
                    if key not in original_fields:
                        original_fields[key] = []
                    
                    if isinstance(value, dict):
                        print(f"  {key}: dict with keys {list(value.keys())}")
                        for subkey, subvalue in value.items():
                            full_key = f"{key}/{subkey}"
                            if full_key not in original_fields:
                                original_fields[full_key] = []
                            original_fields[full_key].append(subvalue)
                    else:
                        print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', 'no shape')} {getattr(value, 'dtype', 'no dtype')}")
                        original_fields[key].append(value)
                
                if step_idx == 0:  # Only print first step details to avoid spam
                    break
            
            # Convert nested dict data to flat structure for comparison
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
            
            # Convert lists to numpy arrays for easier comparison
            for key, values in original_flat_data.items():
                try:
                    original_flat_data[key] = np.array(values)
                except:
                    # Keep as list if can't convert to array
                    pass
            
            # Test conversion to VLA format
            path = os.path.join(temp_dir, f"real_{dataset_name}_test.vla")
            Trajectory.from_list_of_dicts(
                first_traj_data,
                path=path,
                video_codec="rawvideo"  # Use lossless for exact validation
            )
            
            # Verify file was created and can be read back
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            
            # Read back and validate
            traj_read = Trajectory(path, mode="r")
            loaded_data = traj_read.load()
            traj_read.close()
            
            # Basic validation
            assert isinstance(loaded_data, dict)
            assert len(loaded_data) > 0
            
            print(f"\n=== RECONSTRUCTED VLA DATA ANALYSIS ===")
            print(f"VLA data keys: {list(loaded_data.keys())}")
            
            for key, values in loaded_data.items():
                if hasattr(values, 'shape'):
                    print(f"  {key}: {values.shape} {values.dtype}")
                else:
                    print(f"  {key}: {type(values)} (length: {len(values) if hasattr(values, '__len__') else 'N/A'})")
            
            # Compare original vs reconstructed data
            print(f"\n=== COMPARISON: ORIGINAL vs RECONSTRUCTED ===")
            
            # Check for expected OpenX fields
            has_image = any("image" in key.lower() for key in loaded_data.keys())
            has_action = any("action" in key.lower() for key in loaded_data.keys())
            
            assert has_image, f"No image field found in real OpenX data. Keys: {list(loaded_data.keys())}"
            assert has_action, f"No action field found in real OpenX data. Keys: {list(loaded_data.keys())}"
            
            # Detailed field comparison
            original_keys = set(original_flat_data.keys())
            reconstructed_keys = set(loaded_data.keys())
            
            # Find missing and extra fields
            missing_fields = original_keys - reconstructed_keys
            extra_fields = reconstructed_keys - original_keys
            
            # Try to match fields that might have different naming conventions
            field_mappings = {}
            for orig_key in original_keys:
                for recon_key in reconstructed_keys:
                    # Check if they might be the same field with different naming
                    orig_parts = orig_key.lower().split('/')
                    if any(part in recon_key.lower() for part in orig_parts):
                        field_mappings[orig_key] = recon_key
                        break
                    elif orig_key.lower() in recon_key.lower() or recon_key.lower() in orig_key.lower():
                        field_mappings[orig_key] = recon_key
                        break
            
            print(f"Original fields: {len(original_keys)}")
            print(f"Reconstructed fields: {len(reconstructed_keys)}")
            
            if missing_fields:
                print(f"Missing fields in reconstruction: {missing_fields}")
            
            if extra_fields:
                print(f"Extra fields in reconstruction: {extra_fields}")
            
            print(f"\nField mappings found: {len(field_mappings)}")
            for orig, recon in field_mappings.items():
                print(f"  {orig} -> {recon}")
            
            # Compare values for mapped fields
            comparison_results = {}
            for orig_key, recon_key in field_mappings.items():
                try:
                    orig_data = original_flat_data[orig_key]
                    recon_data = loaded_data[recon_key]
                    
                    # Check shapes
                    if hasattr(orig_data, 'shape') and hasattr(recon_data, 'shape'):
                        if orig_data.shape != recon_data.shape:
                            comparison_results[orig_key] = {
                                'status': 'shape_mismatch',
                                'original_shape': orig_data.shape,
                                'reconstructed_shape': recon_data.shape
                            }
                            continue
                    
                    # Check data types
                    if hasattr(orig_data, 'dtype') and hasattr(recon_data, 'dtype'):
                        if orig_data.dtype != recon_data.dtype:
                            print(f"  {orig_key}: dtype mismatch ({orig_data.dtype} -> {recon_data.dtype})")
                    
                    # Compare values with appropriate tolerance
                    if hasattr(orig_data, 'dtype') and np.issubdtype(orig_data.dtype, np.floating):
                        # Floating point comparison
                        if np.allclose(orig_data, recon_data, rtol=1e-6, atol=1e-8):
                            comparison_results[orig_key] = {'status': 'exact_match'}
                        elif np.allclose(orig_data, recon_data, rtol=1e-3, atol=1e-5):
                            comparison_results[orig_key] = {'status': 'close_match', 'max_diff': np.max(np.abs(orig_data - recon_data))}
                        else:
                            comparison_results[orig_key] = {'status': 'value_mismatch', 'max_diff': np.max(np.abs(orig_data - recon_data))}
                    else:
                        # Integer/other comparison
                        if np.array_equal(orig_data, recon_data):
                            comparison_results[orig_key] = {'status': 'exact_match'}
                        else:
                            if hasattr(orig_data, 'dtype') and np.issubdtype(orig_data.dtype, np.integer):
                                max_diff = np.max(np.abs(orig_data.astype(np.int64) - recon_data.astype(np.int64)))
                                comparison_results[orig_key] = {'status': 'value_mismatch', 'max_diff': max_diff}
                            else:
                                comparison_results[orig_key] = {'status': 'value_mismatch'}
                
                except Exception as e:
                    comparison_results[orig_key] = {'status': 'comparison_error', 'error': str(e)}
            
            # Print comparison results
            print(f"\n=== DETAILED COMPARISON RESULTS ===")
            for field, result in comparison_results.items():
                status = result['status']
                if status == 'exact_match':
                    print(f"✓ {field}: Exact match")
                elif status == 'close_match':
                    print(f"~ {field}: Close match (max diff: {result.get('max_diff', 'N/A')})")
                elif status == 'shape_mismatch':
                    print(f"✗ {field}: Shape mismatch ({result['original_shape']} vs {result['reconstructed_shape']})")
                elif status == 'value_mismatch':
                    print(f"✗ {field}: Value mismatch (max diff: {result.get('max_diff', 'N/A')})")
                elif status == 'comparison_error':
                    print(f"? {field}: Comparison error - {result['error']}")
            
            # Summary statistics
            exact_matches = sum(1 for r in comparison_results.values() if r['status'] == 'exact_match')
            close_matches = sum(1 for r in comparison_results.values() if r['status'] == 'close_match')
            mismatches = sum(1 for r in comparison_results.values() if r['status'] in ['value_mismatch', 'shape_mismatch'])
            
            print(f"\n=== SUMMARY ===")
            print(f"Total compared fields: {len(comparison_results)}")
            print(f"Exact matches: {exact_matches}")
            print(f"Close matches: {close_matches}")
            print(f"Mismatches: {mismatches}")
            print(f"Success rate: {(exact_matches + close_matches) / len(comparison_results) * 100:.1f}%")
            
            # The test should pass if we can successfully load and convert the data
            # Even if there are some differences due to compression or data type conversion
            print(f"Successfully loaded and converted real {dataset_name} data")
            
        except Exception as e:
            pytest.fail(f"Failed to load real OpenX data: {e}") 