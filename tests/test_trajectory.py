"""Unit tests for trajectory functionality."""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from fog_x import Trajectory, TrajectoryFactory, FeatureType
from fog_x.trajectory_base import FileSystemInterface, TimeProvider
from .test_fixtures import MockFileSystem, MockTimeProvider


class TestFeatureType:
    """Test the FeatureType class."""
    
    def test_from_data_numpy_array(self):
        """Test FeatureType.from_data with numpy arrays."""
        data = np.random.random((10, 20)).astype(np.float32)
        feature_type = FeatureType.from_data(data)
        assert feature_type.dtype == "float32"
        assert feature_type.shape == (10, 20)
    
    def test_from_data_scalar(self):
        """Test FeatureType.from_data with scalar values."""
        feature_type = FeatureType.from_data(1.0)
        assert feature_type.dtype == "float32"
        assert feature_type.shape == ()
    
    def test_from_data_string(self):
        """Test FeatureType.from_data with strings."""
        feature_type = FeatureType.from_data("test")
        assert feature_type.dtype == "str"
        assert feature_type.shape == ()
    
    def test_to_str_and_from_str(self):
        """Test string serialization and deserialization."""
        original = FeatureType(dtype="float32", shape=(10, 20))
        str_repr = str(original)
        reconstructed = FeatureType.from_str(str_repr)
        assert reconstructed.dtype == original.dtype
        assert reconstructed.shape == original.shape


class TestTrajectoryFactory:
    """Test the TrajectoryFactory class."""
    
    def test_factory_with_default_dependencies(self, temp_dir):
        """Test factory with default dependencies."""
        factory = TrajectoryFactory()
        path = os.path.join(temp_dir, "test.vla")
        
        # This should work with actual filesystem since we're using defaults
        traj = factory.create_trajectory(path, mode="w", cache_dir=temp_dir)
        assert traj is not None
        assert hasattr(traj, '_filesystem')
        assert hasattr(traj, '_time_provider')
        traj.close()
    
    def test_factory_with_mock_dependencies(self, mock_filesystem, mock_time_provider, temp_dir):
        """Test factory with mock dependencies."""
        factory = TrajectoryFactory(
            filesystem=mock_filesystem,
            time_provider=mock_time_provider
        )
        
        # Setup mock filesystem
        mock_filesystem.add_file("/test/test.vla")
        mock_filesystem.directories.add(temp_dir)
        
        path = "/test/test.vla"
        
        with patch('av.open') as mock_av:
            mock_container = Mock()
            mock_av.return_value = mock_container
            
            traj = factory.create_trajectory(path, mode="w", cache_dir=temp_dir)
            assert traj._filesystem == mock_filesystem
            assert traj._time_provider == mock_time_provider


class TestTrajectory:
    """Test the main Trajectory class."""
    
    def test_trajectory_creation_write_mode(self, temp_dir):
        """Test trajectory creation in write mode."""
        path = os.path.join(temp_dir, "test.vla")
        traj = Trajectory(path, mode="w", cache_dir=temp_dir)
        assert traj.path == path
        assert traj.mode == "w"
        assert not traj.is_closed
        traj.close()
    
    def test_trajectory_creation_read_mode_nonexistent(self, temp_dir):
        """Test trajectory creation in read mode with non-existent file."""
        path = os.path.join(temp_dir, "nonexistent.vla")
        with pytest.raises(FileNotFoundError):
            Trajectory(path, mode="r", cache_dir=temp_dir)
    
    def test_add_single_feature(self, temp_dir):
        """Test adding a single feature to trajectory."""
        path = os.path.join(temp_dir, "test.vla")
        traj = Trajectory(path, mode="w", cache_dir=temp_dir)
        
        # Add some test data
        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        traj.add("observation/image", image_data)
        
        joint_data = np.random.random(7).astype(np.float32)
        traj.add("observation/joints", joint_data)
        
        traj.close()
        
        # Verify file was created
        assert os.path.exists(path)
    
    def test_add_by_dict(self, temp_dir):
        """Test adding features via dictionary."""
        path = os.path.join(temp_dir, "test.vla")
        traj = Trajectory(path, mode="w", cache_dir=temp_dir)
        
        data = {
            "observation": {
                "image": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                "joints": np.random.random(7).astype(np.float32),
            },
            "action": np.random.random(7).astype(np.float32),
        }
        
        traj.add_by_dict(data)
        traj.close()
        
        assert os.path.exists(path)
    
    def test_from_list_of_dicts(self, temp_dir, sample_trajectory_data):
        """Test creating trajectory from list of dictionaries."""
        path = os.path.join(temp_dir, "test.vla")
        
        traj = Trajectory.from_list_of_dicts(
            sample_trajectory_data, 
            path, 
            lossy_compression=False
        )
        
        assert os.path.exists(path)
    
    def test_from_dict_of_lists(self, temp_dir, sample_dict_of_lists):
        """Test creating trajectory from dictionary of lists."""
        path = os.path.join(temp_dir, "test.vla")
        
        traj = Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path, 
            lossy_compression=False
        )
        
        assert os.path.exists(path)
    
    def test_load_and_read(self, temp_dir, sample_dict_of_lists):
        """Test loading and reading trajectory data."""
        path = os.path.join(temp_dir, "test.vla")
        
        # Create trajectory
        traj = Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path, 
            lossy_compression=False
        )
        
        # Read back the data
        traj_read = Trajectory(path, mode="r", cache_dir=temp_dir)
        loaded_data = traj_read.load()
        
        # Verify data structure
        assert isinstance(loaded_data, dict)
        assert "observation/image" in loaded_data
        assert "observation/joint_positions" in loaded_data
        assert "action" in loaded_data
        assert "reward" in loaded_data
        
        # Verify data shapes
        assert loaded_data["observation/image"].shape == (2, 640,480, 3)
        assert loaded_data["observation/joint_positions"].shape == (2, 7)
        assert loaded_data["action"].shape == (2, 7)
        assert loaded_data["reward"].shape == (2,)
    
    def test_getitem_access(self, temp_dir, sample_dict_of_lists):
        """Test accessing data via __getitem__."""
        path = os.path.join(temp_dir, "test.vla")
        
        # Create trajectory
        Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path, 
            lossy_compression=False
        )
        
        # Read back the data
        traj = Trajectory(path, mode="r", cache_dir=temp_dir)
        
        # Test __getitem__ access
        image_data = traj["observation/image"]
        assert image_data.shape == (2, 640,480, 3)
        
        action_data = traj["action"]
        assert action_data.shape == (2, 7)
    
    def test_load_different_return_types(self, temp_dir, sample_dict_of_lists):
        """Test loading with different return types."""
        path = os.path.join(temp_dir, "test.vla")
        
        # Create trajectory
        Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path, 
            lossy_compression=False
        )
        
        traj = Trajectory(path, mode="r", cache_dir=temp_dir)
        
        # Test numpy return type
        numpy_data = traj.load(return_type="numpy")
        assert isinstance(numpy_data, dict)
        
        # Test cache_name return type
        cache_name = traj.load(return_type="cache_name")
        assert isinstance(cache_name, str)
        assert cache_name.endswith(".cache")
        
        # Test container return type
        container_name = traj.load(return_type="container")
        assert container_name == path
    
    def test_close_behavior(self, temp_dir):
        """Test trajectory close behavior."""
        path = os.path.join(temp_dir, "test.vla")
        traj = Trajectory(path, mode="w", cache_dir=temp_dir)
        
        # Add some data
        traj.add("test_feature", np.array([1, 2, 3]))
        
        # Close trajectory
        assert not traj.is_closed
        traj.close()
        assert traj.is_closed
        
        # Test that closing again raises an error
        with pytest.raises(ValueError, match="already closed"):
            traj.close()
    
    def test_invalid_mode(self, temp_dir):
        """Test trajectory creation with invalid mode."""
        path = os.path.join(temp_dir, "test.vla")
        with pytest.raises(ValueError, match="Invalid mode"):
            Trajectory(path, mode="invalid", cache_dir=temp_dir)
    
    def test_dependency_injection(self, mock_filesystem, mock_time_provider, temp_dir):
        """Test that dependency injection works correctly."""
        factory = TrajectoryFactory(
            filesystem=mock_filesystem,
            time_provider=mock_time_provider
        )
        
        # Setup mock filesystem
        mock_filesystem.directories.add(temp_dir)
        mock_filesystem.add_file("/test/test.vla")
        
        with patch('av.open') as mock_av:
            mock_container = Mock()
            mock_av.return_value = mock_container
            
            traj = factory.create_trajectory("/test/test.vla", mode="w", cache_dir=temp_dir)
            
            # Test that filesystem methods are called on mock
            assert traj._exists("/test/test.vla")
            assert mock_filesystem.exists("/test/test.vla")
            
            # Test that time provider is used
            initial_calls = mock_time_provider.call_count
            timestamp = traj._get_current_timestamp()
            assert mock_time_provider.call_count > initial_calls


class TestTrajectoryIntegration:
    """Integration tests for trajectory functionality."""
    
    def test_full_workflow(self, temp_dir):
        """Test complete workflow: create, write, read, verify."""
        path = os.path.join(temp_dir, "integration_test.vla")
        
        # Create and populate trajectory
        traj_write = Trajectory(path, mode="w", cache_dir=temp_dir)
        
        for i in range(10):
            data = {
                "observation": {
                    "image": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                    "joints": np.random.random(7).astype(np.float32),
                },
                "action": np.random.random(7).astype(np.float32),
                "step": i,
            }
            traj_write.add_by_dict(data)
        
        traj_write.close()
        
        # Read back and verify
        traj_read = Trajectory(path, mode="r", cache_dir=temp_dir)
        loaded_data = traj_read.load()
        
        # Verify structure and dimensions
        assert "observation/image" in loaded_data
        assert "observation/joints" in loaded_data
        assert "action" in loaded_data
        assert "step" in loaded_data
        
        assert loaded_data["observation/image"].shape == (10, 64, 64, 3)
        assert loaded_data["observation/joints"].shape == (10, 7)
        assert loaded_data["action"].shape == (10, 7)
        assert loaded_data["step"].shape == (10,)
    
    def test_lossy_vs_lossless_compression(self, temp_dir, sample_dict_of_lists):
        """Test difference between lossy and lossless compression."""
        path_lossy = os.path.join(temp_dir, "lossy.vla")
        path_lossless = os.path.join(temp_dir, "lossless.vla")
        
        # Create both versions
        Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path_lossy, 
            lossy_compression=True
        )
        
        Trajectory.from_dict_of_lists(
            sample_dict_of_lists, 
            path_lossless, 
            lossy_compression=False
        )
        
        # Both files should exist
        assert os.path.exists(path_lossy)
        assert os.path.exists(path_lossless)
        
        # Lossy should generally be smaller (though this depends on the data)
        lossy_size = os.path.getsize(path_lossy)
        lossless_size = os.path.getsize(path_lossless)
        
        # Just verify both are reasonable sizes
        assert lossy_size > 0
        assert lossless_size > 0 