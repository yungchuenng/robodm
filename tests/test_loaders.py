"""Unit tests for loader functionality."""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from fog_x import Trajectory
from fog_x.loader import NonShuffleVLALoader, HDF5Loader
from .test_fixtures import BenchmarkDataset


class TestNonShuffleVLALoader:
    """Test the VLA loader."""
    
    def test_vla_loader_basic(self, temp_dir, large_sample_data):
        """Test basic VLA loader functionality."""
        # Create some VLA files
        paths = []
        for i in range(3):
            path = os.path.join(temp_dir, f"test_{i}.vla")
            # Create smaller datasets for faster testing
            small_data = {k: v[:5] for k, v in large_sample_data.items()}
            Trajectory.from_dict_of_lists(small_data, path, lossy_compression=False)
            paths.append(path)
        
        # Test loading
        pattern = os.path.join(temp_dir, "*.vla")
        loader = NonShuffleVLALoader(pattern)
        
        # Test iteration
        trajectories = list(loader)
        assert len(trajectories) == 3
        
        for traj in trajectories:
            assert isinstance(traj, dict)
            assert "observation/image" in traj
            assert "action" in traj
            assert traj["observation/image"].shape == (5, 480, 640, 3)
    
    def test_vla_loader_batch_size(self, temp_dir, large_sample_data):
        """Test VLA loader with different batch sizes."""
        # Create VLA file
        path = os.path.join(temp_dir, "test.vla")
        small_data = {k: v[:10] for k, v in large_sample_data.items()}
        Trajectory.from_dict_of_lists(small_data, path, lossy_compression=False)
        
        # Test with batch size
        from fog_x.loader.vla import get_vla_dataloader
        dataloader = get_vla_dataloader(
            path=temp_dir,
            batch_size=2
        )
        
        batches = list(dataloader)
        assert len(batches) > 0
        
        # Each batch should contain multiple trajectories
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) <= 2  # batch size


class TestHDF5Loader:
    """Test the HDF5 loader."""
    
    def test_hdf5_loader_basic(self, temp_dir, large_sample_data, benchmark_dataset):
        """Test basic HDF5 loader functionality."""
        # Create HDF5 files
        paths = []
        for i in range(3):
            path = os.path.join(temp_dir, f"test_{i}.h5")
            small_data = {k: v[:5] for k, v in large_sample_data.items()}
            benchmark_dataset.create_hdf5_dataset(path, small_data)
            paths.append(path)
        
        # Test loading
        from fog_x.loader.hdf5 import get_hdf5_dataloader
        dataloader = get_hdf5_dataloader(
            path=os.path.join(temp_dir, "*.h5"),
            batch_size=1,
            num_workers=0
        )
        
        # Test iteration
        batches = list(dataloader)
        assert len(batches) == 3
        
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) == 1
            traj = batch[0]
            assert isinstance(traj, dict)
            assert "observation/image" in traj
            assert "action" in traj
    
    def test_hdf5_loader_batch_size(self, temp_dir, large_sample_data, benchmark_dataset):
        """Test HDF5 loader with different batch sizes."""
        # Create multiple HDF5 files
        for i in range(5):
            path = os.path.join(temp_dir, f"test_{i}.h5")
            small_data = {k: v[:3] for k, v in large_sample_data.items()}
            benchmark_dataset.create_hdf5_dataset(path, small_data)
        
        # Test with batch size
        from fog_x.loader.hdf5 import get_hdf5_dataloader
        dataloader = get_hdf5_dataloader(
            path=os.path.join(temp_dir, "*.h5"),
            batch_size=2,
            num_workers=0
        )
        
        batches = list(dataloader)
        
        # Should have ceil(5/2) = 3 batches
        assert len(batches) == 3
        
        # First two batches should have 2 items
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        # Last batch should have 1 item
        assert len(batches[2]) == 1


class TestLoaderComparison:
    """Test comparisons between different loaders."""
    
    def test_vla_vs_hdf5_data_consistency(self, temp_dir, sample_dict_of_lists, benchmark_dataset):
        """Test that VLA and HDF5 loaders return consistent data."""
        # Create both VLA and HDF5 from same source data
        vla_path = os.path.join(temp_dir, "test.vla")
        h5_path = os.path.join(temp_dir, "test.h5")
        
        # Use deterministic data for comparison
        deterministic_data = {
            "observation/image": [
                np.ones((32, 32, 3), dtype=np.uint8) * i for i in range(3)
            ],
            "action": [
                np.ones(4, dtype=np.float32) * i for i in range(3)
            ],
        }
        
        # Create both formats
        Trajectory.from_dict_of_lists(deterministic_data, vla_path, lossy_compression=False)
        benchmark_dataset.create_hdf5_dataset(h5_path, deterministic_data)
        
        # Load via both loaders
        vla_loader = NonShuffleVLALoader(vla_path)
        vla_data = list(vla_loader)[0]
        
        from fog_x.loader.hdf5 import get_hdf5_dataloader
        h5_loader = get_hdf5_dataloader(h5_path, batch_size=1, num_workers=0)
        h5_data = list(h5_loader)[0][0]
        
        # Compare data (allowing for small differences due to compression/encoding)
        assert vla_data.keys() == h5_data.keys()
        
        for key in vla_data.keys():
            vla_array = vla_data[key]
            h5_array = h5_data[key]
            
            assert vla_array.shape == h5_array.shape
            assert vla_array.dtype == h5_array.dtype
            
            # For lossless VLA, the data should be very close
            if vla_array.dtype == np.uint8:
                # Allow small differences for image data
                diff = np.abs(vla_array.astype(np.float32) - h5_array.astype(np.float32))
                assert np.mean(diff) < 1.0
            else:
                # Float data should be identical for lossless
                np.testing.assert_allclose(vla_array, h5_array, rtol=1e-5)


class TestLoaderError:
    """Test error handling in loaders."""
    
    def test_vla_loader_empty_pattern(self, temp_dir):
        """Test VLA loader with pattern that matches no files."""
        pattern = os.path.join(temp_dir, "nonexistent_*.vla")
        loader = NonShuffleVLALoader(pattern)
        
        # Should handle empty results gracefully
        trajectories = list(loader)
        assert len(trajectories) == 0
    
    def test_hdf5_loader_empty_pattern(self, temp_dir):
        """Test HDF5 loader with pattern that matches no files."""
        pattern = os.path.join(temp_dir, "nonexistent_*.h5")
        
        from fog_x.loader.hdf5 import get_hdf5_dataloader
        dataloader = get_hdf5_dataloader(pattern, batch_size=1, num_workers=0)
        
        # Should handle empty results gracefully
        batches = list(dataloader)
        assert len(batches) == 0
    
    def test_vla_loader_corrupted_file(self, temp_dir):
        """Test VLA loader behavior with corrupted files."""
        # Create a fake VLA file
        fake_path = os.path.join(temp_dir, "fake.vla")
        with open(fake_path, "w") as f:
            f.write("This is not a valid VLA file")
        
        loader = NonShuffleVLALoader(fake_path)
        
        # Should handle corrupted files gracefully
        with pytest.raises(Exception):
            list(loader)


class TestLoaderPerformance:
    """Basic performance tests for loaders."""
    
    def test_vla_loader_memory_usage(self, temp_dir, large_sample_data):
        """Test VLA loader memory efficiency."""
        # Create VLA file
        path = os.path.join(temp_dir, "large_test.vla")
        Trajectory.from_dict_of_lists(large_sample_data, path, lossy_compression=False)
        
        # Load and measure (basic test - would need memory profiling for real measurement)
        loader = NonShuffleVLALoader(path)
        trajectories = list(loader)
        
        assert len(trajectories) == 1
        assert "observation/image" in trajectories[0]
        assert trajectories[0]["observation/image"].shape == (100, 480, 640, 3)
    
    def test_hdf5_loader_memory_usage(self, temp_dir, large_sample_data, benchmark_dataset):
        """Test HDF5 loader memory efficiency."""
        # Create HDF5 file
        path = os.path.join(temp_dir, "large_test.h5")
        benchmark_dataset.create_hdf5_dataset(path, large_sample_data)
        
        # Load and measure
        from fog_x.loader.hdf5 import get_hdf5_dataloader
        dataloader = get_hdf5_dataloader(path, batch_size=1, num_workers=0)
        batches = list(dataloader)
        
        assert len(batches) == 1
        traj = batches[0][0]
        assert "observation/image" in traj
        assert traj["observation/image"].shape == (100, 480, 640, 3) 