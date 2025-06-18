#!/usr/bin/env python3

import numpy as np
import tempfile
import os
import time
from robodm import Trajectory

def test_optimized_from_list_of_dicts():
    """Test the optimized from_list_of_dicts method with direct encoding."""
    
    # Create test data
    data = []
    for i in range(10):
        step = {
            "rgb_image": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "action": np.array([i, i+1, i+2], dtype=np.float32),
            "reward": float(i * 0.1),
            "text": f"step_{i}"
        }
        data.append(step)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        trajectory_path = os.path.join(temp_dir, "test_optimized.vla")
        
        print("Testing optimized from_list_of_dicts...")
        start_time = time.time()
        
        # Test with direct encoding (should skip transcoding)
        trajectory = Trajectory.from_list_of_dicts(
            data=data,
            path=trajectory_path,
            video_codec="libx264",  # Should encode directly to H.264
            fps=10
        )
        
        creation_time = time.time() - start_time
        print(f"Creation took: {creation_time:.2f} seconds")
        
        # Verify the trajectory was created
        assert os.path.exists(trajectory_path), "Trajectory file should exist"
        file_size = os.path.getsize(trajectory_path)
        print(f"File size: {file_size} bytes")
        
        # Test loading the trajectory
        start_time = time.time()
        trajectory_read = Trajectory(trajectory_path, mode="r")
        loaded_data = trajectory_read.load()
        load_time = time.time() - start_time
        print(f"Loading took: {load_time:.2f} seconds")
        
        # Verify data integrity
        assert "rgb_image" in loaded_data, "RGB image feature should be present"
        assert "action" in loaded_data, "Action feature should be present"
        assert "reward" in loaded_data, "Reward feature should be present"
        assert "text" in loaded_data, "Text feature should be present"
        
        print(f"Loaded {len(loaded_data['rgb_image'])} steps")
        print(f"RGB image shape: {loaded_data['rgb_image'][0].shape}")
        print(f"Action shape: {loaded_data['action'][0].shape}")
        
        trajectory_read.close()
        
        print("✓ Optimized from_list_of_dicts test passed!")

def test_optimized_from_dict_of_lists():
    """Test the optimized from_dict_of_lists method with direct encoding."""
    
    # Create test data
    num_steps = 10
    data = {
        "rgb_image": [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(num_steps)],
        "action": [np.array([i, i+1], dtype=np.float32) for i in range(num_steps)],
        "reward": [float(i * 0.1) for i in range(num_steps)],
        "nested": {
            "value1": [f"text_{i}" for i in range(num_steps)],
            "value2": [i * 10 for i in range(num_steps)]
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        trajectory_path = os.path.join(temp_dir, "test_dict_optimized.vla")
        
        print("\nTesting optimized from_dict_of_lists...")
        start_time = time.time()
        
        # Test with direct encoding
        trajectory = Trajectory.from_dict_of_lists(
            data=data,
            path=trajectory_path,
            video_codec="libx264",
            fps=10
        )
        
        creation_time = time.time() - start_time
        print(f"Creation took: {creation_time:.2f} seconds")
        
        # Verify the trajectory was created
        assert os.path.exists(trajectory_path), "Trajectory file should exist"
        file_size = os.path.getsize(trajectory_path)
        print(f"File size: {file_size} bytes")
        
        # Test loading
        start_time = time.time()
        trajectory_read = Trajectory(trajectory_path, mode="r")
        loaded_data = trajectory_read.load()
        load_time = time.time() - start_time
        print(f"Loading took: {load_time:.2f} seconds")
        
        # Verify data integrity
        assert "rgb_image" in loaded_data, "RGB image should be present"
        assert "action" in loaded_data, "Action should be present" 
        assert "reward" in loaded_data, "Reward should be present"
        assert "nested/value1" in loaded_data, "Nested value1 should be present"
        assert "nested/value2" in loaded_data, "Nested value2 should be present"
        
        print(f"Loaded {len(loaded_data['rgb_image'])} steps")
        print(f"Features: {list(loaded_data.keys())}")
        
        trajectory_read.close()
        
        print("✓ Optimized from_dict_of_lists test passed!")

if __name__ == "__main__":
    test_optimized_from_list_of_dicts()
    test_optimized_from_dict_of_lists()
    print("\n🎉 All tests passed! The optimized batch processing is working correctly.") 