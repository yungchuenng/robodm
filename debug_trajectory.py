#!/usr/bin/env python3

import tempfile
import numpy as np
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from fog_x import Trajectory

def debug_trajectory_creation():
    print("=== DEBUG: Testing trajectory creation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "debug.vla")
        print(f"Creating trajectory at: {path}")
        
        # Simple test data
        data = {
            "observation/image": [
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),
            ],
            "action": [
                np.random.random(7).astype(np.float32),
                np.random.random(7).astype(np.float32),
                np.random.random(7).astype(np.float32),
                np.random.random(7).astype(np.float32),
                np.random.random(7).astype(np.float32),
            ]
        }
        
        print("Creating trajectory from dict of lists with lossy_compression=False...")
        try:
            traj = Trajectory.from_dict_of_lists(data, path, lossy_compression=True)
            print(f"Trajectory created successfully")
            print(f"File exists: {os.path.exists(path)}")
            print(f"File size: {os.path.getsize(path)} bytes")
        except Exception as e:
            print(f"Error creating trajectory: {e}")
            import traceback
            traceback.print_exc()
            return
        
        
        print("\nAttempting to read trajectory...")
        try:
            traj_read = Trajectory(path, mode="r", cache_dir=temp_dir)
            loaded_data = traj_read.load()
            print(loaded_data)
            print(f"Successfully loaded data with keys: {loaded_data.keys()}")
            for key, value in loaded_data.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        except Exception as e:
            print(f"Error reading trajectory: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_trajectory_creation() 