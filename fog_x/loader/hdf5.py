import torch
from torch.utils.data import IterableDataset, DataLoader
from . import BaseLoader
import numpy as np
import glob
import h5py
import asyncio
import random
import multiprocessing as mp
import time
import logging
from fog_x.utils import _flatten, recursively_read_hdf5_group
import os
from typing import Dict, Any, List, Text, Optional


def convert_vla_data_to_hdf5(
    data: Dict[str, Any], 
    output_path: Text,
    compression: str = "gzip",
    compression_opts: int = 9
) -> None:
    """
    Convert VLA (Vision-Language-Action) data to HDF5 format.
    
    Args:
        data (Dict[str, Any]): Dictionary containing VLA data with feature names as keys
        output_path (Text): Path where the HDF5 file will be saved
        compression (str): Compression algorithm to use (default: "gzip")
        compression_opts (int): Compression level (0-9, default: 9)
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with h5py.File(output_path, "w") as h5_file:
            _write_dict_to_hdf5_group(h5_file, data, compression, compression_opts)
        logging.info(f"Successfully converted VLA data to HDF5: {output_path}")
    except Exception as e:
        logging.error(f"Error converting VLA data to HDF5: {e}")
        raise


def _write_dict_to_hdf5_group(
    group: h5py.Group, 
    data_dict: Dict[str, Any],
    compression: str = "gzip",
    compression_opts: int = 9
) -> None:
    """
    Recursively write a dictionary to an HDF5 group.
    
    Args:
        group (h5py.Group): HDF5 group to write to
        data_dict (Dict[str, Any]): Data dictionary to write
        compression (str): Compression algorithm
        compression_opts (int): Compression level
    """
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create subgroup for nested dictionaries
            subgroup = group.create_group(key)
            _write_dict_to_hdf5_group(subgroup, value, compression, compression_opts)
        else:
            # Convert value to numpy array if needed
            if not isinstance(value, np.ndarray):
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                else:
                    # Single value
                    value = np.array([value])
            
            # Handle object arrays (strings, mixed types)
            if value.dtype == object:
                # Convert object arrays to string arrays for HDF5 compatibility
                string_data = []
                for item in value.flat:
                    if isinstance(item, (str, bytes)):
                        string_data.append(str(item))
                    else:
                        string_data.append(str(item))
                value = np.array(string_data, dtype='S')
            
            # Create dataset with compression
            try:
                group.create_dataset(
                    key, 
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts
                )
            except Exception as e:
                logging.warning(f"Failed to compress {key}, saving without compression: {e}")
                group.create_dataset(key, data=value)


def convert_trajectory_to_hdf5(
    trajectory_path: Text,
    output_path: Text,
    compression: str = "gzip",
    compression_opts: int = 9
) -> None:
    """
    Convert a trajectory container file to HDF5 format.
    
    Args:
        trajectory_path (Text): Path to the trajectory container file
        output_path (Text): Path where the HDF5 file will be saved
        compression (str): Compression algorithm to use (default: "gzip")
        compression_opts (int): Compression level (0-9, default: 9)
    """
    
    # Import here to avoid circular imports
    from ..trajectory import Trajectory
    
    try:
        # Load trajectory data
        traj = Trajectory(trajectory_path, mode="r")
        data = traj.load(return_type="numpy")
        traj.close()
        
        # Convert to HDF5
        convert_vla_data_to_hdf5(data, output_path, compression, compression_opts)
        
    except Exception as e:
        logging.error(f"Error converting trajectory to HDF5: {e}")
        raise


def batch_convert_trajectories_to_hdf5(
    trajectory_paths: List[Text],
    output_dir: Text,
    compression: str = "gzip",
    compression_opts: int = 9,
    parallel: bool = True,
    num_workers: Optional[int] = None
) -> None:
    """
    Convert multiple trajectory files to HDF5 format in batch.
    
    Args:
        trajectory_paths (List[Text]): List of trajectory file paths
        output_dir (Text): Directory where HDF5 files will be saved
        compression (str): Compression algorithm to use
        compression_opts (int): Compression level
        parallel (bool): Whether to use parallel processing
        num_workers (Optional[int]): Number of worker processes (default: CPU count)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if parallel:
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        with mp.Pool(num_workers) as pool:
            tasks = []
            for traj_path in trajectory_paths:
                output_filename = os.path.splitext(os.path.basename(traj_path))[0] + ".h5"
                output_path = os.path.join(output_dir, output_filename)
                
                task = pool.apply_async(
                    convert_trajectory_to_hdf5,
                    (traj_path, output_path, compression, compression_opts)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            for task in tasks:
                task.get()
    else:
        for traj_path in trajectory_paths:
            output_filename = os.path.splitext(os.path.basename(traj_path))[0] + ".h5"
            output_path = os.path.join(output_dir, output_filename)
            
            convert_trajectory_to_hdf5(traj_path, output_path, compression, compression_opts)


def load_and_convert_to_hdf5(
    input_path: Text,
    output_path: Text,
    input_format: str = "auto",
    compression: str = "gzip",
    compression_opts: int = 9
) -> None:
    """
    Load data from various formats and convert to HDF5.
    
    Args:
        input_path (Text): Path to input data file
        output_path (Text): Path for output HDF5 file
        input_format (str): Format of input data ("auto", "trajectory", "numpy", "pickle")
        compression (str): HDF5 compression algorithm
        compression_opts (int): Compression level
    """
    
    if input_format == "auto":
        # Auto-detect format based on file extension
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".mkv", ".mp4", ".avi"]:
            input_format = "trajectory"
        elif ext in [".npy", ".npz"]:
            input_format = "numpy"
        elif ext in [".pkl", ".pickle"]:
            input_format = "pickle"
        else:
            raise ValueError(f"Cannot auto-detect format for file: {input_path}")
    
    if input_format == "trajectory":
        convert_trajectory_to_hdf5(input_path, output_path, compression, compression_opts)
    
    elif input_format == "numpy":
        if input_path.endswith(".npz"):
            data = dict(np.load(input_path))
        else:
            data = {"data": np.load(input_path)}
        convert_vla_data_to_hdf5(data, output_path, compression, compression_opts)
    
    elif input_format == "pickle":
        import pickle
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            data = {"data": data}
        convert_vla_data_to_hdf5(data, output_path, compression, compression_opts)
    
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def main():
    """
    Command-line interface for VLA data to HDF5 conversion.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert VLA data to HDF5 format")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output HDF5 file path")
    parser.add_argument("--format", choices=["auto", "trajectory", "numpy", "pickle"], 
                       default="auto", help="Input data format (default: auto)")
    parser.add_argument("--compression", default="gzip", 
                       help="HDF5 compression algorithm (default: gzip)")
    parser.add_argument("--compression-level", type=int, default=9,
                       help="Compression level 0-9 (default: 9)")
    parser.add_argument("--batch", action="store_true", 
                       help="Treat input as directory and convert all files")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Use parallel processing for batch conversion")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError("Input must be a directory when using --batch")
        
        # Find all relevant files in the directory
        trajectory_files = []
        for ext in ["*.mkv", "*.mp4", "*.avi"]:
            trajectory_files.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
        
        if not trajectory_files:
            print(f"No trajectory files found in {args.input}")
            return
        
        print(f"Found {len(trajectory_files)} trajectory files to convert")
        batch_convert_trajectories_to_hdf5(
            trajectory_files,
            args.output,
            compression=args.compression,
            compression_opts=args.compression_level,
            parallel=args.parallel,
            num_workers=args.workers
        )
        print(f"Batch conversion completed. Files saved to {args.output}")
    else:
        load_and_convert_to_hdf5(
            args.input,
            args.output,
            input_format=args.format,
            compression=args.compression,
            compression_opts=args.compression_level
        )
        print(f"Conversion completed: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()


class HDF5Loader(BaseLoader):
    def __init__(self, path, batch_size=1, buffer_size=50, num_workers=4):
        super(HDF5Loader, self).__init__(path)
        self.files = glob.glob(self.path, recursive=True)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.num_workers = num_workers
        self.processes = []
        random.shuffle(self.files)
        self._start_workers()

    def _worker(self):
        while True:
            if not self.files:
                logging.info("Worker finished")
                break
            file_path = random.choice(self.files)
            data = self._read_hdf5(file_path)
            self.buffer.put(data)

    def _start_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker)
            p.start()
            logging.debug(f"Started worker {p.pid}")
            self.processes.append(p)

    def get_batch(self):
        batch = []
        timeout = 5
        start_time = time.time()

        while len(batch) < self.batch_size:
            if time.time() - start_time > timeout:
                logging.warning(
                    f"Timeout reached while getting batch. Batch size: {len(batch)}"
                )
                break

            try:
                item = self.buffer.get(timeout=1)
                batch.append(item)
            except mp.queues.Empty:
                if (
                    all(not p.is_alive() for p in self.processes)
                    and self.buffer.empty()
                ):
                    if len(batch) == 0:
                        return None
                    else:
                        break
        return batch

    def __next__(self):
        batch = self.get_batch()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

    def _read_hdf5(self, data_path):
        with h5py.File(data_path, "r") as f:
            data_unflattened = recursively_read_hdf5_group(f)
        print(data_unflattened.keys())
        data = {}
        data["observation"] = _flatten(data_unflattened["observation"])
        data["action"] = _flatten(data_unflattened["action"])

        return data_unflattened

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def peek(self):
        if self.buffer.empty():
            return None
        return self.buffer.get()

    def __del__(self):
        for p in self.processes:
            p.terminate()
            p.join()


class HDF5IterableDataset(IterableDataset):
    def __init__(self, path, batch_size=1):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader
        self.hdf5_loader = HDF5Loader(path, 1)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.hdf5_loader)
            return batch[0]  # Return a single item, not a batch
        except StopIteration:
            raise StopIteration


def hdf5_collate_fn(batch):
    # Convert data to PyTorch tensors
    return batch


def get_hdf5_dataloader(path: str, batch_size: int = 1, num_workers: int = 0):
    dataset = HDF5IterableDataset(path, batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=hdf5_collate_fn,
        num_workers=num_workers,
    )
