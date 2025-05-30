import glob
import logging
import os
import random
from typing import Any, List, Optional, Text

import fog_x
from fog_x.loader.base import BaseLoader

logger = logging.getLogger(__name__)


class VLALoader:

    def __init__(self,
                 path: Text,
                 batch_size=1,
                 return_type="numpy",
                 split="all"):
        self.files = self._get_files(path, split)
        self.split = split
        self.batch_size = batch_size
        self.return_type = return_type
        self.index = 0
        random.shuffle(self.files)

    def _get_files(self, path, split):
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]
        if split == "train":
            ret = ret[:int(len(ret) * 0.9)]
        elif split == "val":
            ret = ret[int(len(ret) * 0.9):]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Invalid split: {split}")
        return ret

    def _read_vla(self, data_path, return_type=None):
        if return_type is None:
            return_type = self.return_type
        traj = fog_x.Trajectory(data_path)
        ret = traj.load(return_type=return_type)
        return ret

    def get_batch(self) -> List[Any]:
        batch = []

        for _ in range(self.batch_size):
            if self.index >= len(self.files):
                break  # No more files available

            file_path = self.files[self.index]
            self.index += 1

            try:
                data = self._read_vla(file_path)
                batch.append(data)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue  # Skip this file and continue

        return batch if batch else []

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.get_batch()
        if batch is None:
            # Reset for next epoch
            self.index = 0
            random.shuffle(self.files)
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.files)

    def peek(self):
        if self.index < len(self.files):
            file = self.files[self.index]
            return self._read_vla(file, return_type="numpy")
        return None

    def __del__(self):
        pass


class NonShuffleVLALoader:

    def __init__(self,
                 path: Text,
                 batch_size=1,
                 num_workers=1,
                 return_type="numpy"):
        self.files = self._get_files(path)
        self.batch_size = batch_size
        self.return_type = return_type
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.files):
            raise StopIteration

        max_retries = 3
        for attempt in range(max_retries):
            try:
                file_path = self.files[self.index]
                self.index += 1
                return self._read_vla(file_path, return_type=self.return_type)
            except Exception as e:
                logger.error(
                    f"Error reading {file_path} on attempt {attempt + 1}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(
                        f"Failed to read {file_path} after {max_retries} attempts"
                    )
                    raise e  # Re-raise the last exception instead of returning None

    def _get_files(self, path):
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]
        # for file in ret:
        #     try:
        #         self._read_vla(file, return_type = self.return_type)
        #     except Exception as e:
        #         logger.error(f"Error reading {file}: {e}, ")
        #         ret.remove(file)
        return ret

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]

    def __del__(self):
        pass

    def peek(self):
        file = self.files[self.index]
        return self._read_vla(file, return_type="numpy")

    def _read_vla(self, data_path, return_type=None):
        if return_type is None:
            return_type = self.return_type
        traj = fog_x.Trajectory(data_path)
        ret = traj.load(return_type=return_type)
        return ret

    def get_batch(self):
        return [self.__next__() for _ in range(self.batch_size)]


from typing import Optional, Text

import torch
from torch.utils.data import DataLoader, IterableDataset

from fog_x.loader.vla import VLALoader


class VLAIterableDataset(IterableDataset):

    def __init__(self, path: Text, buffer_size: int = 1000):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader
        # in this case, we use pytorch dataloader for batching
        self.vla_loader = VLALoader(path, batch_size=1)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.vla_loader.get_batch()
        if batch is None:
            raise StopIteration
        return batch[0]  # Return a single item, not a batch


def vla_collate_fn(batch):
    # Convert data to PyTorch tensors
    # You may need to adjust this based on the structure of your VLA data
    return batch  # {k: torch.tensor(v) for k, v in batch[0].items()}


def get_vla_dataloader(path: Text,
                       batch_size: int = 1,
                       buffer_size: int = 1000,
                       num_workers: int = 0):
    dataset = VLAIterableDataset(path, buffer_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=vla_collate_fn,
        num_workers=num_workers,
    )
