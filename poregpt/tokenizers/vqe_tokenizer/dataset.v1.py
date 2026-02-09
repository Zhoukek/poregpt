import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class NanoporeSignalDataset(Dataset):
    def __init__(self, shards_dir, logic_chunk_size=None, max_cache_size=256):
        """
        Args:
            shards_dir (str): Directory containing shards.npy and shards.json.
            logic_chunk_size (int, optional): Logic chunk size for training. 
                                            Must be <= chunk_size and divisible by chunk_size.
                                            If None, uses the original chunk_size.
            max_cache_size (int): Max number of memmap files to keep open (per process).
        """
        meta_path = os.path.join(shards_dir, "shards.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.shard_info = meta["shards"]  # [{"path": "...", "num_samples": N}, ...]
        self.chunk_size = meta["chunk_size"]  # physical chunk size (e.g., 12000)
        self.dtype = np.dtype(meta["dtype"])
        self.shards_dir = shards_dir
        self.max_cache_size = max_cache_size

        # Set logic chunk size
        if logic_chunk_size is None:
            self.logic_chunk_size = self.chunk_size
        else:
            if logic_chunk_size > self.chunk_size:
                raise ValueError(f"logic_chunk_size ({logic_chunk_size}) must be <= chunk_size ({self.chunk_size})")
            if self.chunk_size % logic_chunk_size != 0:
                raise ValueError(f"chunk_size ({self.chunk_size}) must be divisible by logic_chunk_size ({logic_chunk_size})")
            self.logic_chunk_size = logic_chunk_size

        # Calculate how many logic chunks each physical chunk contains
        self.chunks_per_physical = self.chunk_size // self.logic_chunk_size

        # Build global offsets based on logic chunks
        self.offsets = [0]
        for info in self.shard_info:
            num_physical_chunks = info["num_samples"]
            num_logic_chunks = num_physical_chunks * self.chunks_per_physical
            self.offsets.append(self.offsets[-1] + num_logic_chunks)
        self.total_logic_samples = self.offsets[-1]

        # LRU cache for memmap objects: {shard_path: memmap_array}
        self._cache = OrderedDict()

    def _get_memmap(self, shard_path):
        """Get memmap array with LRU caching."""
        if shard_path in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        # Open new memmap
        memmap_arr = np.load(shard_path, mmap_mode='r')

        # Evict oldest if cache is full
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)  # Remove least recently used

        self._cache[shard_path] = memmap_arr
        return memmap_arr

    def _find_physical_shard_and_chunk(self, logic_idx):
        """Find which physical shard and which logic chunk within that physical chunk."""
        # Find physical shard
        physical_shard_id = 0
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= logic_idx < self.offsets[i + 1]:
                physical_shard_id = i
                break
        
        # Find which logic chunk within the physical shard
        logic_chunks_in_prev_shards = self.offsets[physical_shard_id]
        local_logic_idx = logic_idx - logic_chunks_in_prev_shards  # 0-based within this physical shard
        
        # Convert to physical chunk index and logic sub-chunk index
        physical_chunk_idx = local_logic_idx // self.chunks_per_physical
        logic_subchunk_idx = local_logic_idx % self.chunks_per_physical
        
        return physical_shard_id, physical_chunk_idx, logic_subchunk_idx

    def __len__(self):
        return self.total_logic_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_logic_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_logic_samples})")

        # Find the physical location and sub-chunk position
        physical_shard_id, physical_chunk_idx, logic_subchunk_idx = self._find_physical_shard_and_chunk(idx)
        
        shard_filename = self.shard_info[physical_shard_id]["path"]
        shard_path = os.path.join(self.shards_dir, shard_filename)

        # Get the physical chunk
        data = self._get_memmap(shard_path)
        physical_chunk = data[physical_chunk_idx]  # shape: (chunk_size,)

        # Extract the logic sub-chunk
        start_idx = logic_subchunk_idx * self.logic_chunk_size
        end_idx = start_idx + self.logic_chunk_size
        logic_chunk = physical_chunk[start_idx:end_idx]  # shape: (logic_chunk_size,)

        return torch.from_numpy(logic_chunk.copy()).unsqueeze(0)  # shape: (1, logic_chunk_size)

    def get_original_chunk_size(self):
        """Return the original physical chunk size."""
        return self.chunk_size

    def get_logic_chunk_size(self):
        """Return the current logic chunk size."""
        return self.logic_chunk_size

    def get_total_physical_samples(self):
        """Return total number of physical (original) samples."""
        return sum(info["num_samples"] for info in self.shard_info)
