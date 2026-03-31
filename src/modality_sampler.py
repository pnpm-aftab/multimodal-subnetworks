"""
Modality-Specific Sampler for Multimodal Training

This sampler groups samples by modality and yields homogeneous batches,
eliminating the need for sequential forward passes and mask switching.
"""

import torch
from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
import random
from pymongo import MongoClient


class ModalitySpecificBatchSampler(BatchSampler):
    """
    BatchSampler that yields batches containing only a single modality.
    
    This eliminates the M× sequential forward pass bottleneck by ensuring
    each batch is modality-homogeneous. Uses BatchSampler for better
    integration with PyTorch DataLoader - yields lists of indices directly.
    
    Usage:
        dataset = MultimodalMongoDataset(...)
        sampler = ModalitySpecificBatchSampler(
            dataset, 
            batch_size=20,
            modality_field="modalities"
        )
        loader = DataLoader(dataset, batch_sampler=sampler, ...)
    
    Improvements over ModalitySpecificSampler:
    - Uses BatchSampler (PyTorch standard)
    - Yields lists of indices directly (no tuples)
    - Single batched DB query at initialization
    - Simpler dataset integration
    """
    
    def __init__(self, dataset, batch_size, modality_field="modalities", shuffle=True, seed=None,
                 db_host=None, db_name=None, db_collection=None):
        """
        Args:
            dataset: MultimodalMongoDataset instance
            batch_size: Number of samples per batch
            modality_field: Field name in metadata containing available modalities
            shuffle: Whether to shuffle batches
            seed: Random seed for reproducibility
            db_host: MongoDB host (e.g. "hostname:27017") for building groups at init time
            db_name: MongoDB database name
            db_collection: MongoDB collection base name (without .meta/.bin suffix)
        """
        super().__init__(None, batch_size, drop_last=False)
        self.dataset = dataset
        self.batch_size = batch_size
        self.modality_field = modality_field
        self.shuffle = shuffle
        self.seed = seed
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        
        if db_host and db_name and db_collection:
            # Build groups immediately using our own DB connection (main process)
            self.modality_groups = self._build_modality_groups()
            self._groups_built = True
        else:
            # Defer until dataset.collection is available (inside DataLoader worker)
            self._groups_built = False
            self.modality_groups = None
        
    def _build_modality_groups(self):
        """
        Group dataset indices by their available modalities.
        
        Uses a SINGLE batched DB query to fetch all metadata, then groups locally.
        Prefers an explicit DB connection (db_host/db_name/db_collection) if provided,
        otherwise falls back to dataset.collection (available inside DataLoader workers).
        
        Returns:
            dict: {modality_name: [indices]}
        """
        # Choose which connection to use
        if self.db_host and self.db_name and self.db_collection:
            client = MongoClient("mongodb://" + self.db_host)
            db = client[self.db_name]
            meta_col = db[self.db_collection + ".meta"]
        else:
            meta_col = self.dataset.collection["meta"]
        
        # Fetch ALL metadata in one query (not N queries!)
        all_meta = list(meta_col.find(
            {}, 
            {self.modality_field: 1, "id": 1}
        ))
        
        # Build lookup dict for O(1) access
        meta_lookup = {meta["id"]: meta for meta in all_meta}
        
        # Group by modality
        modality_groups = defaultdict(list)
        
        for idx in range(len(self.dataset.indices)):
            subject_id = self.dataset.indices[idx]
            meta = meta_lookup.get(subject_id)
            
            if meta and self.modality_field in meta:
                available_modalities = meta[self.modality_field]
                
                # For each available modality, add this subject to that group
                for mod in available_modalities:
                    if mod in self.dataset.sample:  # Only if modality is in our training set
                        modality_groups[mod].append(idx)  # Store index only, not tuple
        
        # Shuffle if needed
        result = {}
        for mod, indices in modality_groups.items():
            if self.shuffle:
                if self.seed is not None:
                    random.Random(self.seed).shuffle(indices)
                else:
                    random.shuffle(indices)
            result[mod] = indices
        
        return result
    
    def _ensure_groups_built(self):
        """Build modality groups on first call (inside DataLoader worker)."""
        if not self._groups_built:
            self.modality_groups = self._build_modality_groups()
            self._groups_built = True

    def __iter__(self):
        """
        Yield batches of dataset indices.
        Each batch contains only indices from a single modality.
        
        Iterates through each modality completely before moving to the next,
        ensuring homogeneous batches.
        
        Yields:
            list: List of dataset indices [idx1, idx2, ..., idxN]
        """
        self._ensure_groups_built()
        # Determine order of modalities
        mod_order = list(self.modality_groups.keys())
        if self.shuffle:
            if self.seed is not None:
                random.Random(self.seed).shuffle(mod_order)
            else:
                random.shuffle(mod_order)
        
        # Iterate through each modality completely
        for mod in mod_order:
            indices = self.modality_groups[mod]
            
            # Yield batches from this modality
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if batch:
                    yield batch
    
    def __len__(self):
        """
        Total number of batches per epoch.
        """
        self._ensure_groups_built()
        total_samples = sum(len(indices) for indices in self.modality_groups.values())
        return (total_samples + self.batch_size - 1) // self.batch_size
    
    def set_epoch(self, epoch):
        """
        Set epoch for reproducible shuffling.
        """
        if self.seed is not None:
            self.seed = self.seed + epoch
            # Force rebuild groups with new seed on next iteration
            self._groups_built = False


# Backward compatibility alias
ModalitySpecificSampler = ModalitySpecificBatchSampler


class HybridModalitySampler(Sampler):
    """
    Hybrid sampler that can switch between modality-specific and mixed batches.
    
    Useful for gradual transition or comparison testing.
    """
    
    def __init__(self, dataset, batch_size, modality_prob=0.8, shuffle=True, seed=None):
        """
        Args:
            dataset: MultimodalMongoDataset instance
            batch_size: Number of samples per batch
            modality_prob: Probability of yielding modality-specific batch (vs mixed)
            shuffle: Whether to shuffle batches
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.modality_prob = modality_prob
        self.shuffle = shuffle
        self.seed = seed
        
        self.modality_sampler = ModalitySpecificBatchSampler(
            dataset, batch_size, shuffle=shuffle, seed=seed
        )
    
    def __iter__(self):
        # For now, just use modality-specific sampler
        # Can be extended to support mixed batches
        for batch in self.modality_sampler:
            yield batch
    
    def __len__(self):
        return len(self.modality_sampler)
