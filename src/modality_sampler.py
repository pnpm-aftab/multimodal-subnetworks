"""
Modality-Specific Sampler for Multimodal Training

This sampler groups samples by modality and yields homogeneous batches,
eliminating the need for sequential forward passes and mask switching.
"""

import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random


class ModalitySpecificSampler(Sampler):
    """
    Sampler that yields batches containing only a single modality.
    
    This eliminates the M× sequential forward pass bottleneck by ensuring
    each batch is modality-homogeneous.
    
    Usage:
        dataset = MultimodalMongoDataset(...)
        sampler = ModalitySpecificSampler(
            dataset, 
            batch_size=20,
            modality_field="modalities"
        )
        loader = DataLoader(dataset, sampler=sampler, ...)
    """
    
    def __init__(self, dataset, batch_size, modality_field="modalities", shuffle=True, seed=None):
        """
        Args:
            dataset: MultimodalMongoDataset instance
            batch_size: Number of samples per batch
            modality_field: Field name in metadata containing available modalities
            shuffle: Whether to shuffle batches
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.modality_field = modality_field
        self.shuffle = shuffle
        self.seed = seed
        
        # Build modality groups
        self.modality_groups = self._build_modality_groups()
        
    def _build_modality_groups(self):
        """
        Group dataset indices by their available modalities.
        
        Returns:
            dict: {modality_name: [indices]}
        """
        modality_groups = defaultdict(list)
        
        for idx in range(len(self.dataset.indices)):
            # Get subject ID
            subject_id = self.dataset.indices[idx]
            
            # Fetch metadata to get available modalities
            # Note: This requires DB access - we'll cache it
            meta = self.dataset.collection["meta"].find_one(
                {"id": subject_id},
                {self.modality_field: 1, "id": 1}
            )
            
            if meta and self.modality_field in meta:
                available_modalities = meta[self.modality_field]
                
                # For each available modality, add this subject to that group
                for mod in available_modalities:
                    if mod in self.dataset.sample:  # Only if modality is in our training set
                        modality_groups[mod].append((idx, mod))
        
        # Convert to lists and shuffle if needed
        result = {}
        for mod, items in modality_groups.items():
            if self.shuffle:
                if self.seed is not None:
                    random.Random(self.seed).shuffle(items)
                else:
                    random.shuffle(items)
            result[mod] = items
        
        return result
    
    def __iter__(self):
        """
        Yield batches of (dataset_index, modality) tuples.
        Each batch contains only a single modality.
        """
        # Create iterators for each modality group
        modality_iterators = {}
        for mod, items in self.modality_groups.items():
            modality_iterators[mod] = iter(items)
        
        # Track which modalities still have samples
        active_modalities = set(self.modality_groups.keys())
        
        while active_modalities:
            # Shuffle order of modalities to balance training
            mod_order = list(active_modalities)
            if self.shuffle:
                if self.seed is not None:
                    random.Random(self.seed).shuffle(mod_order)
                else:
                    random.shuffle(mod_order)
            
            for mod in mod_order:
                if mod not in active_modalities:
                    continue
                
                # Collect a batch of this modality
                batch = []
                try:
                    for _ in range(self.batch_size):
                        batch.append(next(modality_iterators[mod]))
                except StopIteration:
                    # This modality is exhausted
                    if batch:  # Yield partial batch
                        yield batch
                    active_modalities.discard(mod)
                    continue
                
                if batch:
                    yield batch
    
    def __len__(self):
        """
        Total number of batches per epoch.
        """
        total_samples = sum(len(items) for items in self.modality_groups.values())
        return (total_samples + self.batch_size - 1) // self.batch_size
    
    def set_epoch(self, epoch):
        """
        Set epoch for reproducible shuffling.
        """
        if self.seed is not None:
            self.seed = self.seed + epoch
            # Rebuild groups with new seed
            self.modality_groups = self._build_modality_groups()


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
        
        self.modality_sampler = ModalitySpecificSampler(
            dataset, batch_size, shuffle=shuffle, seed=seed
        )
    
    def __iter__(self):
        # For now, just use modality-specific sampler
        # Can be extended to support mixed batches
        for batch in self.modality_sampler:
            yield batch
    
    def __len__(self):
        return len(self.modality_sampler)
