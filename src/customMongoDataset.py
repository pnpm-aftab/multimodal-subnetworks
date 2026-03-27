from mindfultensors.mongoloader import MongoDataset
import torch
from mindfultensors.utils import unit_interval_normalize

class CustomMongoDataset(MongoDataset):
    """
    CustomMongoDataset is designed to work with the new "MindfulTensors" databasde organization.
    Gender labels there are stored in the .meta collection, while the 3D image data is stored in the .bin collection.
    This class is designed to fetch SINGLE MODALITY and SINGLE LABEL from both collections and returns them appropriately.
    USE MultimodalMongoDataset TO pull MULTIPLE modalities per subject.
    """
    def __init__(
        self,
        indices,
        transform,
        collection,
        sample,
        meta_sample,
        normalize=unit_interval_normalize,
        id="id",
    ):
        super(CustomMongoDataset, self).__init__(indices, transform, collection, sample, normalize, id)
        self.meta_sample = meta_sample

    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter

        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample}, # .bin contains 3D kinds like 'smri', 'falff', 'dwi'. Scalar labels are stored in .meta
                },
                self.fields,
            )
        )

        # Batch metadata query: fetch all metadata for the batch at once (not N queries)
        batch_ids = [self.indices[_] for _ in batch]
        all_meta = list(
            self.collection["meta"].find(
                {
                    self.id: {"$in": batch_ids},
                },
                self.meta_sample,
            )
        )
        # Create mapping from ID to metadata for fast lookup
        meta_lookup = {meta[self.id]: meta for meta in all_meta}

        results = {}
        for id in batch:
            # Separate samples for this id
            samples_for_id = [
                sample
                for sample in samples
                if sample[self.id] == self.indices[id]
            ]

            # Separate processing for each 'kind' # TODO: for multimodal, pull all kinds here and then just match them with labels properly
            data = self.make_serial(samples_for_id, self.sample[0])

            # Lookup metadata from pre-fetched batch (no DB query here)
            meta_for_id = meta_lookup.get(self.indices[id])
            assert meta_for_id is not None, f"No meta entries found for id {id}"

            label = meta_for_id[self.meta_sample[0]]

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": torch.tensor(label).unsqueeze(0),
            }

        return results
    
class MultimodalMongoDataset(MongoDataset):
    """
    MultimodalMongoDataset is designed to work with the new "MindfulTensors" databasde organization.
    Gender labels there are stored in the .meta collection, while the 3D image data is stored in the .bin collection.
    
    UPDATED FOR MODALITY-SPECIFIC BATCHING:
    Now supports efficient modality-homogeneous batches when used with ModalitySpecificSampler.
    Each batch contains samples from a single modality, eliminating sequential forward passes.
    
    Modality codes are assigned using the following dictionary:
    modality_mapping = {
        "smri": 0,
        "falff": 1,
        "dwi": 2,
    }
    """
    def __init__(
        self,
        indices,
        transform,
        collection,
        sample,
        meta_sample,
        normalize=unit_interval_normalize,
        id="id",
    ):
        super(MultimodalMongoDataset, self).__init__(indices, transform, collection, sample, normalize, id)
        self.meta_sample = meta_sample

    def __getitem__(self, batch):
        """
        Fetch samples for a batch.
        
        Args:
            batch: List of dataset indices for modality-specific sampling
                   (From ModalitySpecificBatchSampler)
        
        Returns:
            dict: {batch_idx: {"input": tensor, "modality": str, "label": tensor}}
        """
        return self._get_modality_specific_batch(batch)
    
    def _get_modality_specific_batch(self, batch):
        """
        NEW: Modality-specific batching for optimal performance.
        Each batch contains samples from a single modality.
        
        Args:
            batch: List of dataset indices (from ModalitySpecificBatchSampler)
        
        Returns:
            dict: {batch_idx: {"input": tensor, "modality": str, "label": tensor}}
        """
        # Fetch metadata for all indices to determine modality
        batch_ids = [self.indices[_] for _ in batch]
        all_meta = list(
            self.collection["meta"].find(
                {
                    self.id: {"$in": batch_ids},
                },
                self.meta_sample,
            )
        )
        meta_lookup = {meta[self.id]: meta for meta in all_meta}
        
        # Determine modality from first sample (all should have same available modalities)
        first_subject_id = self.indices[batch[0]]
        first_meta = meta_lookup.get(first_subject_id)
        assert first_meta is not None, f"No meta entries found for id {first_subject_id}"
        
        # Get the first available modality that's in our sample set
        available_modalities = first_meta.get("modalities", [])
        target_modality = None
        for mod in available_modalities:
            if mod in self.sample:
                target_modality = mod
                break
        
        assert target_modality is not None, f"No valid modality found for subject {first_subject_id}"
        
        # Fetch binary data for all subjects in batch (single modality)
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": batch_ids},
                    "kind": target_modality,  # Single modality query
                },
                self.fields,
            )
        )
        
        # Build results
        results = {}
        for batch_pos, dataset_idx in enumerate(batch):
            subject_id = self.indices[dataset_idx]
            
            # Get binary data for this subject
            samples_for_id = [
                sample for sample in samples if sample[self.id] == subject_id
            ]
            data = self.make_serial(samples_for_id, target_modality)
            
            # Get label
            meta_for_id = meta_lookup.get(subject_id)
            assert meta_for_id is not None, f"No meta entries found for id {subject_id}"
            label = meta_for_id[self.meta_sample[0]]
            
            results[batch_pos] = {
                "input": self.normalize(self.transform(data).float()),
                "modality": target_modality,
                "label": torch.tensor(label).unsqueeze(0),
            }
        
        return results
    

def multimodal_collate(results, field=("input", "modality", "label")):
    """
    Collate function for MultimodalMongoDataset.
    
    Supports both modality-specific batches (new, optimized) and mixed-modality batches (legacy).
    Works with BatchPrefetchLoaderWrapper.
    
    Args:
        results: List of dicts from MultimodalMongoDataset.__getitem__
        field: Tuple of field names to extract
    
    Returns:
        tuple: (inputs, modalities, labels)
            - inputs: Tensor [B, 1, H, W, D]
            - modalities: Tensor [B] of modality codes
            - labels: Tensor [B, 1]
    """
    results = results[0]
    
    # Extract data from results dict
    input_tensors = [results[id_][field[0]] for id_ in results.keys()]
    modalities = [results[id_][field[1]] for id_ in results.keys()]
    label_tensors = [results[id_][field[2]] for id_ in results.keys()]
    
    # Stack into batches
    stacked_inputs = torch.stack(input_tensors)
    stacked_modalities = torch.stack([torch.tensor(map_modality_codes(mod)) for mod in modalities])
    stacked_labels = torch.stack(label_tensors)
    
    return stacked_inputs.unsqueeze(1), stacked_modalities.long(), stacked_labels.long()

def map_modality_codes(mod):
    """
    Maps modality strings to integer codes.
    """
    modality_mapping = {
        "smri": 0,
        "falff": 1,
        "dwi": 2,
    }
    return modality_mapping[mod]

def make_serial(samples_for_id, kind):
    """
    Serializes chunks into a single binary blob. From MongoDataset self methods.
    """
    return b"".join(
        [
            sample["chunk"]
            for sample in sorted(
                (
                    sample
                    for sample in samples_for_id
                    if sample["kind"] == kind
                ),
                key=lambda x: x["chunk_id"],
            )
        ]
    )