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
                self.meta_sample + (self.id,),
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
    This class is designed to fetch SINGLE OR MULTIPLE MODALITIES, MODALITY CODE, and SINGLE LABEL.
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
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter
        
        # 1. Fetch binary data in one batch
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample}, # .bin contains 3D kinds like 'smri', 'falff', 'dwi'.
                },
                self.fields,
            )
        )

        # Pre-group chunks by (id, kind) for O(N) access instead of O(N^2) filtering
        chunks_by_id_kind = {}
        for s in samples:
            key = (s[self.id], s["kind"])
            if key not in chunks_by_id_kind:
                chunks_by_id_kind[key] = []
            chunks_by_id_kind[key].append(s)

        # 2. Batch metadata query: fetch all metadata for the batch at once
        batch_ids = [self.indices[_] for _ in batch]
        all_meta = list(
            self.collection["meta"].find(
                {
                    self.id: {"$in": batch_ids},
                },
                self.meta_sample + ("modalities", self.id),
            )
        )
        # Create mapping from ID to metadata for fast lookup
        meta_lookup = {meta[self.id]: meta for meta in all_meta}

        results = {}
        for id in batch:
            # Lookup metadata from pre-fetched batch
            meta_for_id = meta_lookup.get(self.indices[id])
            if meta_for_id is None:
                continue

            label = meta_for_id[self.meta_sample[0]]
            modalities = meta_for_id["modalities"]
            id_modalities = set(modalities).intersection(set(self.sample))

            for mod in id_modalities:
                # O(1) lookup using pre-grouped dict instead of O(N) scan
                data = self.make_serial(chunks_by_id_kind.get((self.indices[id], mod), []), mod)

                result = {
                    "input": self.normalize(self.transform(data).float()),
                    "modality": mod,
                    "label": torch.tensor(label).unsqueeze(0),
                }

                # Add to results
                results[str(id)+'_'+mod] = result

        return results
    

def multimodal_collate(results, field=("input", "modality", "label")):
    """
    Use this collate function with BatchPrefetchLoaderWrapper when using MultimodalMongoDataset.
    It will stack the inputs, modality codes, and labels into tensors properly.
    """
    results = results[0]
    # Assuming 'results' is your dictionary containing all the data
    input_tensors = [results[id_][field[0]] for id_ in results.keys()]
    modalities = [results[id_][field[1]] for id_ in results.keys()]
    label_tensors = [results[id_][field[2]] for id_ in results.keys()]
    # Stack all input tensors into a single tensor
    stacked_inputs = torch.stack(input_tensors)
    # Stack all label tensors into a single tensor
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