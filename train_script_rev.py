import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import csv
import random
import shutil
import math
import time
from packaging import version
import yaml

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import distributed

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from resnet import ResNet3D

from mindfultensors.mongoloader import MongoClient
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

from src.db_client import ClientCreator
from src.customMongoDataset import CustomMongoDataset, MultimodalMongoDataset, multimodal_collate, make_serial
from src.masked_model import MultiMaskSNIPWrapper
from src.utils import setup_distributed_port

SEED = random.randint(0, 9999)
utils.set_global_seed(SEED)
setup_distributed_port(seed=SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
# os.environ["NCCL_P2P_LEVEL"] = "NVL"

torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

torch_version = torch.__version__
if version.parse(torch_version) >= version.parse("2.3"):
    scaler = torch.amp.GradScaler()
else:
    scaler = torch.cuda.amp.GradScaler()


def get_rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


class DistributedDBBatchSampler(DBBatchSampler):
    """
    Rank-sharded variant of DBBatchSampler.

    DataLoader passes each yielded item to MongoDataset.__getitem__ as a batch
    of subject indices, so sharding must happen before those mini-batches are
    formed. Padding keeps each DDP rank at the same number of steps.
    """

    def __init__(
        self,
        data_source,
        batch_size=1,
        seed=None,
        rank=None,
        world_size=None,
        sample_weights=None,
    ):
        super().__init__(data_source, batch_size=batch_size, seed=seed)
        detected_rank, detected_world_size = get_rank_world()
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world_size if world_size is None else world_size
        self.global_batch_size = self.batch_size * self.world_size
        self.num_batches = int(math.ceil(self.data_size / self.global_batch_size))
        self.total_size = self.num_batches * self.global_batch_size
        if sample_weights is None:
            sample_weights = [1] * self.data_size
        if len(sample_weights) != self.data_size:
            raise ValueError("sample_weights must be aligned with dataset indices")
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __iter__(self):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(self.data_size)
        else:
            indices = np.random.permutation(self.data_size)

        padding_size = self.total_size - len(indices)
        if padding_size > 0 and len(indices) > 0:
            repeats = int(math.ceil(padding_size / len(indices)))
            padding = np.tile(indices, repeats)[:padding_size]
            indices = np.concatenate([indices, padding])

        rank_batches = []
        for start in range(0, self.total_size, self.global_batch_size):
            global_batch = indices[start : start + self.global_batch_size]
            per_rank = [[] for _ in range(self.world_size)]
            per_rank_weight = [0.0 for _ in range(self.world_size)]

            # Heaviest subjects first makes greedy balancing effective while the
            # enclosing global batch remains shuffled.
            ordered = sorted(
                global_batch,
                key=lambda idx: float(self.sample_weights[int(idx)]),
                reverse=True,
            )
            for idx in ordered:
                candidates = [
                    r for r in range(self.world_size)
                    if len(per_rank[r]) < self.batch_size
                ]
                target_rank = min(
                    candidates,
                    key=lambda r: (per_rank_weight[r], len(per_rank[r]), r),
                )
                per_rank[target_rank].append(idx)
                per_rank_weight[target_rank] += float(self.sample_weights[int(idx)])

            rank_batches.append(np.asarray(per_rank[self.rank]))

        return iter(rank_batches)

    def __len__(self):
        return self.num_batches
    
# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        wandb_project: str,
        wandb_experiment: str,
        model_path: str,
        n_channels: int,
        n_classes: int,
        n_epochs: int,
        optimize_inline: bool,
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float,
        num_subcubes: int,
        num_volumes: int,
        client_creator,
        off_brain_weight: float,
        indexid: str,
        # modelconfig: str,
        db_host: str,
        db_name: str,
        db_collection: str,
        wandb_team: str,
        db_fields: tuple,
        meta_fields: tuple,
        groupnorm=False,
        prefetches=8,
        num_workers=6,
        prefetch_factor=2,
        train_prefetches=None,
        train_num_workers=None,
        train_prefetch_factor=None,
        train_persistent_workers=True,
        eval_prefetches=None,
        eval_num_workers=None,
        eval_prefetch_factor=None,
        eval_persistent_workers=True,
        run_infer_each_epoch=False,
        profile_timings=False,
        timing_sync_cuda=True,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        lowprecision=False,
        lossweight=[1, 0],
        maxshape=300,
        hparams=None,
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.config_file = modelconfig
        self.optimize_inline = optimize_inline
        self.onecycle_lr = onecycle_lr
        self.validation_percent = validation_percent
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_prefetches = train_prefetches if train_prefetches is not None else prefetches
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.train_prefetch_factor = train_prefetch_factor if train_prefetch_factor is not None else prefetch_factor
        self.train_persistent_workers = train_persistent_workers
        self.eval_prefetches = eval_prefetches if eval_prefetches is not None else prefetches
        self.eval_num_workers = eval_num_workers if eval_num_workers is not None else num_workers
        self.eval_prefetch_factor = eval_prefetch_factor if eval_prefetch_factor is not None else prefetch_factor
        self.eval_persistent_workers = eval_persistent_workers
        self.run_infer_each_epoch = run_infer_each_epoch
        self.profile_timings = profile_timings
        self.timing_sync_cuda = timing_sync_cuda

        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.db_fields = db_fields
        self.meta_fields = meta_fields

        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.n_epochs = n_epochs
        self.off_brain_weight = off_brain_weight
        self.client_creator = client_creator
        self.funcs = None
        self.collate = None
        self.bit16 = lowprecision
        self.index_id = indexid
        self.groupnorm = groupnorm
        self.loss_weight = lossweight
        self.wandb_team = wandb_team
        self.maxshape = maxshape
        self._hparams = hparams

        self.masked = self._hparams["model"].get("masked", False)
        self.timing_metric_keys = [
            "time/data_wait_h2d_sec",
            "time/forward_sec",
            "time/backward_sec",
            "time/optimizer_sec",
            "time/compute_sec",
            "time/batch_total_sec",
        ]

    def _sync_timing(self):
        if self.profile_timings and self.timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _make_sampler(self, dataset, batch_size, seed=None, sample_weights=None):
        if self.engine.is_ddp:
            rank, world_size = get_rank_world()
            return DistributedDBBatchSampler(
                dataset,
                batch_size=batch_size,
                seed=seed,
                rank=rank,
                world_size=world_size,
                sample_weights=sample_weights,
            )
        return DBBatchSampler(dataset, batch_size=batch_size, seed=seed)

    def _make_loader(
        self,
        dataset,
        sampler,
        worker_init_fn,
        num_workers,
        prefetch_factor,
        num_prefetches,
        persistent_workers,
    ):
        loader_kwargs = {
            "sampler": sampler,
            "collate_fn": self.collate,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return BatchPrefetchLoaderWrapper(
            DataLoader(dataset, **loader_kwargs),
            num_prefetches=num_prefetches,
        )

    def get_engine(self):
        # Use SLURM-allocated GPU count, not total visible GPUs on the node
        n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
        if n_gpus > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision="fp16",
                # ddp_kwargs={"backend": "nccl"},
                process_group_kwargs={"backend": "nccl"},
            )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
                entity=self.wandb_team,
            ),
        }

    @property
    def stages(self):
        return ["train"]

    @property
    def num_epochs(self) -> int:
        return self.n_epochs

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        utils.set_global_seed(SEED)
        return SEED

    def get_stage_len(self) -> int:
        return self.n_epochs

    def get_loaders(self):
        #MM
        self.multimodal = True if (len(self.db_fields) > 1 or self.masked) else False

        self.funcs = {
            "createclient": self.client_creator.create_client,
            "createVclient": self.client_creator.create_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }
        
        self.collate = (
            multimodal_collate if self.multimodal else #MM
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        print(
            "[LoaderConfig] "
            f"train_workers={self.train_num_workers}, train_prefetch_factor={self.train_prefetch_factor}, "
            f"train_prefetches={self.train_prefetches}, train_persistent={self.train_persistent_workers}; "
            f"eval_workers={self.eval_num_workers}, eval_prefetch_factor={self.eval_prefetch_factor}, "
            f"eval_prefetches={self.eval_prefetches}, eval_persistent={self.eval_persistent_workers}; "
            f"run_infer_each_epoch={self.run_infer_each_epoch}, profile_timings={self.profile_timings}; "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
        )

        # get all IDs with the required modalities, pull their labels for cross-validation splits

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts_bin = db[self.db_collection + ".bin"]
        posts_meta = db[self.db_collection + ".meta"]

        # get ids, pull labels
        all_ids = posts_meta.distinct( # pull all unique IDs (subjects) with at least one modality in db_fields
            "id",
            {'modalities': {"$in": self.db_fields}}
        )
        all_ids = sorted(all_ids)
        # print(all_ids)

        # Fetch all split labels in one query, preserving all_ids order below.
        label_field = self.meta_fields[0]
        meta_docs = {
            doc["id"]: doc[label_field]
            for doc in posts_meta.find(
                {"id": {"$in": all_ids}},
                {"id": 1, label_field: 1, "modalities": 1, "_id": 0},
            )
        }
        missing_label_ids = [id for id in all_ids if id not in meta_docs]
        if missing_label_ids:
            raise ValueError(f"Missing labels for ids: {missing_label_ids[:10]}")
        labels = np.array([meta_docs[id] for id in all_ids])
    
        # Create CV split
        cv_folds = StratifiedKFold(n_splits=self._hparams["experiment"]["cv_folds"], shuffle=True, random_state=self._hparams["experiment"].get("cv_seed", 42))
        train_idx, test_idx = list(cv_folds.split(all_ids, labels))[self._hparams["fold_idx"]]
        # split train into train and validation
        train_idx, valid_idx = train_test_split(train_idx, test_size=self.validation_percent, stratify=labels[train_idx], random_state=self._hparams["experiment"].get("cv_seed", 42))

        all_ids = np.array(all_ids)
        train_ids = all_ids[train_idx].tolist() # mongo expects default python list, not numpy array
        valid_ids = all_ids[valid_idx].tolist()
        test_ids = all_ids[test_idx].tolist()
        requested_modalities = set(self.db_fields)
        modality_counts = {
            id: max(
                1,
                len(set(meta_docs[id].get("modalities", [])).intersection(requested_modalities)),
            )
            for id in all_ids
        }
        train_sample_weights = [modality_counts[id] for id in train_ids]
        valid_sample_weights = [modality_counts[id] for id in valid_ids]
        test_sample_weights = [modality_counts[id] for id in test_ids]

        # get data for masks calculation
        if self.masked:
            print("Preparing SNIP mask data...")
            snip_batch_size = self._hparams["model"].get("snip_batch_size", 20)
            rng = random.Random(SEED) 
            snip_batch_ids = rng.sample(train_ids, len(train_ids))[:snip_batch_size]

            snip_data, snip_modalities, snip_labels = self.get_snip_data(posts_bin, posts_meta, snip_batch_ids)
            self.snip_data = (snip_data, snip_modalities, snip_labels)
            print(f"SNIP mask data prepared. Data shape: {snip_data.shape}, Modalities: {snip_modalities.shape}, Labels shape: {snip_labels.shape}")


        # save splits into logdir
        with open(os.path.join(self._logdir, 'train_ids.txt'), 'w') as f:
            for id in train_ids:
                f.write(f"{id}\n")
        with open(os.path.join(self._logdir, 'valid_ids.txt'), 'w') as f:
            for id in valid_ids:
                f.write(f"{id}\n")
        with open(os.path.join(self._logdir, 'test_ids.txt'), 'w') as f:
            for id in test_ids:
                f.write(f"{id}\n")


        usedDataset = MultimodalMongoDataset if self.multimodal else CustomMongoDataset #MM
        # Create dataloaders
        train_dataset = usedDataset(
            train_ids, 
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )
        
        # Use standard DBBatchSampler for mixed-modality batches (cross-modality competition)
        train_sampler = self._make_sampler(
            train_dataset,
            batch_size=self.num_volumes,
            seed=SEED,
            sample_weights=train_sample_weights,
        )
        
        train_dataloader = self._make_loader(
            train_dataset,
            train_sampler,
            self.funcs["createclient"],
            self.train_num_workers,
            self.train_prefetch_factor,
            self.train_prefetches,
            self.train_persistent_workers,
        )

        valid_dataset = usedDataset(
            valid_ids,#take first validation_percent percent from list
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )
        
        valid_sampler = self._make_sampler(
            valid_dataset,
            batch_size=self.num_volumes,
            seed=SEED,
            sample_weights=valid_sample_weights,
        )
        
        valid_dataloader = self._make_loader(
            valid_dataset,
            valid_sampler,
            self.funcs["createVclient"],
            self.eval_num_workers,
            self.eval_prefetch_factor,
            self.eval_prefetches,
            self.eval_persistent_workers,
        )

        loaders = {"train": train_dataloader, "valid": valid_dataloader}

        if self.run_infer_each_epoch:
            test_dataset = usedDataset(
                test_ids,#take first validation_percent percent from list
                self.funcs["mytransform"],
                None,
                self.db_fields,
                self.meta_fields,
                normalize=unit_interval_normalize,
                id=self.index_id,
            )
            test_sampler = self._make_sampler(
                test_dataset,
                batch_size=self.num_volumes,
                seed=SEED,
                sample_weights=test_sample_weights,
            )
            test_dataloader = self._make_loader(
                test_dataset,
                test_sampler,
                self.funcs["createVclient"],
                self.eval_num_workers,
                self.eval_prefetch_factor,
                self.eval_prefetches,
                self.eval_persistent_workers,
            )
            loaders["infer"] = test_dataloader

        return loaders

    def get_snip_data(self, posts_bin, posts_meta, snip_ids):
        snip_dict = {}

        # 1. Fetch all binary data for SNIP in one batch
        snip_samples = list(
            posts_bin.find(
                {
                    "id": {"$in": snip_ids},
                    "kind": {"$in": self.db_fields}, 
                },
                {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1},
            )
        )

        # Pre-group chunks by (id, kind) for O(N) access
        chunks_by_id_kind = {}
        for s in snip_samples:
            key = (s["id"], s["kind"])
            if key not in chunks_by_id_kind:
                chunks_by_id_kind[key] = []
            chunks_by_id_kind[key].append(s)

        # 2. Fetch all metadata for SNIP in one batch
        all_meta = list(
            posts_meta.find(
                {"id": {"$in": snip_ids}},
                list(self.meta_fields) + ["modalities", "id"],
            )
        )
        meta_lookup = {meta["id"]: meta for meta in all_meta}

        for id in snip_ids:
            # get ID's label and modalities
            meta_for_id = meta_lookup.get(id)
            if meta_for_id is None:
                continue

            label = meta_for_id[self.meta_fields[0]]
            modalities = meta_for_id["modalities"]
            id_modalities = set(modalities).intersection(set(self.db_fields))

            for mod in id_modalities:
                # Optimized: get pre-grouped chunks and sort them
                samples_for_id_kind = chunks_by_id_kind.get((id, mod), [])
                if not samples_for_id_kind:
                    continue
                
                samples_for_id_kind.sort(key=lambda x: x["chunk_id"])
                data = b"".join([s["chunk"] for s in samples_for_id_kind])

                result = {
                    "input": unit_interval_normalize(self.funcs["mytransform"](data).float()),
                    "modality": mod,
                    "label": torch.tensor(label).unsqueeze(0),
                }

                snip_dict[str(id)+'_'+mod] = result

        return multimodal_collate({0:snip_dict}) # dict is expected in collate

    def get_model(self):
        model = ResNet3D(
            in_channels=1, 
            n_classes=self.n_classes, 
            channels=self.n_channels
        )
        # if self.model_path and os.path.exists(self.model_path):
        #     model.load_state_dict(torch.load(self.model_path))

        if self.masked:
            print("Using MultiMaskSNIPWrapper for masked training")
            model = MultiMaskSNIPWrapper(
                model,
                sparsity=self._hparams["model"].get("sparsity", 0.9),
            )

            print("Initializing masks...")
            snip_data, snip_modalities, snip_labels = self.snip_data
            model.register_multimodal_masks(snip_modalities, snip_data, snip_labels)
            print("Masks initialized.")

        return model

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.onecycle_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.1,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {
            # "sync": False,
            "save_best": True,
            "metric_key": "loss",
            "loader_key": "valid",
            "minimize": True,
        }
        # checkpoint_params = {
        #     # "sync": False,
        #     "save_best": True,
        #     "metric_key": "accuracy",
        #     "loader_key": "valid",
        #     "minimize": False,
        # }
        if self.model_path:
            checkpoint_params.update({"resume_model": self.model_path})
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        metric_keys = ["loss", "accuracy", "learning rate"]
        if self.profile_timings:
            metric_keys.extend(self.timing_metric_keys)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in metric_keys
        }
        self.meters["auc"] = metrics.AUCMetric(
            compute_on_call=False
        )
        self._last_batch_end_time = None
        self._current_data_wait_h2d_sec = 0.0

        # --- CSV LOGGING SETUP ---
        rank = distributed.get_rank()
        loader_key = self.loader_key # e.g., "train", "valid"
        self.csv_filename = os.path.join(
            self._logdir, 
            f"raw_preds_{loader_key}_rank_{rank}.csv"
        )
        file_exists = os.path.isfile(self.csv_filename) and os.path.getsize(self.csv_filename) > 0

        self.csv_file = open(self.csv_filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header only if file is new
        if not file_exists:
            self.csv_writer.writerow(["epoch", "probability", "target"])

        if self.profile_timings:
            self.timing_csv_filename = os.path.join(
                self._logdir,
                f"batch_timing_{loader_key}_rank_{rank}.csv",
            )
            timing_file_exists = (
                os.path.isfile(self.timing_csv_filename)
                and os.path.getsize(self.timing_csv_filename) > 0
            )
            self.timing_csv_file = open(self.timing_csv_filename, 'a', newline='')
            self.timing_csv_writer = csv.writer(self.timing_csv_file)
            if not timing_file_exists:
                self.timing_csv_writer.writerow(
                    [
                        "epoch",
                        "batch",
                        "data_wait_h2d_sec",
                        "forward_sec",
                        "backward_sec",
                        "optimizer_sec",
                        "compute_sec",
                        "batch_total_sec",
                    ]
                )

    def on_batch_start(self, runner):
        parent = super()
        if hasattr(parent, "on_batch_start"):
            parent.on_batch_start(runner)
        if self.profile_timings:
            self._sync_timing()
            now = time.perf_counter()
            self._current_data_wait_h2d_sec = (
                0.0
                if self._last_batch_end_time is None
                else now - self._last_batch_end_time
            )

    def on_batch_end(self, runner):
        if self.profile_timings:
            self._sync_timing()
            self._last_batch_end_time = time.perf_counter()
        parent = super()
        if hasattr(parent, "on_batch_end"):
            parent.on_batch_end(runner)

    def on_loader_end(self, runner):
        metric_keys = ["loss", "accuracy", "learning rate"]
        if self.profile_timings:
            metric_keys.extend(self.timing_metric_keys)
        for key in metric_keys:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        self.loader_metrics["auc"] = self.meters["auc"].compute()[2]

        if self.engine.is_ddp:
            # Get world_size explicitly
            world_size = distributed.get_world_size()
            
            for key in ["loss", "accuracy"]:
                local_val = self.loader_metrics[key]
                
                # Create a tensor on the correct device
                # self.engine.device is reliable for the current worker's device
                val_tensor = torch.tensor([local_val], device=self.engine.device)
                
                # FIX: Pass world_size to mean_reduce
                avg_tensor = distributed.mean_reduce(val_tensor, world_size)
                self.loader_metrics[key] = avg_tensor.item()

        # CSV Safety Close
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        if hasattr(self, 'timing_csv_file') and self.timing_csv_file:
            self.timing_csv_file.close()

        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):

        # # Add synchronization before processing
        # if self.engine.is_ddp:
        #     torch.cuda.synchronize()
        
        if self.multimodal: #MM
            sample, modality, label = batch
        else:
            sample, label = batch

        timing = {}
        if self.profile_timings:
            self._sync_timing()
            compute_start = time.perf_counter()

        # run model forward/backward pass
        if self.model.training:
            if self.bit16:
                forward_start = time.perf_counter() if self.profile_timings else None
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                    loss = self.criterion(y_hat, label.float())
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/forward_sec"] = time.perf_counter() - forward_start

                backward_start = time.perf_counter() if self.profile_timings else None
                scaler.scale(loss).backward()
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/backward_sec"] = time.perf_counter() - backward_start

                optimizer_start = time.perf_counter() if self.profile_timings else None
                scaler.step(self.optimizer)
                self.scheduler.step()
                scaler.update()
                self.optimizer.zero_grad()
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/optimizer_sec"] = time.perf_counter() - optimizer_start
            else:
                forward_start = time.perf_counter() if self.profile_timings else None
                y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                loss = self.criterion(y_hat, label.float())
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/forward_sec"] = time.perf_counter() - forward_start

                backward_start = time.perf_counter() if self.profile_timings else None
                loss.backward()
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/backward_sec"] = time.perf_counter() - backward_start

                optimizer_start = time.perf_counter() if self.profile_timings else None
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.profile_timings:
                    self._sync_timing()
                    timing["time/optimizer_sec"] = time.perf_counter() - optimizer_start
        else:
            forward_start = time.perf_counter() if self.profile_timings else None
            with torch.no_grad():
                y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                loss = self.criterion(y_hat, label.float())
            if self.profile_timings:
                self._sync_timing()
                timing["time/forward_sec"] = time.perf_counter() - forward_start
                timing["time/backward_sec"] = 0.0
                timing["time/optimizer_sec"] = 0.0

        if self.profile_timings:
            self._sync_timing()
            timing["time/compute_sec"] = time.perf_counter() - compute_start
            timing["time/data_wait_h2d_sec"] = self._current_data_wait_h2d_sec
            timing["time/batch_total_sec"] = (
                timing["time/data_wait_h2d_sec"] + timing["time/compute_sec"]
            )

        # Metrics calculation and CSV logging
        with torch.no_grad():
            proba_preds = torch.sigmoid(y_hat)
            preds = proba_preds > 0.5
            accuracy = (preds == label).float().mean()
            
            # CSV logging: Move to CPU / Numpy
            probs_np = proba_preds.detach().cpu().numpy().flatten()
            targets_np = label.detach().cpu().numpy().flatten()
            epochs_np = [self.epoch_step] * len(probs_np)
            rows = zip(epochs_np, probs_np, targets_np)
            self.csv_writer.writerows(rows)


        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy, 
            "learning rate": torch.tensor(
                    self.optimizer.param_groups[0]["lr"]
            )
        })
        if self.profile_timings:
            self.batch_metrics.update(
                {
                    key: torch.tensor(value, device=loss.device)
                    for key, value in timing.items()
                }
            )
            self.timing_csv_writer.writerow(
                [
                    self.epoch_step,
                    getattr(self, "batch_step", ""),
                    timing["time/data_wait_h2d_sec"],
                    timing["time/forward_sec"],
                    timing["time/backward_sec"],
                    timing["time/optimizer_sec"],
                    timing["time/compute_sec"],
                    timing["time/batch_total_sec"],
                ]
            )
        for key in self.batch_metrics:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )
        self.meters["auc"].update(proba_preds, label)

        del sample
        del label
        del y_hat
        del loss

@hydra.main(config_path="conf", config_name="new_conf", version_base=None)
def main(cfg: DictConfig):
    # Loading common parameters
    # Model parameters
    n_classes = cfg.model.n_classes
    # config_file = cfg.model.config_file
    optimize_inline = cfg.model.optimize_inline
    model_channels = cfg.model.model_channels
    use_groupnorm = cfg.model.use_groupnorm
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    validation_percent = cfg.mongo.validation_percent
    wandb_project = cfg.wandb.project
    bit16 = cfg.bit16

    client_creator = ClientCreator(
        db_host, crop_tensor=cfg.client_creator.crop_tensor
    )

    # Evaluate the Python code from the YAML config
    experiment_name = cfg.experiment.experiment_name
    cubesizes = cfg.experiment.cubesizes
    numcubes = cfg.experiment.numcubes
    numvolumes = cfg.experiment.numvolumes
    weights = cfg.experiment.weights
    databases = cfg.experiment.databases
    collections = cfg.experiment.collections
    # dbfields = [tuple(fields) for fields in cfg.experiment.dbfields]  # Convert to tuples
    dbfields = tuple(cfg.experiment.dbfields)
    metafields = tuple(cfg.experiment.metafields)
    epochs = cfg.experiment.epochs
    prefetches = cfg.experiment.prefetches
    num_workers = cfg.experiment.num_workers
    prefetch_factor = cfg.experiment.prefetch_factor
    train_prefetches = cfg.experiment.get("train_prefetches", None)
    train_num_workers = cfg.experiment.get("train_num_workers", None)
    train_prefetch_factor = cfg.experiment.get("train_prefetch_factor", None)
    train_persistent_workers = cfg.experiment.get("train_persistent_workers", True)
    eval_prefetches = cfg.experiment.get("eval_prefetches", None)
    eval_num_workers = cfg.experiment.get("eval_num_workers", None)
    eval_prefetch_factor = cfg.experiment.get("eval_prefetch_factor", None)
    eval_persistent_workers = cfg.experiment.get("eval_persistent_workers", True)
    run_infer_each_epoch = cfg.experiment.get("run_infer_each_epoch", False)
    profile_timings = cfg.experiment.get("profile_timings", False)
    timing_sync_cuda = cfg.experiment.get("timing_sync_cuda", True)
    cudnn_benchmark = cfg.experiment.get("cudnn_benchmark", False)
    max_folds = cfg.experiment.get("max_folds", None)
    attenuates = cfg.experiment.attenuates
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    # we need oneCycleLR, but not the rest of the curiculum
    subvolume_shape = [cubesizes] * 3
    onecycle_lr = rmsprop_lr = (
        attenuates # this comes from 0.8/0.2 training? what is this input for oneCycleLR? TODO: trace it further
        * 1
        * cfg.experiment.lr_scale
        * numcubes
        * numvolumes
        / 256
    )
    wandb_experiment = (
        f"{experiment_name}: {collections}, {dbfields}-{metafields}, masked={cfg.model.get('masked', False)}, sps={cfg.model.get('sparsity', None)}"
    )

    # Set database parameters
    client_creator.set_database(databases)
    client_creator.set_collection(collections)
    client_creator.set_num_subcubes(numcubes)
    client_creator.set_shape(subvolume_shape)

    # paths:
    #     loadcheckpoint: False
    #     model: "../logs/tmp/new_test_fbirn_falff/model.last.pth"
    #     logdir: "./logs/tmp/new_test_fbirn_falff/"
    logdir = f"{cfg.paths.logdir}/{experiment_name}_{collections}_{dbfields}_{metafields}_masked_{cfg.model.get('masked', False)}_sps_{cfg.model.get('sparsity', None)}"
    os.makedirs(logdir, exist_ok=True)

    # Set hparams
    hparams = OmegaConf.to_container(cfg, resolve=True)

    folds_to_run = cfg.experiment.cv_folds if max_folds is None else min(cfg.experiment.cv_folds, int(max_folds))
    if folds_to_run < 1:
        raise ValueError("experiment.max_folds must be at least 1 when set")

    # run cross-validation
    for fold_idx in range(folds_to_run):

        print(f"Starting fold {fold_idx+1}/{cfg.experiment.cv_folds}")
        hparams["fold_idx"] = fold_idx

        rundir = f"{logdir}/fold_{fold_idx}"
        os.makedirs(rundir, exist_ok=True)

        runner = CustomRunner(
            logdir=rundir, # this is self._logdir
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            # modelconfig=config_file,
            n_epochs=epochs,
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            num_subcubes=numcubes,
            num_volumes=numvolumes,
            groupnorm=use_groupnorm,
            client_creator=client_creator,
            off_brain_weight=weights,
            prefetches=prefetches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            train_prefetches=train_prefetches,
            train_num_workers=train_num_workers,
            train_prefetch_factor=train_prefetch_factor,
            train_persistent_workers=train_persistent_workers,
            eval_prefetches=eval_prefetches,
            eval_num_workers=eval_num_workers,
            eval_prefetch_factor=eval_prefetch_factor,
            eval_persistent_workers=eval_persistent_workers,
            run_infer_each_epoch=run_infer_each_epoch,
            profile_timings=profile_timings,
            timing_sync_cuda=timing_sync_cuda,
            indexid=cfg.mongo.index_id,
            db_collection=collections,
            db_name=databases,
            db_fields=dbfields,
            meta_fields=metafields,
            subvolume_shape=subvolume_shape,
            lowprecision=bit16,
            lossweight = [w / sum(cfg.model.loss_weight) for w in cfg.model.loss_weight] if sum(cfg.model.loss_weight) != 0 else ValueError("The sum of loss weights cannot be zero."),
            db_host=db_host,
            wandb_team=cfg.wandb.team,
            maxshape=cfg.model.maxshape,
            hparams=hparams,
        )
        runner.run()


if __name__ == "__main__":
    main()
