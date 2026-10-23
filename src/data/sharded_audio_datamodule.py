"""Verified AudioSet shards with WebDataset iteration and bounded private S3 caching.

Shard ownership is fixed across ranks/workers. Each worker shuffles and cycles its
own shards, so training has no finite epoch and is controlled by ``max_steps``.
Iterator positions are not checkpointed; resuming requires explicit acknowledgement.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
import wave
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.data.wav_datamodule import _positive_integer, _validate_wav
from src.prepare_audioset import REVISION, builtin_sources

SPLITS = ("train", "val", "test")
GIB = 1024**3
SHA = re.compile(r"[a-f0-9]{64}")
VIDEO = re.compile(r"[A-Za-z0-9_-]{11}")


def _s3_client(endpoint: str):
    # boto3's standard credential chain reads the job's bucket-scoped credentials.
    # Never put credentials in Hydra config, checkpoint state, URLs, or logs.
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "auto"),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"mode": "standard", "max_attempts": 4},
            connect_timeout=20,
            read_timeout=60,
        ),
    )


def _safe_key(value: str) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and ".." not in PurePosixPath(value).parts
        and "\\" not in value
    )


def _read_index(index_path: str, endpoint: str) -> tuple[dict, str]:
    parsed = urlparse(index_path)
    if parsed.scheme == "s3":
        if not parsed.netloc or not _safe_key(parsed.path.lstrip("/")) or parsed.query:
            raise ValueError("Expected s3://bucket/key without credentials or query parameters")
        response = _s3_client(endpoint).get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
        )
        with contextlib.closing(response["Body"]) as body:
            raw = body.read(16 * 1024**2 + 1)
    elif not parsed.scheme:
        with Path(index_path).expanduser().open("rb") as handle:
            raw = handle.read(16 * 1024**2 + 1)
    else:
        raise ValueError("index_path must be a local JSON file or private s3://bucket/key")
    if len(raw) > 16 * 1024**2:
        raise ValueError("Staging index exceeds 16 MiB")
    value = json.loads(raw)
    if parsed.scheme == "s3" and value.get("bucket") != parsed.netloc:
        raise ValueError("Staging index bucket differs from its S3 location")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_index(index: dict, endpoint: str, allow_partial: bool) -> dict[str, list[dict]]:
    expected = {source.name: source.split for source in builtin_sources()}
    if (
        index.get("schema") != 1
        or index.get("revision") != REVISION
        or index.get("endpoint") != endpoint
        or not isinstance(index.get("bucket"), str)
        or not index["bucket"]
    ):
        raise ValueError("Staging index does not match the pinned AudioSet/S3 contract")
    if index.get("status") != "complete" and not allow_partial:
        raise ValueError("Production training requires a complete staging index")
    if index.get("status") not in {"complete", "partial"}:
        raise ValueError("Unexpected staging index status")
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Staging index must contain verified shards")
    seen_sources, seen_keys, seen_digests = set(), set(), set()
    result = {split: [] for split in SPLITS}
    totals = {split: 0 for split in SPLITS}
    for shard in shards:
        source = shard.get("source")
        key, digest, size = shard.get("key"), shard.get("sha256"), shard.get("bytes")
        if (
            source not in expected
            or source in seen_sources
            or not _safe_key(key)
            or key in seen_keys
            or not isinstance(digest, str)
            or not SHA.fullmatch(digest)
            or digest in seen_digests
            or type(size) is not int
            or not 0 < size < 4 * GIB
        ):
            raise ValueError("Invalid or duplicate source/object in staging index")
        seen_sources.add(source)
        seen_keys.add(key)
        seen_digests.add(digest)
        for split in SPLITS:
            info = shard.get("splits", {}).get(split, {})
            count = info.get("clips")
            if (
                type(count) is not int
                or count < 0
                or not isinstance(info.get("manifest_sha256"), str)
                or not SHA.fullmatch(info["manifest_sha256"])
                or (split != expected[source] and count != 0)
            ):
                raise ValueError("Shard split counts or manifest checksums are invalid")
            totals[split] += count
        if shard["splits"][expected[source]]["clips"]:
            result[expected[source]].append(shard)
    if index.get("verified_sources") != len(shards) or index.get("split_counts") != totals:
        raise ValueError("Staging index totals do not match its shards")
    if index["status"] == "complete" and (
        seen_sources != set(expected)
        or index.get("expected_sources") != len(expected)
        or index.get("full_cohort_sources") != len(expected)
        or len(index.get("metadata", [])) != 3
        or not all(totals.values())
    ):
        raise ValueError("Complete index must contain all 889 pinned sources and held-out splits")
    return {
        split: sorted(items, key=lambda item: item["source"]) for split, items in result.items()
    }


def _assigned(shards: list[dict], rank: int, world_size: int, worker: int = 0, workers: int = 1):
    """Two explicit partitions keep every source owned by exactly one consumer."""
    return shards[rank::world_size][worker::workers]


def _heldout_ranges(shards: list[dict], split: str, rank: int, quota: int):
    """Contiguous clip ranges balance ranks even when shard sizes differ."""
    start, stop, offset = rank * quota, (rank + 1) * quota, 0
    selected = []
    for shard in shards:
        count = shard["splits"][split]["clips"]
        lower, upper = max(0, start - offset), min(count, stop - offset)
        if lower < upper:
            selected.append({**shard, "sample_start": lower, "sample_stop": upper})
        offset += count
    return selected


class _ShardCache:
    """One consumer owns one bounded directory, including its in-flight download."""

    def __init__(self, directory: Path, client, bucket: str, budget: int, minimum_free: int):
        self.directory, self.client, self.bucket = directory, client, bucket
        self.budget, self.minimum_free = budget, minimum_free
        directory.mkdir(parents=True, exist_ok=True)

    def get(self, shard: dict) -> Path:
        path = self.directory / f"{shard['sha256']}.tar"
        if path.is_symlink():
            raise ValueError("Cache objects must not be symlinks")
        if path.exists():
            with path.open("rb") as handle:
                valid = path.stat().st_size == shard["bytes"] and (
                    hashlib.file_digest(handle, "sha256").hexdigest() == shard["sha256"]
                )
            if valid:
                path.touch()
                return path
            path.unlink()
        if shard["bytes"] > self.budget:
            raise ValueError("A shard exceeds this consumer's cache allocation")
        cached = sorted(self.directory.glob("*.tar"), key=lambda item: item.stat().st_mtime)
        used = sum(item.stat().st_size for item in cached)
        while cached and used + shard["bytes"] > self.budget:
            oldest = cached.pop(0)
            used -= oldest.stat().st_size
            oldest.unlink()
        if shutil.disk_usage(self.directory).free < shard["bytes"] + self.minimum_free:
            raise OSError("Insufficient disk space for the next verified audio shard")
        temporary = path.with_suffix(".partial")
        temporary.unlink(missing_ok=True)
        from botocore.exceptions import (
            IncompleteReadError,
            ReadTimeoutError,
            ResponseStreamingError,
        )

        # botocore retries failed requests, but not interrupted StreamingBody reads.
        # Restart those whole objects; never retain a partially verified download.
        for attempt in range(3):
            digest, size = hashlib.sha256(), 0
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=shard["key"])
                with contextlib.closing(response["Body"]) as body, temporary.open("xb") as output:
                    for chunk in iter(lambda: body.read(1024**2), b""):
                        size += len(chunk)
                        if size > shard["bytes"]:
                            raise ValueError("Audio shard exceeded its indexed byte size")
                        digest.update(chunk)
                        output.write(chunk)
                if size != shard["bytes"] or digest.hexdigest() != shard["sha256"]:
                    raise ValueError("Audio shard byte size or SHA256 mismatch")
                temporary.replace(path)
                return path
            except (
                ConnectionResetError,
                TimeoutError,
                IncompleteReadError,
                ReadTimeoutError,
                ResponseStreamingError,
            ):
                if attempt == 2:
                    raise
                time.sleep(0.25 * 2**attempt)
            finally:
                temporary.unlink(missing_ok=True)


def _manifest(raw: bytes, split: str, expected: dict) -> set[str]:
    if hashlib.sha256(raw).hexdigest() != expected["manifest_sha256"]:
        raise ValueError("Shard manifest SHA256 mismatch")
    paths = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = item.get("path")
        recording = item.get("recording_id")
        if (
            not isinstance(recording, str)
            or not VIDEO.fullmatch(recording)
            or name != f"audio/{hashlib.sha256(recording.encode()).hexdigest()[:2]}/{recording}.wav"
            or name in paths
            or set(item) - {"path", "recording_id"}
        ):
            raise ValueError("Unsafe, duplicate, or unsupported staged manifest row")
        paths.add(name)
    if len(paths) != expected["clips"]:
        raise ValueError("Shard manifest clip count differs from the index")
    return paths


def _audio_records(path: Path, shard: dict, split: str):
    from webdataset.tariterators import tar_file_iterator

    expected = None
    seen = set()
    with path.open("rb") as handle:
        for item in tar_file_iterator(
            handle,
            select_files=lambda name: name == f"{split}.jsonl" or name.startswith("audio/"),
        ):
            name, raw = item["fname"], item["data"]
            if name == f"{split}.jsonl":
                if expected is not None:
                    raise ValueError("Duplicate shard manifest")
                expected = _manifest(raw, split, shard["splits"][split])
            else:
                if expected is None or name not in expected or name in seen:
                    raise ValueError(
                        "Audio member is missing from its split manifest or duplicated"
                    )
                seen.add(name)
                yield name, raw
    if expected is None or seen != expected:
        raise ValueError("Audio shard is missing indexed recordings")


def _decode(
    raw: bytes,
    name: str,
    sample_rate: int,
    num_samples: int,
    seed: int,
    cycle: int,
    random_crop: bool,
):
    if len(raw) > 2 * sample_rate * 30 + 65536:
        raise ValueError("Audio member exceeds the 30 second preparation contract")
    with wave.open(io.BytesIO(raw), "rb") as handle:
        _validate_wav(handle, Path(name), sample_rate)
        frames = handle.getnframes()
        if frames > sample_rate * 30:
            raise ValueError("Audio exceeds the 30 second preparation contract")
        handle.setpos(frames - 1)
        if len(handle.readframes(2)) != 2:
            raise ValueError("Truncated PCM audio")
        slack = frames - num_samples
        if slack < 0:
            raise ValueError("Forecasting requires real audio; short clips cannot be padded")
        offset = 0
        if random_crop and slack:
            digest = hashlib.blake2b(f"{seed}:{cycle}:{name}".encode(), digest_size=8).digest()
            offset = int.from_bytes(digest, "little") % (slack + 1)
        handle.setpos(offset)
        pcm = handle.readframes(num_samples)
        if len(pcm) != num_samples * 2:
            raise ValueError("Truncated PCM audio")
    if sys.byteorder != "little":
        raise RuntimeError("The streaming PCM decoder requires a little-endian host")
    return {"waveform": torch.frombuffer(bytearray(pcm), dtype=torch.int16).float().div_(32768.0)}


class _AudioStream(IterableDataset):
    def __init__(self, *, shards, split, rank, world_size, config, cache_budget, quota=None):
        super().__init__()
        self.shards, self.split, self.rank, self.world_size = shards, split, rank, world_size
        self.config, self.cache_budget, self.quota = config, cache_budget, quota

    def __iter__(self):
        from webdataset.filters import shuffle

        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        owned = (
            _assigned(self.shards, self.rank, self.world_size, worker, workers)
            if self.split == "train"
            else self.shards
        )
        if not owned:
            raise ValueError("Every streaming rank/worker needs at least one nonempty shard")
        directory = (
            Path(self.config["cache_dir"]) / self.split / f"rank-{self.rank}" / f"worker-{worker}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".reader.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "This shard-cache consumer already has an active reader"
                ) from error
            cache = _ShardCache(
                directory,
                _s3_client(self.config["endpoint"]),
                self.config["bucket"],
                self.cache_budget,
                self.config["cache_min_free_gib"] * GIB,
            )
            for stale in directory.glob("*.partial"):
                stale.unlink()
            cycle, emitted = 0, 0
            while True:
                rng = random.Random(f"{self.config['seed']}:{self.rank}:{worker}:{cycle}")
                order = list(owned)
                if self.split == "train":
                    rng.shuffle(order)

                def records():
                    for shard in order:
                        with contextlib.closing(
                            _audio_records(cache.get(shard), shard, self.split)
                        ) as items:
                            if self.split == "train":
                                yield from items
                            else:
                                yield from itertools.islice(
                                    items, shard["sample_start"], shard["sample_stop"]
                                )

                stream = records()
                if self.split == "train":
                    stream = shuffle(self.config["shuffle_buffer"], rng=rng)(stream)
                for name, raw in stream:
                    yield _decode(
                        raw,
                        name,
                        self.config["sample_rate"],
                        self.config["num_samples"],
                        self.config["seed"],
                        cycle,
                        self.split == "train" and self.config["train_random_crop"],
                    )
                    emitted += 1
                    if self.quota is not None and emitted >= self.quota:
                        return
                if self.split != "train":
                    return
                cycle += 1


class ShardedAudioDataModule(L.LightningDataModule):
    """Train from immutable staged WAV tar files without materializing the corpus.

    ``batch_size`` is per rank. No accumulation is needed to obtain global 64:
    use 32 on two ranks or 16 on four. Validation/test use equally many full
    batches on every rank; fewer than one global batch is excluded from objective metrics.
    """

    def __init__(
        self,
        index_path: str,
        cache_dir: str,
        endpoint: str = "https://storage.givemeanode.com",
        sample_rate: int = 16000,
        num_samples: int = 145440,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        minimum_batch_size: int = 2,
        require_full_length: bool = True,
        train_random_crop: bool = True,
        train_drop_last: bool = True,
        seed: int = 0,
        cache_max_gib: float = 40,
        cache_min_free_gib: float = 2,
        shuffle_buffer: int = 128,
        allow_partial_index: bool = False,
        allow_stream_restart_on_resume: bool = False,
    ):
        super().__init__()
        for key, value in {
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "minimum_batch_size": minimum_batch_size,
            "shuffle_buffer": shuffle_buffer,
        }.items():
            _positive_integer(value, key)
        _positive_integer(num_workers, "num_workers", minimum=0)
        _positive_integer(seed, "seed", minimum=0)
        if prefetch_factor is not None:
            _positive_integer(prefetch_factor, "prefetch_factor")
        if num_workers == 0 and (persistent_workers or prefetch_factor is not None):
            raise ValueError("persistent_workers and prefetch_factor require num_workers > 0")
        if sample_rate != 16000 or not require_full_length or not train_drop_last:
            raise ValueError("Staged AudioSet requires 16 kHz real audio and full training batches")
        if batch_size < minimum_batch_size:
            raise ValueError("batch_size must be >= minimum_batch_size")
        if not 0 < cache_max_gib <= 60 or not 0 <= cache_min_free_gib:
            raise ValueError("cache_max_gib must be in (0, 60]; cache_min_free_gib must be >= 0")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.query:
            raise ValueError("S3 endpoint must be an HTTPS origin without credentials")
        for name, value in {
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "require_full_length": require_full_length,
            "train_random_crop": train_random_crop,
            "train_drop_last": train_drop_last,
            "allow_partial_index": allow_partial_index,
            "allow_stream_restart_on_resume": allow_stream_restart_on_resume,
        }.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        self.save_hyperparameters(logger=False)
        self.index = None
        self.index_sha256 = None
        self.shards = None
        self._restored_state = None
        self.evaluation_coverage = {}

    def setup(self, stage=None):
        if self.index is None:
            self.index, self.index_sha256 = _read_index(
                self.hparams.index_path, self.hparams.endpoint
            )
            self.shards = _validate_index(
                self.index, self.hparams.endpoint, self.hparams.allow_partial_index
            )
        world = self.trainer.world_size if self.trainer else 1
        rank = self.trainer.global_rank if self.trainer else 0
        self.world_size, self.rank = world, rank
        global_batch = world * self.hparams.batch_size
        self.evaluation_coverage = {
            split: {
                "available": self.index["split_counts"][split],
                "used": self.index["split_counts"][split] // global_batch * global_batch,
                "omitted": self.index["split_counts"][split] % global_batch,
            }
            for split in ("val", "test")
        }
        consumers = world * (max(1, self.hparams.num_workers) + 2)
        self.cache_budget = int(self.hparams.cache_max_gib * GIB) // consumers
        largest = max(shard["bytes"] for split in SPLITS for shard in self.shards[split])
        if largest > self.cache_budget:
            raise ValueError(
                "Cache budget cannot hold one shard per concurrent consumer; reduce workers"
            )
        if stage in (None, "fit") and len(self.shards["train"]) < world * max(
            1, self.hparams.num_workers
        ):
            raise ValueError(
                "Training needs at least one shard per rank/worker; reduce workers for pilots"
            )
        contract = {
            "index_sha256": self.index_sha256,
            "world_size": world,
            "num_workers": self.hparams.num_workers,
            "cache_max_gib": self.hparams.cache_max_gib,
        }
        directory = Path(self.hparams.cache_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".contract.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = directory / "contract.json"
            if marker.exists() and json.loads(marker.read_text()) != contract:
                raise ValueError(
                    "Cache belongs to another index/topology; use a fresh empty cache directory"
                )
            if not marker.exists():
                if any(path.name != ".contract.lock" for path in directory.iterdir()):
                    raise ValueError("Refusing to adopt a nonempty unowned cache directory")
                marker.write_text(json.dumps(contract))
        self.cache_dir = str(directory)
        if self._restored_state is not None:
            self._check_resume(self._restored_state)

    def state_dict(self):
        return {
            "index_sha256": self.index_sha256,
            "stream_position_checkpointed": False,
            "evaluation_coverage": self.evaluation_coverage,
        }

    def _check_resume(self, state):
        if state.get("index_sha256") != self.index_sha256:
            raise ValueError("Checkpoint was trained with a different staged data index")
        if not self.hparams.allow_stream_restart_on_resume:
            raise ValueError(
                "Streaming positions are not checkpointed. Set allow_stream_restart_on_resume=true "
                "to explicitly restart each rank/worker's data sequence while restoring optimizer state."
            )

    def load_state_dict(self, state_dict):
        self._restored_state = state_dict
        if self.index is not None:
            self._check_resume(state_dict)

    def _loader(self, split):
        if self.shards is None:
            raise RuntimeError("Call setup before requesting a data loader")
        if not self.shards[split]:
            if split == "val" and self.hparams.allow_partial_index:
                return []
            raise ValueError(f"No {split} shards in this index")
        quota = None
        shards = self.shards[split]
        if split != "train":
            total = sum(shard["splits"][split]["clips"] for shard in shards)
            quota = total // (self.world_size * self.hparams.batch_size) * self.hparams.batch_size
            if quota < self.hparams.minimum_batch_size:
                raise ValueError(f"Every {split} rank needs at least one complete batch")
            shards = _heldout_ranges(shards, split, self.rank, quota)
        config = dict(self.hparams)
        config.update(bucket=self.index["bucket"], cache_dir=self.cache_dir)
        dataset = _AudioStream(
            shards=shards,
            split=split,
            rank=self.rank,
            world_size=self.world_size,
            config=config,
            cache_budget=self.cache_budget,
            quota=quota,
        )
        workers = self.hparams.num_workers if split == "train" else 0
        kwargs = {}
        if workers:
            kwargs.update(persistent_workers=self.hparams.persistent_workers)
            if self.hparams.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.hparams.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            **kwargs,
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
