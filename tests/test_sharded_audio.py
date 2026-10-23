"""Private S3 fixtures exercise actual WebDataset TAR decoding and rank ownership."""

import copy
import datetime
import hashlib
import io
import json
import multiprocessing
import struct
import tarfile
import wave
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("webdataset")

from src.data import sharded_audio_datamodule as streaming

ENDPOINT = "https://storage.example.invalid"


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.gets = []

    def get_object(self, *, Bucket, Key):
        self.gets.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key])}


def make_wav(values=range(32), rate=16000):
    out = io.BytesIO()
    values = list(values)
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{len(values)}h", *values))
    return out.getvalue()


def add_shard(store, source, split, identifiers, *, bad_manifest=False):
    files = {}
    rows = []
    for number, recording in enumerate(identifiers):
        folder = hashlib.sha256(recording.encode()).hexdigest()[:2]
        name = f"audio/{folder}/{recording}.wav"
        rows.append({"path": name, "recording_id": recording})
        # The first sample identifies the recording when cropping is disabled.
        files[name] = make_wav(range(int(recording) * 10, int(recording) * 10 + 32))
    manifests = {
        name: ("".join(json.dumps(row) + "\n" for row in rows) if name == split else "").encode()
        for name in streaming.SPLITS
    }
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as handle:
        # This order and hash-prefix layout match stage_audioset._pack.
        for name, data in [
            *((f"{name}.jsonl", raw) for name, raw in manifests.items()),
            ("preparation-report.json", b"{}"),
            *files.items(),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    raw = out.getvalue()
    key = f"cohort/{source}.tar"
    store.objects[key] = raw
    return {
        "source": source,
        "key": key,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "splits": {
            name: {
                "clips": len(rows) if name == split else 0,
                "manifest_sha256": "0" * 64
                if bad_manifest and name == split
                else hashlib.sha256(manifests[name]).hexdigest(),
            }
            for name in streaming.SPLITS
        },
    }


def make_index(store, *, train_shards=4, heldout=True):
    shards = [
        add_shard(store, f"unbal_train{i:03d}", "train", [f"{2 * i + 1:011d}", f"{2 * i + 2:011d}"])
        for i in range(train_shards)
    ]
    if heldout:
        shards.extend(
            add_shard(
                store,
                f"{source}{i:02d}",
                split,
                [f"{base + 2 * i:011d}", f"{base + 2 * i + 1:011d}", f"{base + 2 * i + 20:011d}"],
            )
            for source, split, base in [("bal_train", "val", 100), ("eval", "test", 200)]
            for i in range(2)
        )
    return {
        "schema": 1,
        "revision": streaming.REVISION,
        "status": "partial",
        "endpoint": ENDPOINT,
        "bucket": "private-bucket",
        "verified_sources": len(shards),
        "shards": shards,
        "split_counts": {
            name: sum(shard["splits"][name]["clips"] for shard in shards)
            for name in streaming.SPLITS
        },
    }


def module(tmp_path, monkeypatch, *, store=None, index=None, rank=0, world=1, **kwargs):
    store = store or FakeS3()
    index = index or make_index(store)
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    config = dict(
        index_path=str(path),
        cache_dir=str(tmp_path / "cache"),
        endpoint=ENDPOINT,
        allow_partial_index=True,
        num_samples=8,
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        pin_memory=False,
        cache_min_free_gib=0,
        cache_max_gib=0.1,
        shuffle_buffer=1,
        train_random_crop=False,
    )
    config.update(kwargs)
    result = streaming.ShardedAudioDataModule(**config)
    result.trainer = SimpleNamespace(world_size=world, global_rank=rank)
    result.setup("fit")
    return result, store


def test_real_webdataset_batches_train_val_test_and_private_reads(tmp_path, monkeypatch):
    data, store = module(tmp_path, monkeypatch)
    for loader in (data.train_dataloader(), data.val_dataloader(), data.test_dataloader()):
        iterator = iter(loader)
        batch = next(iterator)["waveform"]
        assert batch.shape == (2, 8)
        assert batch.dtype == torch.float32
        assert torch.isfinite(batch).all()
        del iterator
    assert all(bucket == "private-bucket" for bucket, _ in store.gets)
    assert all(path.suffix == ".tar" for path in (tmp_path / "cache").rglob("*.tar"))


@pytest.mark.parametrize("world,workers", [(2, 1), (2, 4), (4, 4)])
def test_rank_worker_partition_covers_each_source_once(world, workers):
    shards = [{"source": i} for i in range(870)]
    partitions = [
        streaming._assigned(shards, rank, world, worker, workers)
        for rank in range(world)
        for worker in range(workers)
    ]
    all_sources = [entry["source"] for partition in partitions for entry in partition]
    assert sorted(all_sources) == list(range(870))
    assert all(partitions)


def test_two_rank_streams_are_disjoint_and_batches_have_global_four(tmp_path, monkeypatch):
    store = FakeS3()
    index = make_index(store)
    batches = []
    for rank in range(2):
        data, _ = module(tmp_path, monkeypatch, store=store, index=index, rank=rank, world=2)
        iterator = iter(data.train_dataloader())
        batches.append(torch.cat([next(iterator)["waveform"] for _ in range(2)]))
        del iterator
        validation = list(data.val_dataloader())
        assert len(validation) == 1  # equal full batches; the third clip is an explicit tail.
    first, second = [set((batch[:, 0] * 32768).int().tolist()) for batch in batches]
    assert len(first) == len(second) == 4
    assert not first & second
    assert first | second == set(range(10, 81, 10))


def test_actual_stream_worker_partitions_are_disjoint(tmp_path, monkeypatch):
    store = FakeS3()
    index = make_index(store)
    populations = []
    for rank in range(2):
        data, _ = module(
            tmp_path,
            monkeypatch,
            store=store,
            index=index,
            rank=rank,
            world=2,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
        )
        for worker in range(2):
            monkeypatch.setattr(
                streaming,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(id=worker, num_workers=2),
            )
            iterator = iter(data.train_dataloader().dataset)
            populations.append(
                {round(next(iterator)["waveform"][0].item() * 32768) for _ in range(2)}
            )
            iterator.close()
    assert all(len(population) == 2 for population in populations)
    assert len(set.union(*populations)) == sum(map(len, populations)) == 8


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork fixture")
def test_real_dataloader_workers_do_not_repeat_each_others_shards(tmp_path, monkeypatch):
    data, _ = module(tmp_path, monkeypatch, num_workers=2, persistent_workers=False)
    loader = torch.utils.data.DataLoader(
        data.train_dataloader().dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="fork",
        drop_last=True,
    )
    iterator = iter(loader)
    try:
        examples = torch.cat([next(iterator)["waveform"] for _ in range(4)])
    finally:
        iterator._shutdown_workers()
    assert set((examples[:, 0] * 32768).int().tolist()) == set(range(10, 81, 10))
    assert len(list((tmp_path / "cache" / "train" / "rank-0").glob("worker-*"))) == 2


def _ddp_stream_worker(rank, configuration, objects, rendezvous, output):
    torch.set_num_threads(1)
    store = FakeS3()
    store.objects = objects
    streaming._s3_client = lambda endpoint: store
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        data = streaming.ShardedAudioDataModule(**configuration)
        data.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        data.setup("fit")
        iterator = iter(data.train_dataloader())
        recordings = torch.cat([next(iterator)["waveform"][:, 0] for _ in range(2)])
        del iterator
        gathered = [torch.empty_like(recordings) for _ in range(2)]
        torch.distributed.all_gather(gathered, recordings)
        assert len(set((torch.cat(gathered) * 32768).int().tolist())) == 8
        steps = 0
        for batch in data.val_dataloader():
            count = torch.tensor([batch["waveform"].shape[0]])
            torch.distributed.all_reduce(count)
            assert count.item() == 4
            steps += 1
        from pathlib import Path

        Path(output, f"rank-{rank}.json").write_text(json.dumps({"validation_steps": steps}))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_two_real_ddp_ranks_stream_disjoint_audio_and_equal_validation(tmp_path, monkeypatch):
    data, store = module(tmp_path, monkeypatch, world=2)
    torch.multiprocessing.spawn(
        _ddp_stream_worker,
        args=(
            dict(data.hparams),
            store.objects,
            f"file://{tmp_path / 'rendezvous'}",
            str(tmp_path),
        ),
        nprocs=2,
        join=True,
    )
    counts = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())["validation_steps"]
        for rank in range(2)
    ]
    assert counts == [1, 1]


def test_heldout_balances_clips_across_uneven_shard_assignment(tmp_path, monkeypatch):
    store = FakeS3()
    index = make_index(store, heldout=False)
    index["shards"].extend(
        add_shard(store, f"bal_train{i:02d}", "val", [f"{1000 + 5 * i + j:011d}" for j in range(5)])
        for i in range(10)
    )
    index["verified_sources"] = len(index["shards"])
    index["split_counts"]["val"] = 50
    populations = []
    for rank in range(4):
        data, _ = module(tmp_path, monkeypatch, store=store, index=index, rank=rank, world=4)
        batches = list(data.val_dataloader())
        assert len(batches) == 6
        populations.append(
            set((torch.cat([batch["waveform"] for batch in batches])[:, 0] * 32768).int().tolist())
        )
    assert sum(map(len, populations)) == len(set.union(*populations)) == 48
    # Fewer than global batch (8) omitted, even with ten shards across four ranks.
    assert set.union(*populations) == set(range(10000, 10480, 10))


def test_seeded_stream_is_reproducible_and_cycles_change_crop(tmp_path, monkeypatch):
    data, _ = module(tmp_path, monkeypatch, train_random_crop=True, seed=19)
    iterator = iter(data.train_dataloader())
    first = [next(iterator)["waveform"].clone() for _ in range(4)]
    second = [next(iterator)["waveform"].clone() for _ in range(4)]
    del iterator
    iterator = iter(data.train_dataloader())
    repeated = [next(iterator)["waveform"] for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, repeated, strict=True))
    assert not torch.equal(torch.cat(first), torch.cat(second))


def test_checksum_failure_never_exposes_audio_and_discards_partial(tmp_path):
    store = FakeS3()
    shard = add_shard(store, "unbal_train000", "train", ["00000000001"])
    store.objects[shard["key"]] = b"x" + store.objects[shard["key"]][1:]
    cache = streaming._ShardCache(tmp_path, store, "private", shard["bytes"], 0)
    with pytest.raises(ValueError, match="SHA256"):
        cache.get(shard)
    assert not list(tmp_path.glob("*.tar"))
    assert not list(tmp_path.glob("*.partial"))


def test_cache_evicts_before_download_and_rechecks_cached_digest(tmp_path):
    store = FakeS3()
    first = add_shard(store, "unbal_train000", "train", ["00000000001"])
    second = add_shard(store, "unbal_train001", "train", ["00000000002"])
    cache = streaming._ShardCache(tmp_path, store, "private", first["bytes"], 0)
    old = cache.get(first)
    new = cache.get(second)
    assert not old.exists() and new.exists()
    assert sum(path.stat().st_size for path in tmp_path.glob("*.tar")) <= first["bytes"]
    new.write_bytes(b"x" * second["bytes"])
    assert cache.get(second).read_bytes() == store.objects[second["key"]]
    assert len(store.gets) == 3


def test_interrupted_s3_body_restarts_object_with_bounded_retries(tmp_path, monkeypatch):
    store = FakeS3()
    shard = add_shard(store, "unbal_train000", "train", ["00000000001"])
    original = store.get_object
    attempts = []

    class Interrupted(io.BytesIO):
        def read(self, *args):
            if self.tell():
                raise ConnectionResetError("transient body interruption")
            return super().read(100)

    def flaky(**kwargs):
        attempts.append(True)
        result = original(**kwargs)
        if len(attempts) < 3:
            result["Body"] = Interrupted(store.objects[shard["key"]])
        return result

    monkeypatch.setattr(store, "get_object", flaky)
    monkeypatch.setattr(streaming.time, "sleep", lambda _: None)
    cache = streaming._ShardCache(tmp_path, store, "private", shard["bytes"], 0)
    assert cache.get(shard).read_bytes() == store.objects[shard["key"]]
    assert len(attempts) == 3
    assert not list(tmp_path.glob("*.partial"))


def test_partial_cohort_requires_explicit_flag_and_cannot_claim_complete():
    store = FakeS3()
    index = make_index(store)
    with pytest.raises(ValueError, match="requires a complete"):
        streaming._validate_index(index, ENDPOINT, False)
    index["status"] = "complete"
    with pytest.raises(ValueError, match="889"):
        streaming._validate_index(index, ENDPOINT, True)


def test_complete_index_requires_and_accepts_all_original_source_names():
    shards = []
    empty_digest = hashlib.sha256(b"").hexdigest()
    for source in streaming.builtin_sources():
        shards.append(
            {
                "source": source.name,
                "key": f"cohort/{source.name}.tar",
                "bytes": 10240,
                "sha256": hashlib.sha256(source.name.encode()).hexdigest(),
                "splits": {
                    name: {
                        "clips": 2 if name == source.split else 0,
                        "manifest_sha256": empty_digest,
                    }
                    for name in streaming.SPLITS
                },
            }
        )
    index = {
        "schema": 1,
        "revision": streaming.REVISION,
        "status": "complete",
        "endpoint": ENDPOINT,
        "bucket": "private",
        "verified_sources": 889,
        "expected_sources": 889,
        "full_cohort_sources": 889,
        "metadata": [{}, {}, {}],
        "shards": shards,
        "split_counts": {"train": 1740, "val": 20, "test": 18},
    }
    result = streaming._validate_index(index, ENDPOINT, False)
    assert {name: len(rows) for name, rows in result.items()} == {
        "train": 870,
        "val": 10,
        "test": 9,
    }


def test_wrong_source_split_and_duplicate_objects_rejected():
    store = FakeS3()
    index = make_index(store)
    original = copy.deepcopy(index)
    index["shards"][0]["splits"]["test"]["clips"] = 1
    with pytest.raises(ValueError, match="split counts"):
        streaming._validate_index(index, ENDPOINT, True)
    original["shards"][1]["key"] = original["shards"][0]["key"]
    with pytest.raises(ValueError, match="duplicate"):
        streaming._validate_index(original, ENDPOINT, True)


def test_bad_manifest_and_short_wrong_rate_truncated_audio_fail_closed(tmp_path):
    store = FakeS3()
    shard = add_shard(store, "unbal_train000", "train", ["00000000001"], bad_manifest=True)
    path = tmp_path / "shard.tar"
    path.write_bytes(store.objects[shard["key"]])
    with pytest.raises(ValueError, match="manifest SHA256"):
        list(streaming._audio_records(path, shard, "train"))
    for raw, expected in [
        (make_wav([1, 2]), "short"),
        (make_wav(rate=8000), "sample rate"),
        (make_wav()[:-2], "Truncated"),
    ]:
        with pytest.raises(ValueError, match=expected):
            streaming._decode(raw, "fixture.wav", 16000, 8, 0, 0, False)


def test_resume_requires_same_index_and_explicit_stream_restart(tmp_path, monkeypatch):
    data, _ = module(tmp_path, monkeypatch)
    state = data.state_dict()
    assert state["stream_position_checkpointed"] is False
    with pytest.raises(ValueError, match="positions are not checkpointed"):
        data.load_state_dict(state)
    data.hparams.allow_stream_restart_on_resume = True
    data.load_state_dict(state)
    with pytest.raises(ValueError, match="different staged data"):
        data.load_state_dict({"index_sha256": "different"})


def test_partial_train_only_pilot_and_worker_count_guard(tmp_path, monkeypatch):
    store = FakeS3()
    index = make_index(store, train_shards=1, heldout=False)
    data, _ = module(tmp_path, monkeypatch, store=store, index=index)
    assert data.val_dataloader() == []
    assert next(iter(data.train_dataloader()))["waveform"].shape == (2, 8)
    with pytest.raises(ValueError, match="one shard per rank/worker"):
        module(tmp_path, monkeypatch, store=store, index=index, world=2)


def test_s3_index_location_must_match_bucket(tmp_path, monkeypatch):
    store = FakeS3()
    store.objects["index.json"] = json.dumps(make_index(store)).encode()
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    value, digest = streaming._read_index("s3://private-bucket/index.json", ENDPOINT)
    assert value["bucket"] == "private-bucket" and len(digest) == 64
    with pytest.raises(ValueError, match="bucket differs"):
        streaming._read_index("s3://wrong-bucket/index.json", ENDPOINT)


def test_cache_rejects_topology_change_instead_of_silently_exceeding_budget(tmp_path, monkeypatch):
    store = FakeS3()
    index = make_index(store)
    module(tmp_path, monkeypatch, store=store, index=index)
    with pytest.raises(ValueError, match="another index/topology"):
        module(tmp_path, monkeypatch, store=store, index=index, world=2)


def test_actual_stager_archive_can_be_consumed_without_repacking(tmp_path, monkeypatch):
    # Reuse the staging integration fixture to verify the producer/consumer contract.
    from test_stage_audioset import FakeS3 as StageS3
    from test_stage_audioset import runner, source_archive

    store = StageS3()
    source = source_archive(tmp_path)
    index = runner(tmp_path, store, [source]).run(metadata=False)
    data, _ = module(
        tmp_path,
        monkeypatch,
        store=store,
        index=index,
        endpoint=index["endpoint"],
        num_samples=145440,
    )
    batch = next(iter(data.train_dataloader()))["waveform"]
    assert batch.shape == (2, 145440)
    assert torch.equal(batch, torch.full((2, 145440), 1 / 32768))
