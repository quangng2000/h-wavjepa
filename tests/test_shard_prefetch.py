"""One ordered lookahead overlaps real SDK I/O without changing data semantics."""

import contextlib
import json
import random
import select
import subprocess
import sys
import threading
import time

import psutil
import pytest
import torch
from test_bounded_download import gateway as gateway
from test_sharded_audio import FakeS3, add_shard, module

from src.data import sharded_audio_datamodule as streaming


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


def fixtures(gateway, tmp_path, *, capacity=2):
    client, state = gateway
    store = FakeS3()
    shards = [
        add_shard(store, f"unbal_train{i:03d}", "train", [f"{2 * i + 1:011d}", f"{2 * i + 2:011d}"])
        for i in range(4)
    ]
    for key, raw in store.objects.items():
        state["objects"][f"/private/{key}"] = raw
    cache = streaming._ShardCache(tmp_path, client, "private", capacity * shards[0]["bytes"], 0, 10)
    return state, store, shards, cache


def cached(cache, shard, store):
    path = cache.directory / f"{shard['sha256']}.tar"
    path.write_bytes(store.objects[shard["key"]])
    return path


def test_next_transfer_overlaps_current_reader_and_pins_cache(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path)
    current = cached(cache, shards[0], store)
    evictable = cached(cache, shards[2], store)
    generator = streaming._ordered_shards(cache, shards[:2], 1)
    with contextlib.closing(generator):
        shard, path = next(generator)
        assert shard == shards[0] and path == current
        with current.open("rb") as reader:
            following = cache.directory / f"{shards[1]['sha256']}.tar"
            # Keep the current reader open while the actual subprocess completes.
            wait_for(following.exists)
            assert current.exists() and not evictable.exists()
            assert reader.read() == store.objects[shards[0]["key"]]
            assert sum(p.stat().st_size for p in tmp_path.glob("*.tar")) <= cache.budget
        assert next(generator) == (shards[1], following)
        with pytest.raises(StopIteration):
            next(generator)
    assert state["peak"] >= 1
    assert not list(tmp_path.glob(".verified-download-*"))


def test_prefetch_waits_for_reader_when_only_one_shard_fits(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path, capacity=1)
    current = cached(cache, shards[0], store)
    with contextlib.closing(streaming._ordered_shards(cache, shards[:2], 1)) as iterator:
        assert next(iterator)[1] == current
        time.sleep(0.1)
        assert not state["keys"]  # Nothing was admitted before the reader finished.
        assert current.exists()
        _, following = next(iterator)
        assert following.exists() and not current.exists()
        assert sum(p.stat().st_size for p in tmp_path.glob("*.tar")) <= cache.budget


def test_prefetch_corruption_is_propagated_and_never_published(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path)
    current = cached(cache, shards[0], store)
    state["corrupt"] = True
    with contextlib.closing(streaming._ordered_shards(cache, shards[:2], 1)) as iterator:
        assert next(iterator)[1] == current
        with pytest.raises(RuntimeError, match="Managed object download failed"):
            next(iterator)
    assert current.exists()
    assert not (tmp_path / f"{shards[1]['sha256']}.tar").exists()
    assert not list(tmp_path.glob(".verified-download-*"))


def test_close_cancels_and_reaps_inflight_sdk_before_return(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path)
    cached(cache, shards[0], store)
    state["block"] = True
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    iterator = streaming._ordered_shards(cache, shards[:2], 1)
    next(iterator)
    wait_for(lambda: state["active"] > 0)
    owned = [p for p in psutil.Process().children(recursive=True) if p.pid not in before]
    assert owned
    started = time.monotonic()
    iterator.close()
    assert time.monotonic() - started < 5
    assert all(not p.is_running() for p in owned)
    assert not list(tmp_path.glob(".verified-download-*"))
    assert not any(t.name.startswith("audio-shard") for t in threading.enumerate())


def test_abrupt_worker_death_kills_background_thread_sdk_child(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path)
    cached(cache, shards[0], store)
    state["block"] = True
    source = tmp_path / "worker.py"
    source.write_text(
        "import boto3,time\nfrom pathlib import Path\n"
        "from src.data.sharded_audio_datamodule import _ShardCache,_ordered_shards\n"
        f"client=boto3.client('s3',endpoint_url={cache.client.meta.endpoint_url!r},region_name='auto')\n"
        f"cache=_ShardCache(Path({str(tmp_path)!r}),client,'private',{cache.budget},0,30)\n"
        f"iterator=_ordered_shards(cache,{shards[:2]!r},1)\n"
        "next(iterator)\ntime.sleep(30)\n"
    )
    # -c preserves the repository import root without changing any scoped env.
    process = subprocess.Popen(
        [sys.executable, "-c", source.read_text()],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = None
    try:
        deadline = time.monotonic() + 12
        while child is None and time.monotonic() < deadline:
            if select.select([process.stdout], [], [], 0.1)[0]:
                line = process.stdout.readline()
                if line:
                    event = json.loads(line)
                    if event.get("phase") == "started":
                        child = psutil.Process(event["pid"])
        assert child is not None
        process.terminate()  # No Python finally: exercise creator-thread death protection.
        process.wait(timeout=3)
        wait_for(lambda: not child.is_running() or child.status() == psutil.STATUS_ZOMBIE, 3)
        assert not (tmp_path / f"{shards[1]['sha256']}.tar").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        process.stdout.close()
        if child is not None and child.is_running():
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()


def test_existing_scratch_counts_against_prefetch_cache_budget(gateway, tmp_path):
    state, store, shards, cache = fixtures(gateway, tmp_path)
    cached(cache, shards[0], store)
    scratch = tmp_path / ".verified-download-retained"
    scratch.mkdir()
    (scratch / "object.partial").write_bytes(b"x" * shards[0]["bytes"])
    with contextlib.closing(streaming._ordered_shards(cache, shards[:2], 1)) as iterator:
        next(iterator)
        with pytest.raises(OSError, match="cache budget"):
            next(iterator)
    assert not state["keys"]
    assert scratch.exists()  # Never delete a possible writer's retained scratch.


def stream_config(client, directory, prefetch):
    return {
        "cache_dir": str(directory),
        "endpoint": client.meta.endpoint_url,
        "bucket": "private",
        "cache_min_free_gib": 0,
        "download_timeout_seconds": 10,
        "sample_rate": 16000,
        "num_samples": 8,
        "seed": 117,
        "shuffle_buffer": 3,
        "train_random_crop": True,
        "masked_crops_per_recording": 2,
        "masked_crop_samples": 8,
        "clean_binaural": True,
        "shard_prefetch": prefetch,
    }


@pytest.mark.parametrize("workers", [0, 1])
def test_real_stream_preserves_order_crops_rng_and_rank_ownership(gateway, tmp_path, workers):
    _, _, shards, cache = fixtures(gateway, tmp_path / "unused")
    outputs = []
    for prefetch in (0, 1):
        config = stream_config(cache.client, tmp_path / f"prefetch-{prefetch}", prefetch)
        dataset = streaming._AudioStream(
            shards=shards,
            split="train",
            rank=1,
            world_size=2,
            config=config,
            cache_budget=cache.budget,
            quota=8,
        )
        if workers:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=1,
                multiprocessing_context="spawn",
                timeout=30,
                generator=torch.Generator().manual_seed(9),
            )
            outputs.append(list(loader))
        else:
            rng_before, torch_before = random.getstate(), torch.get_rng_state()
            outputs.append(list(dataset))
            assert random.getstate() == rng_before
            assert torch.equal(torch_before, torch.get_rng_state())
        present = {p.stem for p in (tmp_path / f"prefetch-{prefetch}").rglob("*.tar")}
        assert present <= {shards[1]["sha256"], shards[3]["sha256"]}
    assert len(outputs[0]) == len(outputs[1]) == 8
    for expected, actual in zip(*outputs, strict=True):
        assert set(actual) == set(expected) == {"waveform", "masked_crops"}
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_prefetch_is_runtime_only_and_fake_stores_remain_synchronous(tmp_path, monkeypatch):
    model, store = module(tmp_path, monkeypatch, shard_prefetch=0)
    state = model.state_dict()
    model.hparams.shard_prefetch = 1
    assert model.state_dict() == state
    assert streaming.data_state_sha256(model.state_dict()) == streaming.data_state_sha256(state)
    loader = iter(model.train_dataloader())
    try:
        assert next(loader)["waveform"].shape == (2, 8)
    finally:
        loader._dataset_fetcher.dataset_iter.close()
    assert store.gets
    assert not any(t.name.startswith("audio-shard") for t in threading.enumerate())


@pytest.mark.parametrize("value", [True, -1, 2, 1.0, "1"])
def test_prefetch_option_rejects_unbounded_or_ambiguous_values(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="shard_prefetch"):
        module(tmp_path, monkeypatch, shard_prefetch=value)
