"""Bounded, read-only private-S3 input probe; never rents or modifies cloud resources.

Uses the production streaming dataset/cache and one CPU data-loader reader. The
index is fetched once and frozen for this run. Default bounds: two training shards,
1.5 GiB of temporary cache, 4096 decoded clips, and 180 seconds. The owned temporary
directory is removed on normal completion, errors, SIGTERM, or the internal deadline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProbeLimit(BaseException):
    # SDK HTTP adapters catch Exception and may retry it. A hard cancellation
    # must escape those handlers so this one-shot deadline cannot be swallowed.
    pass


@contextlib.contextmanager
def deadline(seconds):
    """SIGALRM aborts blocked reads too; our error is not a retryable socket timeout."""

    def expire(signum, frame):
        raise ProbeLimit("Read-only input probe reached its time limit")

    old_alarm = signal.signal(signal.SIGALRM, expire)
    old_term = signal.signal(signal.SIGTERM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)


def select_shards(shards, max_shards, max_bytes):
    if type(max_shards) is not int or not 1 <= max_shards <= 2:
        raise ValueError("max_shards must be 1 or 2")
    if type(max_bytes) is not int or not 0 < max_bytes <= int(1.5 * 1024**3):
        raise ValueError("Local cache budget must be positive and at most 1.5 GiB")
    selected, used = [], 0
    for shard in shards:
        if len(selected) == max_shards:
            break
        if used + shard["bytes"] <= max_bytes:
            selected.append(shard)
            used += shard["bytes"]
    if not selected:
        raise ValueError("No training shard fits the bounded probe cache")
    return selected


class TimedBody:
    def __init__(self, body, metrics):
        self.body, self.metrics = body, metrics

    def read(self, size=-1):
        started = time.perf_counter()
        try:
            data = self.body.read(size)
            self.metrics["s3_body_bytes"] += len(data)
            return data
        finally:
            self.metrics["s3_body_read_seconds"] += time.perf_counter() - started

    def close(self):
        self.body.close()


class TimedS3:
    def __init__(self, client, metrics):
        self.client, self.metrics = client, metrics

    def get_object(self, **kwargs):
        self.metrics["s3_get_object_calls"] += 1
        started = time.perf_counter()
        try:
            response = self.client.get_object(**kwargs)
        finally:
            self.metrics["s3_get_response_seconds"] += time.perf_counter() - started
        return {**response, "Body": TimedBody(response["Body"], self.metrics)}


def run_probe(
    *,
    index_path,
    endpoint,
    cache_parent,
    allow_partial=False,
    max_shards=2,
    max_bytes=int(1.5 * 1024**3),
    max_samples=4096,
    report=None,
):
    """No remote writes; instrument production functions within this process only."""
    from src.data import sharded_audio_datamodule as production

    if type(max_samples) is not int or not 1 <= max_samples <= 4096:
        raise ValueError("max_samples must be in [1, 4096]")
    result = report if report is not None else {}
    result.update(
        status="running",
        probe="production_sharded_audio_single_cpu_reader",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    started = time.perf_counter()
    index, index_sha = production._read_index(index_path, endpoint)
    result["index_fetch_seconds"] = time.perf_counter() - started
    splits = production._validate_index(index, endpoint, allow_partial)
    selected = select_shards(splits["train"], max_shards, max_bytes)
    population = sum(shard["splits"]["train"]["clips"] for shard in selected)
    quota = min(population, max_samples)
    result.update(
        frozen_index_sha256=index_sha,
        frozen_index_status=index["status"],
        available_verified_sources=index["verified_sources"],
        available_training_shards=len(splits["train"]),
        selected_source_count=len(selected),
        selected_payload_bytes=sum(shard["bytes"] for shard in selected),
        selected_clips=population,
        sample_limit=quota,
        cache_limit_bytes=max_bytes,
        selected_sources=[
            {"source": shard["source"], "bytes": shard["bytes"], "sha256": shard["sha256"]}
            for shard in selected
        ],
    )
    metrics = {
        "s3_get_object_calls": 0,
        "s3_body_bytes": 0,
        "s3_get_response_seconds": 0.0,
        "s3_body_read_seconds": 0.0,
        "cache_get_and_sha256_verify_seconds": 0.0,
        "tar_iteration_seconds": 0.0,
        "pcm_decode_seconds": 0.0,
        "clips": 0,
        "batches": 0,
        "cache_peak_observed_bytes": 0,
    }
    result["measurements"] = metrics
    original_client = production._s3_client
    original_cache = production._ShardCache
    original_records = production._audio_records
    original_decode = production._decode

    class TimedCache(original_cache):
        def get(self, shard):
            begun = time.perf_counter()
            try:
                return super().get(shard)
            finally:
                metrics["cache_get_and_sha256_verify_seconds"] += time.perf_counter() - begun
                size = sum(path.stat().st_size for path in self.directory.glob("*.tar"))
                metrics["cache_peak_observed_bytes"] = max(
                    metrics["cache_peak_observed_bytes"], size
                )
                if size > max_bytes:
                    raise ProbeLimit("Owned temporary cache exceeded its byte limit")

    def timed_records(*args):
        with contextlib.closing(original_records(*args)) as iterator:
            while True:
                begun = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                finally:
                    metrics["tar_iteration_seconds"] += time.perf_counter() - begun
                yield item

    def timed_decode(*args, **kwargs):
        begun = time.perf_counter()
        try:
            return original_decode(*args, **kwargs)
        finally:
            metrics["pcm_decode_seconds"] += time.perf_counter() - begun

    cache_parent = Path(cache_parent).expanduser().resolve()
    cache_parent.mkdir(parents=True, exist_ok=True)
    cache_location = None
    try:
        with tempfile.TemporaryDirectory(prefix="hwavjepa-read-probe-", dir=cache_parent) as folder:
            cache_location = Path(folder)
            config = {
                "endpoint": endpoint,
                "bucket": index["bucket"],
                "cache_dir": folder,
                "cache_min_free_gib": 1,
                "sample_rate": 16000,
                "num_samples": 145440,
                "seed": 42,
                "shuffle_buffer": 128,
                "train_random_crop": True,
            }
            dataset = production._AudioStream(
                shards=selected,
                split="train",
                rank=0,
                world_size=1,
                config=config,
                cache_budget=max_bytes,
                quota=quota,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=32, num_workers=0, pin_memory=False, drop_last=False
            )
            result["reader"] = {
                "ranks": 1,
                "workers": 0,
                "batch_size": 32,
                "num_samples": 145440,
                "sample_rate": 16000,
                "torch_num_threads": torch.get_num_threads(),
                "torch_version": str(torch.__version__),
            }
            with (
                patch.object(production, "_ShardCache", TimedCache),
                patch.object(
                    production, "_s3_client", lambda value: TimedS3(original_client(value), metrics)
                ),
                patch.object(production, "_audio_records", timed_records),
                patch.object(production, "_decode", timed_decode),
            ):
                begun = time.perf_counter()
                iterator = iter(loader)
                try:
                    for batch in iterator:
                        metrics["clips"] += batch["waveform"].shape[0]
                        metrics["batches"] += 1
                finally:
                    del iterator
                    metrics["cold_stream_seconds"] = time.perf_counter() - begun
            if metrics["clips"] != quota:
                raise RuntimeError("Production input yielded fewer clips than its frozen index")
            seconds = metrics["cold_stream_seconds"]
            metrics["cold_clips_per_second"] = metrics["clips"] / seconds
            non_network = metrics["cache_get_and_sha256_verify_seconds"] - (
                metrics["s3_get_response_seconds"] + metrics["s3_body_read_seconds"]
            )
            metrics["cache_hash_and_disk_seconds_estimate"] = max(0.0, non_network)
            result["required_aggregate_input_rates"] = [
                {
                    "updates_per_second": rate,
                    "global_batch": 64,
                    "required_clips_per_second": 64 * rate,
                    "measured_single_reader_ratio": metrics["cold_clips_per_second"] / (64 * rate),
                }
                for rate in (2.1, 3.3)
            ]
            result["limitations"] = [
                "Single CPU reader on a busy staging node; not a GPU throughput measurement.",
                "Cache/verification time includes S3 read time; timing fields are not all additive.",
                "One partial tail batch is allowed for this input-only probe.",
                "Input demand comparison does not establish GPU compute or multi-rank scaling.",
            ]
            result["status"] = "completed"
    finally:
        result["owned_temporary_cache_removed"] = (
            cache_location is not None and not cache_location.exists()
        )
        result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-bucket", default=os.environ.get("HWAVJEPA_INDEX_BUCKET"))
    parser.add_argument("--index-key", default=os.environ.get("HWAVJEPA_INDEX_KEY"))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HWAVJEPA_S3_ENDPOINT", "https://storage.givemeanode.com"),
    )
    parser.add_argument("--cache-parent", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial-index", action="store_true")
    parser.add_argument("--max-shards", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-gib", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    if not args.index_bucket or not args.index_key:
        parser.error("Set HWAVJEPA_INDEX_BUCKET and HWAVJEPA_INDEX_KEY, or pass their flags")
    if not 0 < args.max_gib <= 1.5 or not 1 <= args.max_seconds <= 180:
        parser.error("max-gib must be in (0, 1.5] and max-seconds in [1, 180]")
    torch.set_num_threads(1)
    if hasattr(os, "nice"):
        os.nice(10)
    report = {"status": "running", "wall_time_limit_seconds": args.max_seconds}
    try:
        with deadline(args.max_seconds):
            run_probe(
                index_path=f"s3://{args.index_bucket}/{args.index_key}",
                endpoint=args.endpoint,
                cache_parent=args.cache_parent,
                allow_partial=args.allow_partial_index,
                max_shards=args.max_shards,
                max_bytes=int(args.max_gib * 1024**3),
                max_samples=args.max_samples,
                report=report,
            )
    except (Exception, ProbeLimit) as error:
        # Exception class is enough for triage without dumping credential-bearing SDK context.
        report.update(status="failed", error_type=type(error).__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
