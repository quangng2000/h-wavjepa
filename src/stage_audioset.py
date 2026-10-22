"""Stage verified AudioSet tar shards to private S3 using bounded local CPU storage.

One source is materialized at a time using prepare_audioset. Local WAVs are
removed only after complete remote SHA256 readback, durable global ID enrollment,
and a verified remote dataset index. Credentials are read from environment only.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from src.prepare_audioset import (
    CATALOG,
    MIN_SAMPLES,
    REVISION,
    SAMPLE_RATE,
    Source,
    atomic_json,
    builtin_sources,
    connect_ledger,
    export_manifests,
    inspect_wav,
    sha256_file,
)

MAX_OBJECT_BYTES = 4 * 1024**3
EXPECTED_SOURCES = 889


class StageLimit(RuntimeError):
    """A configured bound was reached; staging remains explicitly partial."""


def missing_object(error: Exception) -> bool:
    response = getattr(error, "response", {})
    return str(response.get("Error", {}).get("Code", "")) in {"404", "NoSuchKey", "NotFound"}


class DeadlineReader:
    def __init__(self, handle, check):
        self.handle = handle
        self.check = check

    def read(self, size=-1):
        self.check()
        return self.handle.read(size)

    def __getattr__(self, name):
        return getattr(self.handle, name)


class Store:
    """Use real object contents, not unsupported S3 user metadata, as evidence."""

    def __init__(self, client, bucket, check):
        self.client = client
        self.bucket = bucket
        self.check = check

    def head(self, key):
        self.check()
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            if missing_object(error):
                return None
            raise

    def read(self, key, *, maximum=1024**2):
        self.check()
        result = self.client.get_object(Bucket=self.bucket, Key=key)
        with contextlib.closing(result["Body"]) as handle:
            value = handle.read(maximum + 1)
        if len(value) > maximum:
            raise ValueError(f"Unexpected oversized control object: {key}")
        return value

    def verify(self, key, digest, size):
        self.check()
        result = self.client.get_object(Bucket=self.bucket, Key=key)
        actual = hashlib.sha256()
        count = 0
        with contextlib.closing(result["Body"]) as handle:
            while True:
                self.check()
                chunk = handle.read(1024**2)
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise ValueError(f"Remote object exceeds expected size: {key}")
                actual.update(chunk)
        if count != size or actual.hexdigest() != digest:
            raise ValueError(f"Remote object checksum mismatch: {key}; local data retained")
        head = self.head(key)
        if head is None or head["ContentLength"] != size:
            raise ValueError(f"Remote object disappeared or changed during verification: {key}")
        return {
            "key": key,
            "sha256": digest,
            "bytes": size,
            "etag": head.get("ETag"),
            "verified_at": time.time(),
        }

    def put_file(self, key, path, digest):
        size = path.stat().st_size
        if size > MAX_OBJECT_BYTES:
            raise StageLimit("Prepared shard exceeds the supported 4 GiB single-PUT limit")
        if self.head(key) is None:
            with path.open("rb") as handle:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=DeadlineReader(handle, self.check),
                    ContentLength=size,
                )
        # Existing keys are content-addressed: verify rather than overwrite.
        return self.verify(key, digest, size)

    def put_json(self, key, value, *, immutable=False):
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if not immutable or self.head(key) is None:
            self.check()
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentLength=len(payload),
                ContentType="application/json",
            )
        self.verify(key, digest, len(payload))
        return digest

    def confirm_receipt(self, receipt):
        head = self.head(receipt["key"])
        if head is None:
            raise ValueError(f"Previously verified remote object is missing: {receipt['key']}")
        if head["ContentLength"] != receipt["bytes"]:
            raise ValueError(f"Previously verified remote object changed size: {receipt['key']}")
        # The initial full GET proved content. An unchanged server ETag and size
        # permit a cheap resume check. Without an ETag, redo the full readback.
        if not receipt.get("etag") or head.get("ETag") != receipt["etag"]:
            self.verify(receipt["key"], receipt["sha256"], receipt["bytes"])


class Stager:
    def __init__(
        self,
        output: Path,
        client,
        *,
        bucket: str,
        endpoint: str,
        sources: list[Source],
        catalog: Path,
        prefix="audioset",
        workers=2,
        max_sources=1,
        max_runtime_seconds=3600,
        max_upload_bytes=2 * 1024**3,
        max_local_bytes=8 * 1024**3,
        ffmpeg="ffmpeg",
        ffmpeg_version=None,
        prepare_callback=None,
        reuse_prepared=None,
    ):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.sources = sources
        self.catalog = Path(catalog).resolve()
        self.workers = workers
        self.max_sources = max_sources
        self.deadline = time.monotonic() + max_runtime_seconds
        self.max_upload_bytes = max_upload_bytes
        self.max_local_bytes = max_local_bytes
        self.ffmpeg = ffmpeg
        self.prepare_callback = prepare_callback
        self.reuse_prepared = Path(reuse_prepared).resolve() if reuse_prepared else None
        self.store = Store(client, bucket, self.check_time)
        if (
            not prefix
            or PurePosixPath(prefix).is_absolute()
            or ".." in PurePosixPath(prefix).parts
            or "\\" in prefix
        ):
            raise ValueError("S3 prefix must be a nonempty relative path without traversal")
        self.base_prefix = prefix.rstrip("/")
        if ffmpeg_version is None:
            ffmpeg_version = subprocess.run(
                [ffmpeg, "-version"], check=True, capture_output=True, text=True, timeout=10
            ).stdout
        self.contract = {
            "schema": 2,
            "revision": REVISION,
            "bucket": bucket,
            "endpoint": endpoint,
            "base_prefix": self.base_prefix,
            "catalog_sha256": sha256_file(self.catalog),
            "ffmpeg_version": ffmpeg_version,
            "prepare_script_sha256": sha256_file(Path(__file__).with_name("prepare_audioset.py")),
            "sources": [asdict(source) for source in sources],
        }
        self.db = None
        self.config = None
        self.root_key = None

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise StageLimit("Runtime budget reached; resume with the same staging state")

    def _initialize(self):
        config_path = self.output / "staging-config.json"
        ledger_path = self.output / "staging.sqlite3"
        if not config_path.exists():
            if ledger_path.exists():
                raise ValueError("Staging configuration is missing; restore the original state")
            existing = self.store.client.list_objects_v2(
                Bucket=self.store.bucket, Prefix=self.base_prefix + "/", MaxKeys=1
            )
            if existing.get("Contents"):
                raise ValueError(
                    "Remote prefix is populated but local global ledger is absent; restore staging state or choose a new prefix"
                )
            self.config = {**self.contract, "cohort_id": uuid.uuid4().hex}
            atomic_json(config_path, self.config)
        else:
            self.config = json.loads(config_path.read_text())
            if {
                key: value for key, value in self.config.items() if key != "cohort_id"
            } != self.contract:
                raise ValueError("Staging configuration, source catalog or FFmpeg version changed")
        self.root_key = f"{self.base_prefix}/{self.config['cohort_id']}"
        marker = {
            "cohort_id": self.config["cohort_id"],
            "contract_sha256": hashlib.sha256(
                json.dumps(self.contract, sort_keys=True).encode()
            ).hexdigest(),
        }
        # A crash between initial config and database creation is recoverable
        # only while this unique remote cohort is completely empty.
        fresh = not ledger_path.exists() or ledger_path.stat().st_size == 0
        if fresh:
            remote = self.store.client.list_objects_v2(
                Bucket=self.store.bucket, Prefix=self.root_key + "/", MaxKeys=1
            )
            if remote.get("Contents"):
                raise ValueError(
                    "Global ledger missing or empty for a populated cohort; restore its original state"
                )
        self.db = sqlite3.connect(ledger_path)
        if fresh:
            self.db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE shards (source TEXT PRIMARY KEY, receipt TEXT NOT NULL, report TEXT NOT NULL);
                CREATE TABLE recordings (recording_id TEXT PRIMARY KEY, source TEXT NOT NULL, split TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE INDEX recordings_source ON recordings(source, split);
                CREATE TABLE metadata (name TEXT PRIMARY KEY, receipt TEXT NOT NULL);
                CREATE TABLE control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            self.db.executemany("INSERT INTO control VALUES (?, ?)", marker.items())
            self.db.commit()
        else:
            tables = {
                row[0]
                for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if (
                not {"shards", "recordings", "metadata", "control"} <= tables
                or self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            ):
                raise ValueError(
                    "Global staging ledger is invalid; restore a verified state backup"
                )
            if dict(self.db.execute("SELECT key, value FROM control")) != marker:
                raise ValueError("Global staging ledger belongs to another cohort")
        counts = {
            (source, split): count
            for source, split, count in self.db.execute(
                "SELECT source, split, COUNT(*) FROM recordings GROUP BY source, split"
            )
        }
        local = {}
        for source, receipt, report in self.db.execute("SELECT * FROM shards"):
            data = json.loads(report)
            for split in ("train", "val", "test"):
                if counts.pop((source, split), 0) != data["splits"][split]["clips"]:
                    raise ValueError(
                        "Global recording-ID registry is incomplete; restore the original ledger"
                    )
            local[source] = json.loads(receipt)
        if counts:
            raise ValueError("Global recording-ID registry contains unenrolled sources")
        remote_index = f"{self.root_key}/index.json"
        if self.store.head(remote_index) is not None:
            remote = json.loads(self.store.read(remote_index, maximum=16 * 1024**2))
            if remote["cohort_id"] != self.config["cohort_id"]:
                raise ValueError("Remote index belongs to another cohort")
            for shard in remote["shards"]:
                receipt = local.get(shard["source"])
                if receipt is None or any(
                    receipt[field] != shard[field] for field in ("key", "bytes", "sha256")
                ):
                    raise ValueError(
                        "Local staging ledger is behind or differs from the remote index; restore its original state"
                    )
        self.store.put_json(f"{self.root_key}/cohort.json", marker, immutable=True)

    def _work_bytes(self):
        work = self.output / "work"
        return (
            sum(path.stat().st_size for path in work.rglob("*") if path.is_file())
            if work.exists()
            else 0
        )

    def _run_preparation(self, command):
        self.check_time()
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[1], start_new_session=True
        )
        try:
            code = process.wait(timeout=max(0.01, self.deadline - time.monotonic()))
            if code:
                raise RuntimeError(f"Preparation subprocess failed (exit {code}); work retained")
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise StageLimit(
                "Runtime budget reached during CPU preparation; resumable work retained"
            ) from error

    def _owned_folder(self, source):
        folder = self.output / "work" / source.name
        marker = folder / ".stage-owned.json"
        expected = {"cohort_id": self.config["cohort_id"], "source": source.name}
        if (
            folder.exists()
            and not folder.is_symlink()
            and not marker.exists()
            and not any(folder.iterdir())
        ):
            atomic_json(marker, expected)
        if folder.exists():
            if (
                folder.is_symlink()
                or not marker.is_file()
                or json.loads(marker.read_text()) != expected
            ):
                raise ValueError("Refusing to use an unowned staging work directory")
        else:
            folder.mkdir(parents=True)
            atomic_json(marker, expected)
        return folder

    def _prepare(self, source, folder):
        self.check_time()
        if self.prepare_callback is not None:
            self.prepare_callback(source, folder)
            return
        if self.reuse_prepared is not None and self._copy_prepared(source, folder):
            return
        report = folder / "preparation-report.json"
        if report.exists():
            existing = json.loads(report.read_text())
            if any(
                item["name"] == source.name
                and item["status"] == "complete"
                and item["verified_sha256"] == source.sha256
                for item in existing["sources"]
            ):
                return
        remaining = self.max_local_bytes - self._work_bytes()
        if remaining <= 0:
            raise StageLimit("Local working-space budget exhausted")
        floor = max(1024**3, shutil.disk_usage(folder).free - remaining)
        command = [
            sys.executable,
            "-m",
            "src.prepare_audioset",
            "--output",
            str(folder),
            "--catalog",
            str(self.catalog),
            "--source",
            source.name,
            "--workers",
            str(self.workers),
            "--minimum-free-gib",
            str(floor / 1024**3),
            "--ffmpeg",
            self.ffmpeg,
        ]
        self._run_preparation(command)

    def _validate_preparation_contract(self, folder):
        expected = {
            "revision": REVISION,
            "sample_rate": SAMPLE_RATE,
            "minimum_samples": MIN_SAMPLES,
            "maximum_seconds": 30,
        }
        config = json.loads((folder / "preparation-config.json").read_text())
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Prepared source uses a different audio preparation contract")
        runtime = json.loads((folder / "preparation-runtime.json").read_text())
        if runtime["ffmpeg_version"] != self.contract["ffmpeg_version"]:
            raise ValueError("Prepared source used a different FFmpeg version")

    def _copy_prepared(self, source, folder):
        self._validate_preparation_contract(self.reuse_prepared)
        original = self.reuse_prepared / "preparation.sqlite3"
        if not original.is_file():
            raise ValueError("Reuse directory has no preparation ledger")
        used = self._work_bytes()
        with sqlite3.connect(f"file:{original}?mode=ro", uri=True) as old:
            row = old.execute("SELECT * FROM sources WHERE name=?", (source.name,)).fetchone()
            if row is None or row[2] != "complete" or row[3] != source.sha256:
                return False
            if json.loads(row[1]) != asdict(source):
                raise ValueError(
                    "Reused prepared source descriptor differs from the pinned catalog"
                )
            with contextlib.closing(connect_ledger(folder)) as new:
                new.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?)", row)
                for clip in old.execute(
                    "SELECT * FROM clips WHERE source=? AND verified=1", (source.name,)
                ):
                    self.check_time()
                    path = Path(clip[4])
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("Unsafe reused clip path")
                    src = (self.reuse_prepared / path).resolve()
                    if not src.is_relative_to(self.reuse_prepared):
                        raise ValueError("Reused clip escapes the source directory")
                    destination = folder / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        size = src.stat().st_size
                        if used + size + 16 * 1024**2 > self.max_local_bytes:
                            raise StageLimit(
                                "Reusing prepared files would exceed the local working-space budget"
                            )
                        if shutil.disk_usage(folder).free < 1024**3 + size:
                            raise StageLimit("Insufficient free disk for reused prepared source")
                        used += size
                        try:
                            os.link(src, destination)
                        except OSError:
                            shutil.copyfile(src, destination)
                    new.execute(
                        "INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", clip
                    )
                new.executemany(
                    "INSERT OR REPLACE INTO members VALUES (?, ?, ?, ?)",
                    old.execute("SELECT * FROM members WHERE source=?", (source.name,)),
                )
                new.commit()
        for name in ("preparation-config.json", "preparation-runtime.json"):
            if (self.reuse_prepared / name).is_file():
                shutil.copyfile(self.reuse_prepared / name, folder / name)
        return True

    def _validate_and_filter(self, source, folder):
        if self.prepare_callback is None:
            self._validate_preparation_contract(folder)
        with contextlib.closing(connect_ledger(folder)) as prepared:
            row = prepared.execute(
                "SELECT descriptor, status, verified_sha256 FROM sources WHERE name=?",
                (source.name,),
            ).fetchone()
            if (
                row is None
                or row[1] != "complete"
                or row[2] != source.sha256
                or json.loads(row[0]) != asdict(source)
            ):
                raise ValueError("Only a complete SHA256-verified source may be staged")
            other = prepared.execute(
                "SELECT COUNT(*) FROM clips WHERE source!=?", (source.name,)
            ).fetchone()[0]
            if other:
                raise ValueError("Staging work directory contains more than one source")
            for recording_id, member, split, relative, samples in prepared.execute(
                "SELECT recording_id, member, split, path, samples FROM clips"
            ).fetchall():
                self.check_time()
                audio = (folder / relative).resolve()
                if (
                    not audio.is_relative_to(folder.resolve())
                    or inspect_wav(audio, MIN_SAMPLES, 30 * SAMPLE_RATE) != samples
                ):
                    raise ValueError(
                        "Prepared audio path or sample count violates the staging contract"
                    )
                previous = self.db.execute(
                    "SELECT source, split FROM recordings WHERE recording_id=?", (recording_id,)
                ).fetchone()
                if previous and previous[0] != source.name:
                    if previous[1] != split:
                        raise ValueError(
                            f"Cross-split recording-ID collision: {recording_id}; source {source.name} conflicts with {previous[0]}"
                        )
                    prepared.execute("DELETE FROM clips WHERE recording_id=?", (recording_id,))
                    prepared.execute(
                        "UPDATE members SET status='rejected', error=? WHERE source=? AND member=?",
                        (f"Already staged recording ID from {previous[0]}", source.name, member),
                    )
            prepared.commit()
            report = export_manifests(folder, prepared, verify_files=True)
        runtime = folder / "preparation-runtime.json"
        if (
            runtime.exists()
            and json.loads(runtime.read_text())["ffmpeg_version"] != self.contract["ffmpeg_version"]
        ):
            raise ValueError("Prepared source used a different FFmpeg version")
        return report

    def _pack(self, source, folder):
        archive = folder / "prepared.tar"
        temporary = archive.with_suffix(".tar.partial")
        # Owned source WAVs remain intact; drop stale packaging before repacking
        # so a retry never holds two complete tar copies beside those WAVs.
        archive.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        paths = [
            folder / name
            for name in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "clips.provenance.jsonl",
                "rejections.jsonl",
                "preparation-report.json",
                "preparation-config.json",
                "preparation-runtime.json",
            )
            if (folder / name).is_file()
        ]
        with connect_ledger(folder) as prepared:
            paths.extend(
                folder / row[0]
                for row in prepared.execute(
                    "SELECT path FROM clips WHERE verified=1 ORDER BY recording_id"
                )
            )
        estimated = sum(((path.stat().st_size + 511) // 512 + 1) * 512 for path in paths) + 10_240
        work_size = self._work_bytes()
        if estimated > MAX_OBJECT_BYTES or work_size + estimated > self.max_local_bytes:
            raise StageLimit("Packing would exceed the local or single-object size budget")
        uploaded = self.uploaded_bytes()
        if uploaded + estimated > self.max_upload_bytes:
            raise StageLimit("Packing would exceed the configured cumulative uploaded-byte budget")
        temporary = archive.with_suffix(".tar.partial")
        try:
            with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as handle:
                for path in paths:
                    self.check_time()
                    resolved = path.resolve()
                    if not resolved.is_relative_to(folder.resolve()) or path.is_symlink():
                        raise ValueError("Prepared file escapes the owned work directory")
                    info = tarfile.TarInfo(path.relative_to(folder).as_posix())
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    with path.open("rb") as payload:
                        handle.addfile(info, DeadlineReader(payload, self.check_time))
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        return archive, sha256_file(archive)

    def uploaded_bytes(self):
        return sum(
            json.loads(row[0])["bytes"]
            for table in ("shards", "metadata")
            for row in self.db.execute(f"SELECT receipt FROM {table}")
        )

    def _enroll(self, source, folder, receipt, report):
        with contextlib.closing(connect_ledger(folder)) as prepared, self.db:
            for recording_id, split, digest in prepared.execute(
                "SELECT recording_id, split, sha256 FROM clips WHERE verified=1"
            ):
                self.db.execute(
                    "INSERT INTO recordings VALUES (?, ?, ?, ?)",
                    (recording_id, source.name, split, digest),
                )
            self.db.execute(
                "INSERT INTO shards VALUES (?, ?, ?)",
                (source.name, json.dumps(receipt), json.dumps(report)),
            )

    def index(self, reason=None):
        shards = []
        totals = {split: 0 for split in ("train", "val", "test")}
        for source, receipt, report in self.db.execute("SELECT * FROM shards ORDER BY source"):
            data = json.loads(report)
            for split in totals:
                totals[split] += data["splits"][split]["clips"]
            shards.append(
                {
                    "source": source,
                    **json.loads(receipt),
                    "splits": data["splits"],
                    "provenance_sha256": data["provenance_sha256"],
                    "rejections": data["rejections"],
                }
            )
        staged = {entry["source"] for entry in shards}
        expected = {source.name for source in self.sources}
        complete = (
            len(expected) == EXPECTED_SOURCES
            and staged == expected
            and self.db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] == 3
        )
        return {
            "schema": 1,
            "revision": REVISION,
            "cohort_id": self.config["cohort_id"],
            "status": "complete" if complete else "partial",
            "stop_reason": reason,
            "bucket": self.store.bucket,
            "endpoint": self.contract["endpoint"],
            "expected_sources": len(expected),
            "full_cohort_sources": EXPECTED_SOURCES,
            "verified_sources": len(shards),
            "split_counts": totals,
            "stored_payload_bytes": self.uploaded_bytes(),
            "shards": shards,
            "metadata": [
                json.loads(row[0])
                for row in self.db.execute("SELECT receipt FROM metadata ORDER BY name")
            ],
            "index_key": f"{self.root_key}/index.json",
        }

    def publish(self, reason=None):
        value = self.index(reason)
        self.store.put_json(value["index_key"], value)
        atomic_json(self.output / "index.json", value)
        return value

    def _purge(self, source):
        folder = self.output / "work" / source.name
        if not folder.exists():
            return
        self._owned_folder(source)
        shutil.rmtree(folder)

    def stage_metadata(self):
        self.check_time()
        candidates = [
            item
            for item in json.loads(self.catalog.read_text())["files"]
            if item["path"].endswith("_segments.csv")
        ]
        required = sum(item["size"] for item in candidates)
        existing = sum(
            json.loads(row[0])["bytes"] for row in self.db.execute("SELECT receipt FROM metadata")
        )
        if (
            self.uploaded_bytes() + required - existing > self.max_upload_bytes
            or required > self.max_local_bytes
        ):
            raise StageLimit("Metadata would exceed the configured storage budget")
        self._run_preparation(
            [
                sys.executable,
                "-m",
                "src.prepare_audioset",
                "--output",
                str(self.output),
                "--catalog",
                str(self.catalog),
                "--metadata-only",
            ]
        )
        prepared = json.loads((self.output / "metadata" / "provenance.json").read_text())
        for item in prepared["files"]:
            name = Path(item["path"]).name
            row = self.db.execute("SELECT receipt FROM metadata WHERE name=?", (name,)).fetchone()
            if row:
                self.store.confirm_receipt(json.loads(row[0]))
                continue
            self.check_time()
            if self.uploaded_bytes() + item["size"] > self.max_upload_bytes:
                raise StageLimit("Metadata would exceed the uploaded-byte budget")
            key = f"{self.root_key}/metadata/{item['sha256']}/{name}"
            receipt = self.store.put_file(key, self.output / item["local_path"], item["sha256"])
            self.db.execute("INSERT INTO metadata VALUES (?, ?)", (name, json.dumps(receipt)))
            self.db.commit()
        self.publish()

    def run(self, selected=None, *, metadata=True):
        selected = (
            set(selected) if selected is not None else {source.name for source in self.sources}
        )
        unknown = selected - {source.name for source in self.sources}
        if unknown:
            raise ValueError(f"Unknown requested sources: {sorted(unknown)}")
        with (self.output / ".staging.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another writer owns this staging state") from error
            try:
                self._initialize()
                if metadata:
                    self.stage_metadata()
                processed = 0
                for source in self.sources:
                    self.check_time()
                    row = self.db.execute(
                        "SELECT receipt FROM shards WHERE source=?", (source.name,)
                    ).fetchone()
                    if row:
                        self.store.confirm_receipt(json.loads(row[0]))
                        if (self.output / "work" / source.name).exists():
                            self.publish()
                            self._purge(source)
                        continue
                    if source.name not in selected:
                        continue
                    if processed >= self.max_sources:
                        return self.publish("Per-invocation source limit reached")
                    folder = self._owned_folder(source)
                    self._prepare(source, folder)
                    report = self._validate_and_filter(source, folder)
                    archive, digest = self._pack(source, folder)
                    key = f"{self.root_key}/{REVISION}/{source.name}/{digest}.tar"
                    receipt = self.store.put_file(key, archive, digest)
                    self._enroll(source, folder, receipt, report)
                    self.publish()
                    self._purge(source)
                    processed += 1
                    print(
                        json.dumps(
                            {
                                "source": source.name,
                                "status": "uploaded_and_verified",
                                "bytes": receipt["bytes"],
                                "clips": report["splits"][source.split]["clips"],
                            }
                        ),
                        flush=True,
                    )
                return self.publish()
            except StageLimit as error:
                if self.db is not None:
                    value = self.index(str(error))
                    value["status"] = "partial"
                    value["publication_pending"] = True
                    atomic_json(self.output / "index.json", value)
                    # The deadline can prevent publishing the latest stop reason;
                    # prior remote indexes still truthfully say partial.
                    if time.monotonic() < self.deadline:
                        return self.publish(str(error))
                    return value
                raise
            finally:
                if self.db is not None:
                    self.db.close()
                    self.db = None


def make_client(endpoint):
    import boto3
    from botocore.config import Config

    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise ValueError(
            "Set bucket-scoped AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "auto"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=20,
            retries={"max_attempts": 2},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Persistent staging state directory; retain it for every resume",
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HWAVJEPA_S3_ENDPOINT"),
        required=not os.environ.get("HWAVJEPA_S3_ENDPOINT"),
    )
    parser.add_argument("--prefix", default="audioset")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--source", action="append")
    parser.add_argument("--max-sources", type=int, default=1)
    parser.add_argument("--max-runtime-hours", type=float, default=1.0)
    parser.add_argument(
        "--max-upload-gib",
        type=float,
        default=2.0,
        help="Cumulative archive+metadata payload budget across all resumes",
    )
    parser.add_argument("--max-local-gib", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--reuse-prepared",
        type=Path,
        help="Optional existing prepared corpus; only complete verified source shards are reused",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.max_sources,
            args.max_runtime_hours,
            args.max_upload_gib,
            args.max_local_gib,
            args.workers,
        )
        <= 0
        or args.workers > 32
    ):
        parser.error("Budgets must be positive and workers must be 1..32")
    if not args.endpoint.startswith("https://"):
        parser.error("An explicit HTTPS S3 endpoint is required")
    stager = Stager(
        args.output,
        make_client(args.endpoint),
        bucket=args.bucket,
        endpoint=args.endpoint,
        sources=builtin_sources(args.catalog),
        catalog=args.catalog,
        prefix=args.prefix,
        workers=args.workers,
        max_sources=args.max_sources,
        max_runtime_seconds=args.max_runtime_hours * 3600,
        max_upload_bytes=int(args.max_upload_gib * 1024**3),
        max_local_bytes=int(args.max_local_gib * 1024**3),
        ffmpeg=args.ffmpeg,
        reuse_prepared=args.reuse_prepared,
    )
    print(json.dumps(stager.run(args.source), indent=2), flush=True)


if __name__ == "__main__":
    main()
