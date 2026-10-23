"""Bounded rolling staging of pinned WavJEPA-Nat assets to private S3.

No compute is provisioned by this command. Credentials use the standard AWS
environment. Download one source, verify its publisher digest, convert locally,
PUT each content-addressed object and verify its entire GET, durably enroll its
receipts, then remove only this command's scratch files. A full training manifest
is published only after all seventy BRIR sources and the WHAM training split pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.prepare_nat_assets import (
    BRIR_REPOSITORY,
    BRIR_REVISION,
    PreparationLimit,
    _atomic_json,
    _digest,
    _read_component,
    _safe_member,
    assemble_nat_manifest,
    prepare_brir_shard,
    prepare_wham_noise,
    sha256_file,
)
from src.stage_audioset import DeadlineReader, Store

WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
WHAM_MD5 = "d5af15645d521d3920e01954c6cd7594"
WHAM_BYTES = 18194930748
INVENTORY = (
    Path(__file__).resolve().parents[1] / "docs/validation/wavjepa-nat-assets-2026-10-23.json"
)


def _error_status_code(error):
    # Transport exceptions have response=None, unlike botocore ClientError.
    # Preserve the real failure instead of masking it with AttributeError.
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None, ""
    metadata, detail = response.get("ResponseMetadata"), response.get("Error")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    code = str(detail.get("Code", "")) if isinstance(detail, Mapping) else ""
    return status, code


def _transient_transport_error(error):
    # Keep boto3 optional for converter-only installations and CPU fixtures.
    try:
        from botocore.exceptions import (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
            ResponseStreamingError,
        )
    except ImportError:
        return False
    return isinstance(
        error,
        (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
            ResponseStreamingError,
        ),
    )


def _wait_checked(check, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check()
        time.sleep(max(0, min(0.25, end - time.monotonic())))
    check()


def _failure_diagnostic(error):
    """Keep useful frames and types, never exception messages, source or locals."""
    failures, seen = [], set()
    while error is not None and id(error) not in seen and len(failures) < 4:
        seen.add(id(error))
        frames, trace = [], error.__traceback__
        while trace is not None:
            code = trace.tb_frame.f_code
            frames.append(
                {
                    "file": Path(code.co_filename).name,
                    "function": code.co_name,
                    "line": trace.tb_lineno,
                }
            )
            trace = trace.tb_next
        failures.append({"type": type(error).__name__, "frames": frames[-12:]})
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return {"failure_type": failures[0]["type"], "failure_trace": failures}


def read_inventory(path: Path) -> list[dict]:
    """Accept only the exact published source family, never a supplied download URL."""
    with path.open("rb") as handle:
        payload = handle.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError("Inventory exceeds 1 MiB")
    value = json.loads(payload)
    brir, noise = value["brir"], value["noise"]
    if brir["repository"] != BRIR_REPOSITORY or brir["revision"] != BRIR_REVISION:
        raise ValueError("Inventory must use the pinned BRIR publisher revision")
    shards = brir["shards"]
    expected = {f"train/webdataset_output/shard-{index:06d}.tar" for index in range(70)}
    if len(shards) != 70 or {item["path"] for item in shards} != expected:
        raise ValueError("Inventory requires all seventy distinct BRIR source paths")
    sources = []
    for item in sorted(shards, key=lambda item: item["path"]):
        _digest(item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] != 9217034240:
            raise ValueError("Unexpected official BRIR archive size")
        sources.append(
            {
                **item,
                "id": Path(item["path"]).stem,
                "kind": "brir",
                "url": f"https://huggingface.co/datasets/{BRIR_REPOSITORY}/resolve/{BRIR_REVISION}/{item['path']}",
            }
        )
    if (
        noise["download_url"] != WHAM_URL
        or noise["published_md5"] != WHAM_MD5
        or noise["download_bytes"] != WHAM_BYTES
    ):
        raise ValueError("WHAM source must match the publisher URL, size and MD5")
    sources.append(
        {"id": "wham-train", "kind": "wham", "url": WHAM_URL, "bytes": WHAM_BYTES, "md5": WHAM_MD5}
    )
    return sources


class LimitedClient:
    """One shared rate limiter, including retries; no hidden SDK retry bursts."""

    def __init__(self, client, check, requests_per_second=8.0):
        if not math.isfinite(requests_per_second) or not 0 < requests_per_second <= 9:
            raise ValueError("S3 request rate must be in (0, 9] per second")
        self.client, self.check = client, check
        self.interval = 1 / requests_per_second
        self.lock = threading.Lock()
        self.next_request = 0.0

    def __getattr__(self, method):
        operation = getattr(self.client, method)

        def call(**kwargs):
            body = kwargs.get("Body")
            offset = body.tell() if hasattr(body, "tell") else None
            for attempt in range(6):
                with self.lock:
                    self.check()
                    delay = max(0, self.next_request - time.monotonic())
                    self._wait(delay)
                    self.next_request = time.monotonic() + self.interval
                if offset is not None:
                    body.seek(offset)
                try:
                    return operation(**kwargs)
                except Exception as error:
                    status, code = _error_status_code(error)
                    if attempt == 5 or (
                        status not in (429, 500, 502, 503, 504)
                        and code not in {"SlowDown", "Throttling", "RequestTimeout"}
                        and not _transient_transport_error(error)
                    ):
                        raise
                    self._wait(min(30, 2**attempt))

        return call

    def _wait(self, seconds):
        _wait_checked(self.check, seconds)


class NatStore(Store):
    """Two requests per fresh payload: conditional PUT, then full GET proof."""

    def head(self, key):
        self.check()
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            status, code = _error_status_code(error)
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def verify(self, key, digest, size):
        for attempt in range(6):
            self.check()
            # Request failures already receive LimitedClient's bounded retries.
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            try:
                return self._verify_response(response, key, digest, size)
            except Exception as error:
                # get_object can succeed before the streaming body disconnects.
                # Restart its full proof from byte zero; never accept a prefix.
                if attempt == 5 or not _transient_transport_error(error):
                    raise
                _wait_checked(self.check, min(30, 2**attempt))

    def _verify_response(self, response, key, digest, size):
        actual, count = hashlib.sha256(), 0
        with contextlib.closing(response["Body"]) as handle:
            if response.get("ContentLength") != size:
                raise ValueError("Remote Nat object has unexpected size")
            while True:
                self.check()
                block = handle.read(1024**2)
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("Remote Nat object exceeds its size bound")
                actual.update(block)
        if count != size or actual.hexdigest() != digest:
            raise ValueError("Remote Nat object failed full GET SHA-256 verification")
        return {
            "key": key,
            "sha256": digest,
            "bytes": size,
            "etag": response.get("ETag"),
            "verified_at": time.time(),
        }

    def put_payload(self, key, path, digest):
        size = path.stat().st_size
        if not 0 < size <= 3 * 1024**2 or sha256_file(path, self.check) != digest:
            raise ValueError("Local converted payload failed size/SHA verification")
        try:
            with path.open("rb") as handle:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=DeadlineReader(handle, self.check),
                    ContentLength=size,
                    IfNoneMatch="*",
                )
        except Exception as error:
            status, code = _error_status_code(error)
            if status != 412 and code not in {"412", "PreconditionFailed"}:
                raise
        return self.verify(key, digest, size)

    def _put_immutable(self, **kwargs):
        try:
            return self.client.put_object(Bucket=self.bucket, IfNoneMatch="*", **kwargs)
        except Exception as error:
            status, code = _error_status_code(error)
            if status != 412 and code not in {"412", "PreconditionFailed"}:
                raise

    def put_file(self, key, path, digest):
        size = path.stat().st_size
        if not 0 < size <= 4 * 1024**3 or sha256_file(path, self.check) != digest:
            raise ValueError("Control object failed its bounded local SHA verification")
        with path.open("rb") as handle:
            self._put_immutable(
                Key=key, Body=DeadlineReader(handle, self.check), ContentLength=size
            )
        return self.verify(key, digest, size)

    def put_json(self, key, value, *, immutable=False):
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        kwargs = {
            "Key": key,
            "Body": payload,
            "ContentLength": len(payload),
            "ContentType": "application/json",
        }
        if immutable:
            self._put_immutable(**kwargs)
        else:
            self.client.put_object(Bucket=self.bucket, **kwargs)
        self.verify(key, digest, len(payload))
        return digest


def download_source(source: dict, destination: Path, check, *, opener=urlopen) -> dict:
    """Stream to a partial file; verify published SHA/MD5 before atomic promotion.

    Failed partial downloads remain for diagnostics. A later attempt restarts the
    partial file, while an already promoted archive is rehashed and reused.
    """
    expected_size = source["bytes"]
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest, publisher_md5, count = hashlib.sha256(), hashlib.md5(), 0

    def consume(handle, output=None):
        nonlocal count
        while True:
            check()
            block = handle.read(4 * 1024**2)
            if not block:
                break
            count += len(block)
            if count > expected_size:
                raise ValueError("Publisher download exceeds expected size")
            digest.update(block)
            publisher_md5.update(block)
            if output is not None:
                output.write(block)

    if destination.exists():
        with destination.open("rb") as handle:
            consume(handle)
    else:
        request = Request(source["url"], headers={"User-Agent": "h-wavjepa-nat-stager/1"})
        with opener(request, timeout=30) as response, temporary.open("wb") as output:
            if response.status != 200:
                raise ValueError("Publisher download did not return the complete archive")
            if (
                response.headers.get("Content-Length") is not None
                and int(response.headers["Content-Length"]) != expected_size
            ):
                raise ValueError("Publisher Content-Length differs from pinned size")
            consume(response, output)
            output.flush()
            os.fsync(output.fileno())
    if count != expected_size:
        raise ValueError("Publisher archive is truncated")
    if source.get("sha256") is not None and digest.hexdigest() != source["sha256"]:
        raise ValueError("BRIR publisher SHA-256 mismatch")
    if source.get("md5") is not None and publisher_md5.hexdigest() != source["md5"]:
        raise ValueError("WHAM publisher MD5 mismatch; SHA-256 is not established")
    if not destination.exists():
        os.replace(temporary, destination)
    return {
        "bytes": count,
        "sha256": digest.hexdigest(),
        "publisher_digest_verified": True,
        "publisher_md5": source.get("md5"),
    }


class NatStager:
    def __init__(
        self,
        output: Path,
        client,
        *,
        endpoint: str,
        bucket: str,
        prefix: str,
        inventory: Path = INVENTORY,
        max_runtime_seconds=36000,
        max_sources=71,
        max_scratch_bytes=100 * 1024**3,
        max_upload_bytes=192 * 1024**3,
        workers=8,
        requests_per_second=9.0,
        prefetch_sources=1,
        download_backend="urllib",
    ):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("Endpoint must be HTTPS without credentials, query or path")
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
            raise ValueError("Invalid S3 bucket")
        if not prefix or str(_safe_member(prefix)) != prefix or prefix == ".":
            raise ValueError("Prefix must be a canonical nonempty relative path")
        if not math.isfinite(max_runtime_seconds) or not 0 < max_runtime_seconds <= 43200:
            raise ValueError("Runtime must be positive and at most twelve hours per invocation")
        if type(max_sources) is not int or not 1 <= max_sources <= 71:
            raise ValueError("max_sources must be between one and seventy-one")
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("workers must be between one and eight")
        if any(
            type(value) is not int or value <= 0 for value in (max_scratch_bytes, max_upload_bytes)
        ):
            raise ValueError("Scratch/upload byte bounds must be positive integers")
        self.output = Path(output).resolve()
        self.inventory = Path(inventory).resolve()
        self.sources = read_inventory(self.inventory)
        self.endpoint, self.bucket, self.prefix = endpoint.rstrip("/"), bucket, prefix
        self.max_sources, self.workers = max_sources, workers
        self.max_scratch_bytes, self.max_upload_bytes = max_scratch_bytes, max_upload_bytes
        if type(prefetch_sources) is not int or prefetch_sources not in (0, 1):
            raise ValueError("prefetch_sources must be zero or one (two source slots maximum)")
        if download_backend not in ("urllib", "huggingface"):
            raise ValueError("Unsupported BRIR download backend")
        self.prefetch_sources = prefetch_sources
        self.download_backend = download_backend
        self.reservations: dict[str, int] = {}
        self.cancelled = threading.Event()
        self.deadline = time.monotonic() + max_runtime_seconds
        self.run_id = uuid.uuid4().hex
        self.store = NatStore(
            LimitedClient(client, self.check, requests_per_second), bucket, self.check
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "staging.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.scratch = self.output / "scratch"
        self.scratch.mkdir(exist_ok=True)
        self.components = self.output / "components"
        self.components.mkdir(exist_ok=True)
        identity = {
            "schema_version": 1,
            "endpoint": self.endpoint,
            "bucket": bucket,
            "prefix": prefix,
            "inventory_sha256": sha256_file(self.inventory),
        }
        self.identity = identity
        config = self.output / "configuration.json"
        if config.exists() and json.loads(config.read_text()) != identity:
            raise ValueError("Existing staging configuration differs; use a new output directory")
        if not config.exists():
            _atomic_json(config, identity)
        self.database = sqlite3.connect(self.output / "ledger.sqlite3")
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=FULL")
        self.database.executescript(
            "CREATE TABLE IF NOT EXISTS objects (key TEXT PRIMARY KEY, receipt TEXT NOT NULL, run_id TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS components (id TEXT PRIMARY KEY, manifest TEXT NOT NULL, receipt TEXT NOT NULL);"
        )

    def close(self):
        self.database.close()
        self.lock.close()

    def preserve_diagnostics(self, destination: Path):
        """Keep compact control evidence in job artifacts; never copy asset payloads."""
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("configuration.json", "status.json", "final-receipt.json"):
            source = self.output / name
            if source.exists():
                if source.stat().st_size > 1024**2:
                    raise ValueError("Unexpected oversized staging diagnostic")
                shutil.copy2(source, destination / name)
        pages = self.database.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.database.execute("PRAGMA page_size").fetchone()[0]
        if pages * page_size > 128 * 1024**2:
            raise ValueError("Staging ledger exceeds the 128 MiB diagnostic bound")
        descriptor, temporary = tempfile.mkstemp(prefix=".ledger-", dir=destination)
        os.close(descriptor)
        try:
            with contextlib.closing(sqlite3.connect(temporary)) as target:
                self.database.backup(target)
            os.replace(temporary, destination / "ledger.sqlite3")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    def check(self):
        if self.cancelled.is_set():
            raise PreparationLimit("Nat preparation cancelled after a pipeline failure")
        if time.monotonic() >= self.deadline:
            raise PreparationLimit("Nat staging runtime ceiling reached")

    def object_receipt(self, key):
        row = self.database.execute(
            "SELECT receipt, run_id FROM objects WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else (json.loads(row[0]), row[1])

    def verify_enrolled(self, key, digest, size):
        self.check()
        enrolled = self.object_receipt(key)
        if enrolled is None or enrolled[0]["sha256"] != digest or enrolled[0]["bytes"] != size:
            raise ValueError("Assembly asset lacks a matching full-GET verification receipt")
        if enrolled[1] != self.run_id:
            self.store.confirm_receipt(enrolled[0])
            with self.database:
                self.database.execute("UPDATE objects SET run_id=? WHERE key=?", (self.run_id, key))

    def _reserve_source(self, source, *, optional=False):
        """Reserve worst-case bytes before starting either of the two source slots.

        Only the main thread reserves/releases slots. Reservation happens when
        the preceding producer has finished, so no producer writes race this
        accounting. A low disk budget falls back to sequential preparation.
        """
        self.check()
        identity = source["id"]
        if identity in self.reservations:
            raise ValueError("A source already has a preparation writer")
        converted_cap = (3 if source["kind"] == "brir" else 8) * 1024**3
        # Metadata, converter's temporary object, and a bounded HF shard cache.
        needed = source["bytes"] + converted_cap + 256 * 1024**2
        sizes = {}
        for child in self.scratch.iterdir():
            sizes[child.name] = (
                sum(path.stat().st_size for path in child.rglob("*") if path.is_file())
                if child.is_dir()
                else child.stat().st_size
            )
        previous = sizes.pop(identity, 0)
        additional = max(0, needed - previous)
        occupied = sum(self.reservations.values()) + needed
        for active, capacity in self.reservations.items():
            actual = sizes.pop(active, 0)
            if actual > capacity:
                raise PreparationLimit("An active source exceeded its scratch reservation")
            additional += capacity - actual
        occupied += sum(sizes.values())
        fits = (
            len(self.reservations) < 2
            and previous <= needed
            and occupied <= self.max_scratch_bytes
            and shutil.disk_usage(self.scratch).free >= additional + 1024**3
        )
        if not fits:
            if optional:
                return False
            raise PreparationLimit("Nat scratch byte ceiling or free disk prevents the next source")
        self.reservations[identity] = needed
        return True

    def _upload(self, component, directory):
        unique = {record["key"]: record for record in component["assets"].values()}
        stored_bytes = sum(
            json.loads(row[0])["bytes"]
            for row in self.database.execute("SELECT receipt FROM objects")
        )
        pending = []
        for local_key, record in unique.items():
            key = f"{self.prefix}/{local_key}"
            if self.object_receipt(key) is not None:
                self.verify_enrolled(key, record["sha256"], record["bytes"])
            else:
                stored_bytes += record["bytes"]
                if stored_bytes > self.max_upload_bytes:
                    raise PreparationLimit("Nat total verified object byte ceiling reached")
                pending.append((key, directory / local_key, record["sha256"]))
        # Refill each free worker immediately; a slow object does not stall a
        # whole batch. At most workers payloads are outstanding, with SQLite
        # enrollment exclusively on the main thread.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        active = set()
        try:
            items = iter(pending)
            active = {
                executor.submit(self.store.put_payload, *item)
                for item in list(pending[: self.workers])
            }
            for _ in range(min(self.workers, len(pending))):
                next(items)
            while active:
                self.check()
                finished, active = concurrent.futures.wait(
                    active, timeout=0.25, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in finished:
                    receipt = future.result()
                    with self.database:
                        self.database.execute(
                            "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                            (receipt["key"], json.dumps(receipt, sort_keys=True), self.run_id),
                        )
                    item = next(items, None)
                    if item is not None:
                        active.add(executor.submit(self.store.put_payload, *item))
        except BaseException:
            self.cancelled.set()
            for future in active:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _existing_component(self, source):
        row = self.database.execute(
            "SELECT manifest, receipt FROM components WHERE id=?", (source["id"],)
        ).fetchone()
        if row is None:
            return None
        path = Path(row[0])
        receipt = json.loads(row[1])
        if (
            not path.is_relative_to(self.components)
            or sha256_file(path, self.check) != receipt["sha256"]
        ):
            raise ValueError("Enrolled component manifest changed or left its directory")
        component = _read_component(path)
        if component["source"]["bytes"] != source["bytes"] or (
            source.get("sha256") and component["source"]["sha256"] != source["sha256"]
        ):
            raise ValueError("Enrolled component differs from pinned source")
        self.store.confirm_receipt(receipt)
        for record in {item["key"]: item for item in component["assets"].values()}.values():
            self.verify_enrolled(
                f"{self.prefix}/{record['key']}", record["sha256"], record["bytes"]
            )
        return path

    def _restore_component(self, source):
        """Reconstruct an ephemeral job's ledger from a verified remote receipt."""
        key = f"{self.prefix}/receipts/{source['id']}.json"
        if self.store.head(key) is None:
            return None
        envelope = json.loads(self.store.read(key, maximum=64 * 1024**2))
        if envelope.get("schema_version") != 1 or envelope.get("configuration") != self.identity:
            raise ValueError("Remote component receipt has incompatible configuration")
        component = envelope["component"]
        proof = envelope["source_verification"]
        if (
            proof.get("publisher_digest_verified") is not True
            or proof.get("bytes") != source["bytes"]
            or component["source"]["sha256"] != proof.get("sha256")
            or (source.get("sha256") and source["sha256"] != proof["sha256"])
            or (source.get("md5") and source["md5"] != proof.get("publisher_md5"))
        ):
            raise ValueError("Remote component lacks its publisher digest proof")
        receipts = envelope["object_receipts"]
        by_key = {record["key"]: record for record in receipts}
        unique = {
            f"{self.prefix}/{record['key']}": record for record in component["assets"].values()
        }
        if len(receipts) != len(by_key) or set(unique) != set(by_key):
            raise ValueError("Remote receipt must cover exactly its component objects")
        for object_key, record in unique.items():
            receipt = by_key[object_key]
            if receipt.get("sha256") != record["sha256"] or receipt.get("bytes") != record["bytes"]:
                raise ValueError("Remote component receipt does not match its assets")
            self.store.confirm_receipt(receipt)
        destination = self.components / f"{source['id']}.json"
        if destination.exists() and json.loads(destination.read_text()) != component:
            raise ValueError("Local component differs from saved remote receipt")
        _atomic_json(destination, component)
        _read_component(destination)
        component_receipt = envelope["component_receipt"]
        if component_receipt["key"] != f"{self.prefix}/components/{source['id']}.json" or (
            component_receipt["sha256"] != sha256_file(destination, self.check)
        ):
            raise ValueError("Remote component manifest checksum differs from receipt")
        self.store.confirm_receipt(component_receipt)
        with self.database:
            for receipt in receipts:
                self.database.execute(
                    "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                    (receipt["key"], json.dumps(receipt, sort_keys=True), self.run_id),
                )
            self.database.execute(
                "INSERT INTO components VALUES (?, ?, ?)",
                (source["id"], str(destination), json.dumps(component_receipt, sort_keys=True)),
            )
        return destination

    def _download_brir_huggingface(self, source, archive):
        """Supervise the official downloader; Rust threads cannot bypass our bound."""
        self.check()
        if archive.exists():
            return  # The caller still verifies its entire SHA-256 and byte count.
        work = archive.parent
        local_dir = work / "huggingface"
        local_dir.mkdir(exist_ok=True)
        for name in ("tmp",):
            (local_dir / name).mkdir(exist_ok=True)
        # An urllib partial is not a verified archive and cannot seed Xet's cache.
        archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)
        environment = {
            name: os.environ[name]
            for name in (
                "PATH",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "LD_LIBRARY_PATH",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
            )
            if name in os.environ
        }
        environment.update({"TMPDIR": str(local_dir / "tmp")})
        # No AWS/GMN/HF credentials or private-storage environment reaches this
        # public-data helper. SDK stdout/stderr may contain signed URLs: discard it.
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.download_nat_brir",
                "--source-path",
                source["path"],
                "--local-dir",
                str(local_dir),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        def check_scratch():
            self.check()
            occupied = 0
            for path in work.rglob("*"):
                # Xet atomically renames partials while we count; missing old
                # paths are harmless, and another check follows within a second.
                try:
                    if path.is_file():
                        occupied += path.stat().st_size
                except FileNotFoundError:
                    continue
            if occupied > self.reservations[source["id"]]:
                raise PreparationLimit(
                    "Hugging Face download exceeded its source scratch reservation"
                )

        try:
            while process.poll() is None:
                check_scratch()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
            check_scratch()
            if process.returncode != 0:
                raise URLError("Pinned Hugging Face download helper failed")
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        downloaded = local_dir / source["path"]
        if (
            not downloaded.is_file()
            or downloaded.is_symlink()
            or downloaded.resolve() != downloaded
            or downloaded.stat().st_size != source["bytes"]
        ):
            raise ValueError("Pinned Hugging Face download returned an invalid source path or size")
        os.replace(downloaded, archive)  # Same filesystem; no second archive-sized copy.

    def _prepare_source(self, source):
        """Producer-only local work: no S3 requests or SQLite access."""
        self.check()
        cap = (3 if source["kind"] == "brir" else 8) * 1024**3
        work = self.scratch / source["id"]
        work.mkdir(exist_ok=True)
        archive = work / ("source.tar" if source["kind"] == "brir" else "source.zip")
        print(
            json.dumps(
                {"stage": "source", "source": source["id"], "source_bytes": source["bytes"]}
            ),
            flush=True,
        )
        for attempt in range(3):
            try:
                if self.download_backend == "huggingface" and source["kind"] == "brir":
                    self._download_brir_huggingface(source, archive)
                proof = download_source(source, archive, self.check)
                break
            except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
                if attempt == 2 or (
                    isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504)
                ):
                    raise
                print(
                    json.dumps(
                        {"stage": "download_retry", "source": source["id"], "attempt": attempt + 2}
                    ),
                    flush=True,
                )
                for _ in range(4 * 2**attempt):
                    self.check()
                    time.sleep(0.25)
        _atomic_json(work / "source-receipt.json", proof)
        converted = work / "converted"
        component_path = converted / "manifest.json"
        if (
            component_path.exists()
            and json.loads(component_path.read_text()).get("status") == "complete"
        ):
            component = _read_component(component_path)
            if component["source"]["sha256"] != proof["sha256"]:
                raise ValueError("Scratch conversion source checksum changed")
        else:
            if converted.exists():
                # Only restart this stager's uncommitted conversion; preserve its diagnostic manifest.
                if component_path.exists():
                    shutil.copy2(component_path, work / "previous-conversion.json")
                shutil.rmtree(converted)
            kwargs = {
                "max_seconds": max(0.001, self.deadline - time.monotonic()),
                "max_output_bytes": cap,
                "external_check": self.check,
            }
            if source["kind"] == "brir":
                component = prepare_brir_shard(
                    archive, proof["sha256"], converted, source_member=source["path"], **kwargs
                )
            else:
                component = prepare_wham_noise(archive, proof["sha256"], converted, **kwargs)
            component["source"]["publisher_md5"] = proof["publisher_md5"]
            _atomic_json(component_path, component)
        expected = 1000 if source["kind"] == "brir" else 20000
        if component["converted_count"] != expected:
            raise ValueError("Completed conversion differs from the full publisher partition count")
        return {
            "source": source,
            "work": work,
            "converted": converted,
            "component": component,
            "component_path": component_path,
            "proof": proof,
        }

    def _publish_source(self, prepared):
        """Consumer-only verification/publication: all ledger mutations stay here."""
        source, work = prepared["source"], prepared["work"]
        component, converted = prepared["component"], prepared["converted"]
        component_path, proof = prepared["component_path"], prepared["proof"]
        self._upload(component, converted)
        destination = self.components / f"{source['id']}.json"
        if destination.exists() and sha256_file(destination) != sha256_file(component_path):
            raise ValueError("Previously saved component differs from this conversion")
        if not destination.exists():
            _atomic_json(destination, component)
        receipt = self.store.put_file(
            f"{self.prefix}/components/{source['id']}.json",
            destination,
            sha256_file(destination, self.check),
        )
        envelope = {
            "schema_version": 1,
            "configuration": self.identity,
            "source_verification": proof,
            "component": component,
            "component_receipt": receipt,
            "object_receipts": [
                self.object_receipt(key)[0]
                for key in sorted(
                    {f"{self.prefix}/{record['key']}" for record in component["assets"].values()}
                )
            ],
        }
        # Receipt contains server ETags and the original full-GET hashes; unlike the
        # immutable component its timestamps can be refreshed by a resumed attempt.
        self.store.put_json(f"{self.prefix}/receipts/{source['id']}.json", envelope)
        with self.database:
            self.database.execute(
                "INSERT INTO components VALUES (?, ?, ?)",
                (source["id"], str(destination), json.dumps(receipt, sort_keys=True)),
            )
        # Durable ledger + remote component + all full-GET receipts now exist.
        shutil.rmtree(work)
        return destination

    def _completed_source(self, source):
        existing = self._existing_component(source)
        return existing if existing is not None else self._restore_component(source)

    def stage_source(self, source):
        """Sequential compatibility path, using the same reservation/proof rules."""
        existing = self._completed_source(source)
        if existing is not None:
            return existing
        self._reserve_source(source)
        try:
            return self._publish_source(self._prepare_source(source))
        finally:
            self.reservations.pop(source["id"], None)

    def _stage_sources(self):
        selected = self.sources[: self.max_sources]
        if len({source["id"] for source in selected}) != len(selected):
            raise ValueError("Preparation sources must have unique identities")
        paths = {}
        remaining = iter(selected)

        def next_pending():
            for source in remaining:
                completed = self._completed_source(source)
                if completed is None:
                    return source
                paths[source["id"]] = completed
                print(json.dumps(self.status("partial")), flush=True)
            return None

        current = next_pending()
        if current is None:
            return [paths[source["id"]] for source in selected]
        self._reserve_source(current)
        producer = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nat-prepare"
        )
        future = producer.submit(self._prepare_source, current)
        try:
            while current is not None:
                prepared = future.result()
                following = next_pending()
                next_future = None
                if following is not None and self.prefetch_sources:
                    if self._reserve_source(following, optional=True):
                        next_future = producer.submit(self._prepare_source, following)
                # Only this consumer touches S3 publication and the SQLite ledger.
                paths[current["id"]] = self._publish_source(prepared)
                self.reservations.pop(current["id"])
                print(json.dumps(self.status("partial")), flush=True)
                if following is not None and next_future is None:
                    self._reserve_source(following)
                    next_future = producer.submit(self._prepare_source, following)
                current, future = following, next_future
        except BaseException:
            # The converter and downloader share this check; do not spend the
            # remaining runtime preparing another source after a consumer failure.
            self.cancelled.set()
            if future is not None:
                future.cancel()
            raise
        finally:
            producer.shutdown(wait=True, cancel_futures=True)
            self.reservations.clear()
        return [paths[source["id"]] for source in selected]

    def status(self, state, *, failure=None):
        rows = list(self.database.execute("SELECT manifest FROM components"))
        scene_count = noise_count = 0
        for row in rows:
            component = _read_component(Path(row[0]))
            scene_count += len(component["scenes"])
            noise_count += len(component["noise_files"])
        value = {
            "schema_version": 1,
            "status": state,
            "completed_components": len(rows),
            "expected_components": 71,
            "scene_count": scene_count,
            "noise_count": noise_count,
            "verified_object_bytes": sum(
                json.loads(row[0])["bytes"]
                for row in self.database.execute("SELECT receipt FROM objects")
            ),
            "updated_at": time.time(),
        }
        if failure is not None:
            value.update(_failure_diagnostic(failure))
        _atomic_json(self.output / "status.json", value)
        return value

    def run(self):
        try:
            paths = self._stage_sources()
            if len(paths) != 71:
                return self.status("partial")
            # New generation paths allow retry after an interrupted publication without overwrite.
            final = self.output / f"manifest-{self.run_id}.json"
            result = assemble_nat_manifest(
                paths[:-1],
                paths[-1],
                final,
                endpoint=self.endpoint,
                bucket=self.bucket,
                prefix=self.prefix,
                brir_inventory=self.inventory,
                max_seconds=max(0.001, self.deadline - time.monotonic()),
                max_total_object_bytes=self.max_upload_bytes,
                _remote_verifier=self.verify_enrolled,
            )
            receipt = self.store.put_file(
                f"{self.prefix}/manifest.json", final, result["manifest_sha256"]
            )
            _atomic_json(self.output / "final-receipt.json", receipt)
            report = {
                **self.status("complete"),
                "ready_for_training": True,
                "bucket": self.bucket,
                "endpoint": self.endpoint,
                "manifest": {field: receipt[field] for field in ("key", "bytes", "sha256")},
                "inventory_sha256": self.identity["inventory_sha256"],
                "brir_repository": BRIR_REPOSITORY,
                "brir_revision": BRIR_REVISION,
            }
            # Stable completion content supports idempotent publication after a crash.
            report.pop("updated_at")
            self.store.put_json(f"{self.prefix}/ready.json", report, immutable=True)
            _atomic_json(self.output / "status.json", report)
            return report
        except BaseException as error:
            self.status(
                "partial" if isinstance(error, PreparationLimit) else "failed", failure=error
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--max-runtime-seconds", type=float, default=36000)
    parser.add_argument("--max-sources", type=int, default=71)
    parser.add_argument("--max-scratch-bytes", type=int, default=100 * 1024**3)
    parser.add_argument("--max-upload-bytes", type=int, default=192 * 1024**3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=9.0)
    parser.add_argument(
        "--prefetch-sources",
        type=int,
        choices=(0, 1),
        default=1,
        help="Prepare one next source while uploading the current source",
    )
    parser.add_argument("--download-backend", choices=("urllib", "huggingface"), default="urllib")
    parser.add_argument("--result-path", type=Path, default=os.environ.get("GMN_RESULT_PATH"))
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("/output/nat-staging-diagnostics")
    )
    args = parser.parse_args()

    def save_result(result):
        paths = {
            Path(path) for path in (args.result_path, os.environ.get("GMN_RESULT_PATH")) if path
        }
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, result)

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        region_name=os.environ.get("AWS_REGION", "auto"),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=30,
            max_pool_connections=args.workers + 2,
        ),
    )
    stager = None
    try:
        stager = NatStager(
            args.output_dir,
            client,
            endpoint=args.endpoint,
            bucket=args.bucket,
            prefix=args.prefix,
            inventory=args.inventory,
            max_runtime_seconds=args.max_runtime_seconds,
            max_sources=args.max_sources,
            max_scratch_bytes=args.max_scratch_bytes,
            max_upload_bytes=args.max_upload_bytes,
            workers=args.workers,
            requests_per_second=args.requests_per_second,
            prefetch_sources=args.prefetch_sources,
            download_backend=args.download_backend,
        )
        result = stager.run()
        save_result(result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        # Never print unfiltered network exceptions, credentials or redirected URLs.
        save_result({"status": "failed", "ready_for_training": False, **_failure_diagnostic(error)})
        print(json.dumps(_failure_diagnostic(error)), flush=True)
        parser.exit(
            2,
            f"Nat staging stopped ({type(error).__name__}); inspect retained local status and receipts.\n",
        )
    finally:
        if stager is not None:
            try:
                stager.preserve_diagnostics(args.diagnostics_dir)
            except Exception as error:
                print(
                    json.dumps(
                        {"diagnostic_capture": "failed", "failure_type": type(error).__name__}
                    ),
                    flush=True,
                )
            stager.close()


if __name__ == "__main__":
    main()
