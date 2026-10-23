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
import sqlite3
import tempfile
import threading
import time
import uuid
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
        if not math.isfinite(requests_per_second) or not 0 < requests_per_second <= 8:
            raise ValueError("S3 request rate must be in (0, 8] per second")
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
                    self.next_request = max(time.monotonic(), self.next_request) + self.interval
                self._wait(delay)
                if offset is not None:
                    body.seek(offset)
                try:
                    return operation(**kwargs)
                except Exception as error:
                    response = getattr(error, "response", {})
                    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    code = str(response.get("Error", {}).get("Code", ""))
                    if attempt == 5 or (
                        status not in (429, 500, 502, 503, 504)
                        and code not in {"SlowDown", "Throttling", "RequestTimeout"}
                    ):
                        raise
                    self._wait(min(30, 2**attempt))

        return call

    def _wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(max(0, min(0.25, end - time.monotonic())))
        self.check()


class NatStore(Store):
    """Two requests per fresh payload: conditional PUT, then full GET proof."""

    def verify(self, key, digest, size):
        self.check()
        response = self.client.get_object(Bucket=self.bucket, Key=key)
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
            response = getattr(error, "response", {})
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412 and str(
                response.get("Error", {}).get("Code")
            ) not in {"412", "PreconditionFailed"}:
                raise
        return self.verify(key, digest, size)

    def _put_immutable(self, **kwargs):
        try:
            return self.client.put_object(Bucket=self.bucket, IfNoneMatch="*", **kwargs)
        except Exception as error:
            response = getattr(error, "response", {})
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412 and str(
                response.get("Error", {}).get("Code")
            ) not in {"412", "PreconditionFailed"}:
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
        workers=4,
        requests_per_second=8.0,
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

    def _space_check(self, source, converted_cap):
        occupied = sum(path.stat().st_size for path in self.scratch.rglob("*") if path.is_file())
        needed = source["bytes"] + converted_cap
        if occupied + needed > self.max_scratch_bytes:
            raise PreparationLimit("Nat scratch byte ceiling would be exceeded")
        if shutil.disk_usage(self.scratch).free < needed + 1024**3:
            raise PreparationLimit("Insufficient scratch disk for the next bounded archive")

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
        # Only workers futures are outstanding; SQLite enrollment stays on the main thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            for offset in range(0, len(pending), self.workers):
                self.check()
                futures = [
                    executor.submit(self.store.put_payload, *item)
                    for item in pending[offset : offset + self.workers]
                ]
                for future in futures:
                    receipt = future.result()
                    with self.database:
                        self.database.execute(
                            "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                            (receipt["key"], json.dumps(receipt, sort_keys=True), self.run_id),
                        )

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

    def stage_source(self, source):
        existing = self._existing_component(source)
        if existing is not None:
            return existing
        restored = self._restore_component(source)
        if restored is not None:
            return restored
        self.check()
        cap = (3 if source["kind"] == "brir" else 8) * 1024**3
        work = self.scratch / source["id"]
        work.mkdir(exist_ok=True)
        archive = work / ("source.tar" if source["kind"] == "brir" else "source.zip")
        self._space_check(source, cap)
        print(
            json.dumps(
                {"stage": "source", "source": source["id"], "source_bytes": source["bytes"]}
            ),
            flush=True,
        )
        for attempt in range(3):
            try:
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
            # Exception type only: SDK errors can contain signed URLs or auth details.
            value["failure_type"] = type(failure).__name__
        _atomic_json(self.output / "status.json", value)
        return value

    def run(self):
        try:
            paths = []
            for source in self.sources[: self.max_sources]:
                paths.append(self.stage_source(source))
                progress = self.status("partial")
                print(json.dumps(progress), flush=True)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
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
        )
        result = stager.run()
        save_result(result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        # Never print unfiltered network exceptions, credentials or redirected URLs.
        save_result(
            {"status": "failed", "ready_for_training": False, "failure_type": type(error).__name__}
        )
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
