"""Prepare the pinned AudioSet research mirror on a CPU host, without loading Torch.

Archives are streamed, never extracted by pathname or retained. FFmpeg handles
resampling/mono conversion; an already canonical WAV uses a lossless copy. Each
archive is SHA256 verified even when a pilot cap stops decoding early.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

REVISION = "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef"
CATALOG = Path(__file__).resolve().parents[1] / "configs" / "data" / "audioset-source-catalog.json"
SAMPLE_RATE = 16_000
MIN_SAMPLES = 145_440
MAX_MEMBER_BYTES = 32 * 1024**2
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    size: int
    split: str
    archive_directory: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("Unsafe source name")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Source split must be train, val or test")
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256) or self.size <= 0:
            raise ValueError("Every source needs a pinned SHA256 and positive byte size")
        if self.archive_directory not in {"unbal_train", "bal_train", "eval"}:
            raise ValueError("Unexpected AudioSet archive directory")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def builtin_sources(catalog: Path = CATALOG) -> list[Source]:
    data = json.loads(catalog.read_text())
    if data["revision"] != REVISION:
        raise ValueError("Catalog revision differs from the pinned WavJEPA data revision")
    sources = []
    for item in data["files"]:
        match = re.fullmatch(r"data/(unbal_train|bal_train|eval)(\d+)\.tar", item["path"])
        if not match:
            continue
        directory = match[1]
        sources.append(
            Source(
                name=Path(item["path"]).stem,
                url=item["url"],
                sha256=item["sha256"],
                size=item["size"],
                split={"unbal_train": "train", "bal_train": "val", "eval": "test"}[directory],
                archive_directory=directory,
            )
        )
    # Reserve held-out recordings first if a source mirror accidentally overlaps.
    return sorted(
        sources, key=lambda item: ({"test": 0, "val": 1, "train": 2}[item.split], item.name)
    )


class HashingReader:
    def __init__(self, handle, expected_size: int):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0
        self.expected_size = expected_size

    def read(self, size=-1):
        data = self.handle.read(size)
        self.size += len(data)
        if self.size > self.expected_size:
            raise ValueError("Archive exceeded its pinned byte size")
        self.digest.update(data)
        return data

    def finish(self, expected_sha256: str):
        while self.read(1024**2):
            pass
        if self.size != self.expected_size or self.digest.hexdigest() != expected_sha256:
            raise ValueError(
                "Archive byte size or SHA256 mismatch; outputs excluded from manifests"
            )


def member_identity(member: tarfile.TarInfo, source: Source) -> str:
    path = PurePosixPath(member.name)
    if (
        not member.isfile()
        or member.issparse()
        or member.size <= 0
        or member.size > MAX_MEMBER_BYTES
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in member.name
        or len(path.parts) != 3
        or path.parts[:2] != ("audio", source.archive_directory)
        or path.suffix.lower() not in {".flac", ".wav"}
        or not VIDEO_ID.fullmatch(path.stem)
    ):
        raise ValueError(f"Unsafe or unexpected archive member: {member.name!r}")
    return path.stem


def inspect_wav(path: Path, minimum_samples: int, maximum_samples: int) -> int:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getcomptype(),
        ) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("Expected mono PCM16 16 kHz WAV")
        samples = handle.getnframes()
        if not minimum_samples <= samples <= maximum_samples:
            raise ValueError(
                f"Audio has {samples} samples; expected {minimum_samples}..{maximum_samples}; no padding allowed"
            )
        handle.setpos(samples - 1)
        if len(handle.readframes(2)) != 2:
            raise ValueError("Truncated or misaligned WAV payload")
        return samples


def convert_audio(
    input_path: Path, output_path: Path, *, ffmpeg: str, minimum_samples: int, maximum_seconds: int
) -> dict:
    """Decode bounded local input; never pad or accept silently truncated long clips."""
    temporary = output_path.with_suffix(".wav.partial")
    maximum_samples = maximum_seconds * SAMPLE_RATE
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canonical = False
        if input_path.suffix == ".wav":
            try:
                inspect_wav(input_path, minimum_samples, maximum_samples)
                canonical = True
            except (ValueError, EOFError, wave.Error):
                pass
        if canonical:
            shutil.copyfile(input_path, temporary)
        else:
            # Decode one second past the allowed maximum; rejecting its resulting
            # sample count distinguishes an overlong recording from a valid clip.
            command = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                "flac" if input_path.suffix == ".flac" else "wav",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-t",
                str(maximum_seconds + 1),
                "-threads",
                "1",
                "-f",
                "wav",
                "-y",
                str(temporary),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=120)
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or b"").decode("utf-8", errors="replace").lower()
                infrastructure_errors = (
                    "no space left on device",
                    "input/output error",
                    "permission denied",
                    "cannot allocate memory",
                    "too many open files",
                    "read-only file system",
                    "resource temporarily unavailable",
                    "no such file or directory",
                )
                if error.returncode < 0 or any(
                    message in detail for message in infrastructure_errors
                ):
                    raise OSError(
                        f"FFmpeg infrastructure failure; retry after fixing the host: {detail[:1000]}"
                    ) from error
                raise
        samples = inspect_wav(temporary, minimum_samples, maximum_samples)
        digest = sha256_file(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
        return {"samples": samples, "sha256": digest, "bytes": output_path.stat().st_size}
    finally:
        temporary.unlink(missing_ok=True)
        input_path.unlink(missing_ok=True)


def connect_ledger(output: Path):
    db = sqlite3.connect(output / "preparation.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            name TEXT PRIMARY KEY, descriptor TEXT NOT NULL, status TEXT NOT NULL,
            verified_sha256 TEXT, error TEXT, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS clips (
            recording_id TEXT PRIMARY KEY, source TEXT NOT NULL, member TEXT NOT NULL,
            split TEXT NOT NULL, path TEXT NOT NULL, samples INTEGER NOT NULL,
            sha256 TEXT NOT NULL, input_sha256 TEXT NOT NULL, bytes INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS members (
            source TEXT NOT NULL, member TEXT NOT NULL, status TEXT NOT NULL,
            error TEXT, PRIMARY KEY(source, member));
    """)
    columns = {row[1] for row in db.execute("PRAGMA table_info(clips)")}
    if "verified" not in columns:
        db.execute("ALTER TABLE clips ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        db.execute(
            "UPDATE clips SET verified=1 WHERE source IN (SELECT name FROM sources WHERE verified_sha256 IS NOT NULL)"
        )
        db.commit()
    return db


def record_failure(db, source: Source, member: str, error: str):
    db.execute(
        "INSERT OR REPLACE INTO members VALUES (?, ?, 'rejected', ?)",
        (source.name, member, error[:1000]),
    )
    db.commit()


def open_source(source: Source):
    if source.url.startswith("https://"):
        return urllib.request.urlopen(
            urllib.request.Request(
                source.url, headers={"User-Agent": "h-wavjepa-data-preparation/1"}
            ),
            timeout=60,
        )
    if "://" in source.url:
        raise ValueError("Only HTTPS or explicitly supplied local source paths are supported")
    return Path(source.url).open("rb")


def prepare_source(
    source: Source,
    output: Path,
    db,
    *,
    workers=2,
    max_clips=None,
    minimum_samples=MIN_SAMPLES,
    maximum_seconds=30,
    minimum_free_bytes=1024**3,
    ffmpeg="ffmpeg",
) -> dict:
    descriptor = json.dumps(asdict(source), sort_keys=True)
    previous = db.execute(
        "SELECT descriptor, status, verified_sha256 FROM sources WHERE name=?", (source.name,)
    ).fetchone()
    if previous and previous[0] != descriptor:
        raise ValueError(f"Source {source.name} changed; use a new output directory")
    if previous and previous[1] == "complete":
        return {"source": source.name, "status": "already_complete", "new_attempts": 0}
    if previous and previous[1] == "invalid":
        raise ValueError(
            f"Source {source.name} failed integrity verification; use a new output directory"
        )
    if previous and previous[1] == "conflict":
        raise ValueError(
            f"Source {source.name} has an unresolved cross-split recording-ID collision; inspect provenance before proceeding"
        )
    previously_verified = previous[2] if previous else None
    db.execute(
        "INSERT OR REPLACE INTO sources VALUES (?, ?, 'processing', ?, NULL, ?)",
        (source.name, descriptor, previously_verified, time.time()),
    )
    db.commit()
    pending = {}
    pending_ids = set()
    attempts = 0
    capped = False
    tempdir = output / ".temporary"
    tempdir.mkdir(exist_ok=True)

    def collect(done):
        infrastructure_failure = None
        for future in done:
            member, recording_id, destination, input_sha256 = pending.pop(future)
            pending_ids.remove(recording_id)
            try:
                info = future.result()
            except (OSError, subprocess.TimeoutExpired) as error:
                # A full disk, missing executable or timeout says nothing about
                # whether this recording is valid. Leave the member retryable.
                infrastructure_failure = error
                continue
            except (ValueError, EOFError, wave.Error, subprocess.CalledProcessError) as error:
                record_failure(db, source, member, f"{type(error).__name__}: {error}")
                continue
            db.execute(
                "INSERT INTO clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    recording_id,
                    source.name,
                    member,
                    source.split,
                    destination.relative_to(output).as_posix(),
                    info["samples"],
                    info["sha256"],
                    input_sha256,
                    info["bytes"],
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO members VALUES (?, ?, 'accepted', NULL)",
                (source.name, member),
            )
            db.commit()
        if infrastructure_failure is not None:
            raise infrastructure_failure

    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                with open_source(source) as raw:
                    reader = HashingReader(raw, source.size)
                    with tarfile.open(fileobj=reader, mode="r|") as archive:
                        for member in archive:
                            archive.members.clear()
                            if max_clips is not None and attempts >= max_clips:
                                capped = True
                                break
                            if member.isdir():
                                continue
                            if db.execute(
                                "SELECT 1 FROM members WHERE source=? AND member=?",
                                (source.name, member.name),
                            ).fetchone():
                                continue
                            try:
                                recording_id = member_identity(member, source)
                            except ValueError as error:
                                record_failure(db, source, member.name, str(error))
                                continue
                            existing = db.execute(
                                "SELECT split FROM clips WHERE recording_id=?", (recording_id,)
                            ).fetchone()
                            if existing or recording_id in pending_ids:
                                # Never allow one video ID to enter two model splits.
                                reason = f"Duplicate recording ID {recording_id}; existing split={existing[0] if existing else source.split}"
                                record_failure(db, source, member.name, reason)
                                if existing and existing[0] != source.split:
                                    raise ValueError(
                                        f"Cross-split recording-ID collision: {reason}; inspect source provenance before proceeding"
                                    )
                                continue
                            if (
                                shutil.disk_usage(output).free
                                < minimum_free_bytes
                                + MAX_MEMBER_BYTES
                                + (maximum_seconds + 1) * SAMPLE_RATE * 2
                            ):
                                raise OSError(
                                    "Free disk space reached the configured safety floor; resume after adding storage"
                                )
                            input_path = None
                            try:
                                with tempfile.NamedTemporaryFile(
                                    dir=tempdir,
                                    suffix=PurePosixPath(member.name).suffix.lower(),
                                    delete=False,
                                ) as handle:
                                    input_path = Path(handle.name)
                                    digest = hashlib.sha256()
                                    remaining = member.size
                                    with archive.extractfile(member) as payload:
                                        while remaining:
                                            chunk = payload.read(min(1024**2, remaining))
                                            if not chunk:
                                                raise ValueError("Truncated archive member")
                                            handle.write(chunk)
                                            digest.update(chunk)
                                            remaining -= len(chunk)
                                folder = hashlib.sha256(recording_id.encode()).hexdigest()[:2]
                                destination = output / "audio" / folder / f"{recording_id}.wav"
                                future = executor.submit(
                                    convert_audio,
                                    input_path,
                                    destination,
                                    ffmpeg=ffmpeg,
                                    minimum_samples=minimum_samples,
                                    maximum_seconds=maximum_seconds,
                                )
                            except BaseException:
                                if input_path is not None:
                                    input_path.unlink(missing_ok=True)
                                raise
                            pending[future] = (
                                member.name,
                                recording_id,
                                destination,
                                digest.hexdigest(),
                            )
                            pending_ids.add(recording_id)
                            attempts += 1
                            if len(pending) >= workers:
                                done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                                collect(done)
                            if attempts % 100 == 0:
                                print(
                                    json.dumps(
                                        {
                                            "source": source.name,
                                            "new_attempts": attempts,
                                            "bytes_streamed": reader.size,
                                        }
                                    ),
                                    flush=True,
                                )
                    reader.finish(source.sha256)
            finally:
                if pending:
                    collect(futures.wait(pending)[0])
        status = "partial" if capped else "complete"
        db.execute("UPDATE clips SET verified=1 WHERE source=?", (source.name,))
        db.execute(
            "UPDATE sources SET status=?, verified_sha256=?, error=NULL, updated=? WHERE name=?",
            (status, source.sha256, time.time(), source.name),
        )
        db.commit()
        return {
            "source": source.name,
            "status": status,
            "new_attempts": attempts,
            "verified_sha256": source.sha256,
        }
    except Exception as error:
        integrity_failure = "SHA256 mismatch" in str(error) or "pinned byte size" in str(error)
        conflict = "Cross-split recording-ID collision" in str(error)
        db.execute(
            "UPDATE sources SET status=?, verified_sha256=?, error=?, updated=? WHERE name=?",
            (
                "invalid" if integrity_failure else "conflict" if conflict else "interrupted",
                None if integrity_failure else previously_verified,
                str(error)[:1000],
                time.time(),
                source.name,
            ),
        )
        db.commit()
        raise


def export_manifests(output: Path, db, *, verify_files=False) -> dict:
    report = {"revision": REVISION, "splits": {}, "sources": [], "rejections": {}}
    for split in ("train", "val", "test"):
        path = output / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        count = samples = byte_count = 0
        with temporary.open("w", encoding="utf-8") as handle:
            rows = db.execute(
                """SELECT c.recording_id, c.path, c.samples, c.sha256, c.bytes
                FROM clips c JOIN sources s ON c.source=s.name
                WHERE c.split=? AND s.verified_sha256 IS NOT NULL AND c.verified=1 ORDER BY c.recording_id""",
                (split,),
            )
            for recording_id, relative, frames, digest, size in rows:
                audio = output / relative
                if not audio.is_file() or audio.stat().st_size != size:
                    raise ValueError(f"Prepared output missing or changed: {audio}")
                if verify_files and sha256_file(audio) != digest:
                    raise ValueError(f"Prepared output checksum mismatch: {audio}")
                handle.write(
                    json.dumps({"path": relative, "recording_id": recording_id}, sort_keys=True)
                    + "\n"
                )
                count += 1
                samples += frames
                byte_count += size
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        report["splits"][split] = {
            "clips": count,
            "samples": samples,
            "bytes": byte_count,
            "manifest_sha256": sha256_file(path),
        }
    provenance = output / "clips.provenance.jsonl"
    temporary = provenance.with_suffix(".jsonl.tmp")
    cursor = db.execute(
        "SELECT c.* FROM clips c JOIN sources s ON c.source=s.name WHERE s.verified_sha256 IS NOT NULL AND c.verified=1 ORDER BY c.recording_id"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        names = [column[0] for column in cursor.description]
        for row in cursor:
            handle.write(json.dumps(dict(zip(names, row, strict=True)), sort_keys=True) + "\n")
    temporary.replace(provenance)
    for name, descriptor, status, digest, error, _ in db.execute(
        "SELECT * FROM sources ORDER BY name"
    ):
        report["sources"].append(
            {**json.loads(descriptor), "status": status, "verified_sha256": digest, "error": error}
        )
    failures = output / "rejections.jsonl"
    temporary = failures.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for source, member, error in db.execute(
            "SELECT source, member, error FROM members WHERE status='rejected' ORDER BY source, member"
        ):
            report["rejections"][source] = report["rejections"].get(source, 0) + 1
            handle.write(json.dumps({"source": source, "member": member, "error": error}) + "\n")
    temporary.replace(failures)
    report["provenance_sha256"] = sha256_file(provenance)
    report["rejections_sha256"] = sha256_file(failures)
    runtime_path = output / "preparation-runtime.json"
    if runtime_path.exists():
        report["runtime"] = json.loads(runtime_path.read_text())
    metadata_path = output / "metadata" / "provenance.json"
    if metadata_path.exists():
        report["metadata"] = json.loads(metadata_path.read_text())
    atomic_json(output / "preparation-report.json", report)
    return report


def prepare_metadata(catalog: Path, output: Path) -> dict:
    """Retain the original segment/label CSVs without reapplying their offsets."""
    data = json.loads(catalog.read_text())
    if data["revision"] != REVISION:
        raise ValueError("Metadata catalog revision differs from the pinned revision")
    allowed = {"unbalanced_train_segments.csv", "balanced_train_segments.csv", "eval_segments.csv"}
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    report = {"revision": REVISION, "files": []}
    for item in data["files"]:
        name = Path(item["path"]).name
        if item["path"] != f"data/{name}" or name not in allowed:
            continue
        if not 0 < item["size"] < 150 * 1024**2:
            raise ValueError("Unexpected metadata file size")
        destination = metadata / name
        if not destination.exists() or sha256_file(destination) != item["sha256"]:
            temporary = destination.with_suffix(".csv.partial")
            source = Source(
                name.replace(".", "_"), item["url"], item["sha256"], item["size"], "test", "eval"
            )
            try:
                with open_source(source) as raw, temporary.open("wb") as handle:
                    reader = HashingReader(raw, source.size)
                    while chunk := reader.read(1024**2):
                        handle.write(chunk)
                    reader.finish(source.sha256)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        report["files"].append({**item, "local_path": destination.relative_to(output).as_posix()})
    if {Path(item["path"]).name for item in report["files"]} != allowed:
        raise ValueError("Catalog must include all three official split metadata CSVs")
    atomic_json(metadata / "provenance.json", report)
    return report


def record_ffmpeg_version(output: Path, ffmpeg: str) -> None:
    version = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    runtime = {
        "ffmpeg_version": version,
        "ffmpeg_executable": str(Path(shutil.which(ffmpeg)).resolve()),
        "preparation_script_sha256": sha256_file(Path(__file__)),
    }
    path = output / "preparation-runtime.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["ffmpeg_version"] != version:
            raise ValueError(
                "FFmpeg version changed; use a new output directory to keep preprocessing fixed"
            )
        runtime["previous_script_sha256"] = previous.get("preparation_script_sha256")
    atomic_json(path, runtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument(
        "--source",
        action="append",
        help="Exact shard stem; repeat to select a pilot. Default: all pinned shards.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        help="Maximum new audio conversion attempts PER source this invocation; still downloads/verifies its full archive.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download and verify the three pinned official metadata CSVs only (~104 MB).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rebuild manifests and SHA256-check prepared WAVs without downloading.",
    )
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.workers > 32
        or args.minimum_free_gib < 0
        or (args.max_clips is not None and args.max_clips < 1)
    ):
        parser.error("workers must be 1..32, free-space floor nonnegative and max-clips positive")
    if args.metadata_only and (args.verify_only or args.source):
        parser.error("--metadata-only cannot be combined with --verify-only or --source")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preparation process owns this output directory") from error
        tempdir = output / ".temporary"
        if tempdir.exists():
            for orphan in tempdir.iterdir():
                if orphan.is_file() and orphan.suffix in {".flac", ".wav"}:
                    orphan.unlink()
        if args.metadata_only:
            print(json.dumps(prepare_metadata(args.catalog, output), indent=2), flush=True)
            return
        configuration = {
            "schema": 2,
            "revision": REVISION,
            "sample_rate": SAMPLE_RATE,
            "minimum_samples": MIN_SAMPLES,
            "maximum_seconds": 30,
            "split_policy": "unbalanced=train, balanced=val, official_eval=test; cross-split recording-ID collision is fatal",
        }
        config_path = output / "preparation-config.json"
        if config_path.exists():
            previous_config = json.loads(config_path.read_text())
            if previous_config.get("schema") == 1:
                previous_config.update(schema=2, split_policy=configuration["split_policy"])
            if previous_config != configuration:
                raise ValueError("Preparation configuration changed; use a new output directory")
        atomic_json(config_path, configuration)
        with contextlib.closing(connect_ledger(output)) as db:
            if not args.verify_only:
                if shutil.which(args.ffmpeg) is None:
                    parser.error(
                        "FFmpeg is required; install it on the CPU host before downloading"
                    )
                record_ffmpeg_version(output, args.ffmpeg)
                sources = builtin_sources(args.catalog)
                if args.source:
                    selected = set(args.source)
                    unknown = selected - {item.name for item in sources}
                    if unknown:
                        parser.error(f"Unknown source shards: {sorted(unknown)}")
                    sources = [item for item in sources if item.name in selected]
                try:
                    for source in sources:
                        print(
                            json.dumps(
                                prepare_source(
                                    source,
                                    output,
                                    db,
                                    workers=args.workers,
                                    max_clips=args.max_clips,
                                    minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
                                    ffmpeg=args.ffmpeg,
                                )
                            ),
                            flush=True,
                        )
                finally:
                    export_manifests(output, db)
            report = export_manifests(output, db, verify_files=args.verify_only)
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "splits": report["splits"],
                        "rejections": report["rejections"],
                    },
                    indent=2,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
