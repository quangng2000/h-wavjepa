"""Run one bounded four-H100 segment; never rent, restart, or top up compute.

The orchestrator must check the organization's balance before EACH launch and
submit the printed platform limit with max_restarts=0. This process cannot read
account balances using bucket credentials. --plan-only does no network I/O.

Use a frozen, completed staging index and a successful streaming GPU report for
the spawn-worker path, unless the user explicitly waives that separate preflight.
A waiver records its reason and never disables training validation or checkpoint integrity. Resume restores full Lightning state but restarts data
iterators; --acknowledge-stream-restart makes this limitation explicit.
The data seed is base_data_seed + resumed optimizer step; the model seed and
LR/EMA schedules stay fixed. This changes restart order, not sample-exact resume.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import ipaddress
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.segment_checkpoint import (  # noqa: E402
    SHA,
    CheckpointRepository,
    checkpoint_client,
    minimum_checkpoint_collective_timeout,
)
from src.utils.runtime import atomic_json, file_sha256  # noqa: E402


@dataclass(frozen=True)
class SegmentBudget:
    segment_hours: float
    budget_usd: float
    maximum_rate_usd_per_hour: float
    finalization_reserve_seconds: float = 1800
    transfer_timeout_seconds: float = 600

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError(
                "Segment budget, rate, duration and margins must be finite and positive"
            )
        if self.segment_hours > 11.5:
            raise ValueError("Segments must be <=11.5 hours, leaving room within a 12-hour job cap")
        if self.finalization_reserve_seconds < 2 * self.transfer_timeout_seconds + 120:
            raise ValueError(
                "Reserve must cover pending and final checkpoint transfers plus validation/serialization/exit"
            )
        if self.finalization_reserve_seconds >= self.seconds:
            raise ValueError("Finalization reserve leaves no training time")
        if self.maximum_platform_cost_usd > self.budget_usd:
            raise ValueError(
                "Requested segment and platform margin exceed the allocated spending ceiling"
            )

    @property
    def seconds(self):
        return self.segment_hours * 3600

    @property
    def platform_max_minutes(self):
        return math.ceil(self.segment_hours * 60) + 5

    @property
    def maximum_platform_cost_usd(self):
        return self.platform_max_minutes / 60 * self.maximum_rate_usd_per_hour

    def report(self):
        return asdict(self) | {
            "minimum_ddp_timeout_seconds": minimum_checkpoint_collective_timeout(
                self.transfer_timeout_seconds
            ),
            "platform_max_duration_minutes": self.platform_max_minutes,
            "platform_max_restarts": 0,
            "platform_resume": "none",
            "maximum_platform_compute_cost_usd": self.maximum_platform_cost_usd,
            "account_balance_checked_by_runner": False,
            "separate_build_and_storage_charges_included": False,
            "finalization_reserve_includes": [
                "full validation at Timer stop",
                "checkpoint serialization",
                "upload and complete SHA256 readback",
                "process teardown",
            ],
        }


def verify_streaming_report(
    path: Path,
    *,
    expected_index_sha256: str,
    temporal: bool = False,
    nat: bool = False,
    naturalistic_manifest_sha256: str | None = None,
):
    """Bind production to the tested model, completed corpus and durable handoff."""
    from scripts.benchmark_streaming_gpu import (
        _require_nat_training_evidence,
        require_architecture,
        temporal_loss_metrics,
    )

    report = json.loads(path.read_text())
    temporal = temporal or nat
    global_batch = 16 if nat else 64
    variant = (
        "nat_temporal_aggregation" if nat else ("temporal_aggregation" if temporal else "pointwise")
    )
    if not nat and (
        report.get("model_variant") == "nat_temporal_aggregation"
        or report.get("architecture", {}).get("input_channels", 1) != 1
        or report.get("architecture", {}).get("masked_crops_per_recording", 0) != 0
    ):
        raise ValueError("A Nat hardware report requires an explicit --nat production plan")
    ranks, validation = report.get("ranks", []), report.get("validation", [])
    frozen, checkpoint = report.get("frozen_index", {}), report.get("checkpoint", {})
    durable = checkpoint.get("durable_s3", {})
    receipt = durable.get("receipt", {})
    try:
        validation_clips = frozen["split_counts"]["val"]
        used_clips = validation_clips // global_batch * global_batch
        steps = ranks[0]["optimizer_steps"]
        if (
            not SHA.fullmatch(expected_index_sha256)
            or report.get("ok") is not True
            or report.get("model_variant") != variant
            or report.get("data_worker_start_method") != "spawn"
            or report.get("workers_per_rank") != 1
            or report.get("cache_refills_observed_on_all_ranks") is not True
            or report.get("durable_checkpoint_required") is not True
            or frozen.get("status") != "complete"
            or frozen.get("verified_sources") != 889
            or frozen.get("frozen_index_sha256") != expected_index_sha256
            or frozen.get("source_index_sha256") != expected_index_sha256
            or used_clips <= 0
            or frozen.get("validation_used_clips") != used_clips
            or frozen.get("validation_clips") != validation_clips
            or frozen.get("validation_excluded_tail_clips") != validation_clips - used_clips
            or checkpoint.get("strict_tensor_and_metadata_roundtrip") is not True
            or checkpoint.get("bytes", 0) <= 0
            or len(ranks) != 4
            or len(validation) != 4
            or type(steps) is not int
            or steps <= 0
            or any(
                durable.get(field) is not True
                for field in (
                    "complete_remote_readback_verified",
                    "download_sha256_verified",
                    "strict_tensor_and_metadata_roundtrip",
                )
            )
            or receipt.get("owner", {}).get("index_sha256") != expected_index_sha256
            or (nat and receipt.get("owner", {}).get("global_batch") != global_batch)
            or (
                nat
                and (
                    not isinstance(naturalistic_manifest_sha256, str)
                    or not SHA.fullmatch(naturalistic_manifest_sha256)
                    or frozen.get("naturalistic_manifest_sha256") != naturalistic_manifest_sha256
                )
            )
            or receipt.get("checkpoint", {}).get("step") != steps
            or receipt.get("checkpoint", {}).get("bytes") != checkpoint["bytes"]
        ):
            raise ValueError("Incomplete or mismatched streaming acceptance evidence")
        for item in (report, checkpoint, durable):
            require_architecture(item.get("architecture", {}), temporal=temporal, nat=nat)
        for rank, (train, val) in enumerate(zip(ranks, validation, strict=True)):
            for row in (train, val):
                require_architecture(row.get("architecture", {}), temporal=temporal, nat=nat)
                if (
                    row.get("rank") != rank
                    or row.get("world_size") != 4
                    or row.get("optimizer_steps") != steps
                    or row.get("ema_updates") != steps
                    or row.get("frozen_index_sha256") != expected_index_sha256
                    or (
                        nat
                        and row.get("naturalistic_manifest_sha256") != naturalistic_manifest_sha256
                    )
                ):
                    raise ValueError("Rank topology, progress or frozen corpus mismatch")
            if (
                train.get("global_batch") != global_batch
                or train.get("data_worker_start_method") != "spawn"
                or train.get("workers_per_rank") != 1
                or train.get("online_weights_changed") is not True
                or train.get("all_model_parameters_finite") is not True
                or train.get("cache_refills_observed_after_warmup") is not True
                or train["rank_cache_at_train_end"]["verified_files"]
                <= train["rank_cache_at_warmup"]["verified_files"]
                or val.get("clips") != used_clips // 4
                or val.get("global_used_clips") != used_clips
                or val.get("global_excluded_tail_clips") != validation_clips - used_clips
                or not val.get("metrics")
                or any(not math.isfinite(value) for value in val["metrics"].values())
            ):
                raise ValueError("Rank lacks finite updates, cache refill or full validation")
            if nat:
                if train.get("global_masked_crops") != 128:
                    raise ValueError("Nat report does not prove 128 independent masked crops")
                _require_nat_training_evidence(train, naturalistic_manifest_sha256)
            if temporal:
                samples = train.get("temporal_training_samples", [])
                if not samples or samples[0].get("optimizer_step") != 1:
                    raise ValueError("Temporal training loss evidence is missing")
                for sample in samples:
                    if not 1 <= sample.get("optimizer_step", 0) <= steps:
                        raise ValueError("Temporal training sample is outside the probe")
                    temporal_loss_metrics(
                        {f"train/{key}_step": value for key, value in sample["metrics"].items()},
                        stage="train",
                    )
                temporal_loss_metrics(val["metrics"], stage="val")
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Production requires a matching successful four-GPU streaming probe: " + str(error)
        ) from error
    return file_sha256(path)


def launch_topology(args):
    """Supported GPU layouts; multihost launches require a pinned resume."""
    nodes = getattr(args, "nnodes", 1)
    processes = getattr(args, "nproc_per_node", 4)
    rank = getattr(args, "node_rank", 0)
    port = getattr(args, "master_port", 29500)
    if (
        any(type(value) is not int for value in (nodes, processes, rank, port))
        or (nodes, processes) not in {(1, 4), (1, 8), (4, 1)}
        or not 0 <= rank < nodes
        or not 1 <= port <= 65535
    ):
        raise ValueError("Use one node with four/eight ranks or four nodes with one rank each")
    digest = getattr(args, "resume_sha256", None)
    started = getattr(args, "shared_started_at", None)
    address = getattr(args, "master_addr", None)
    if digest is not None and (not isinstance(digest, str) or not SHA.fullmatch(digest)):
        raise ValueError("Resume SHA256 must identify the exact immutable checkpoint")
    if nodes > 1:
        if (
            not getattr(args, "resume", False)
            or digest is None
            or getattr(args, "curriculum_source_prefix", None) is not None
        ):
            raise ValueError(
                "Multihost training requires --resume and --resume-sha256; no prefix initialization or phase import"
            )
        try:
            host = ipaddress.ip_address(address)
        except ValueError:
            raise ValueError("Multihost training requires a private-network master IP") from None
        private_ipv4 = (
            host.version == 4
            and not host.is_loopback
            and not host.is_unspecified
            and (
                host in ipaddress.ip_network("10.0.0.0/8")
                or host in ipaddress.ip_network("172.16.0.0/12")
                or host in ipaddress.ip_network("192.168.0.0/16")
                or host in ipaddress.ip_network("100.64.0.0/10")
            )
        )
        if not private_ipv4:
            raise ValueError("Multihost master must use the confirmed private IPv4 network")
        if (
            isinstance(started, bool)
            or not isinstance(started, (int, float))
            or not math.isfinite(started)
            or started <= 0
            or started > time.time() + 5
        ):
            raise ValueError(
                "Multihost training requires one shared launch timestamp, captured before launch"
            )
        executed = getattr(args, "execute_started", None)
        if executed is not None and executed != started:
            raise ValueError("The child cannot reset the shared allocation deadline")
    elif started is not None or address is not None:
        raise ValueError("Shared-start and master-address options require multihost training")
    return {
        "nnodes": nodes,
        "nproc_per_node": processes,
        "node_rank": rank,
        "master_addr": address,
        "master_port": port,
    }


def torchrun_command(plan, plan_path):
    topology = plan.get("distributed", {"nnodes": 1, "nproc_per_node": 4, "node_rank": 0})
    command = [sys.executable, "-m", "torch.distributed.run"]
    if topology["nnodes"] == 1:
        command += [
            "--standalone",
            "--local-addr=127.0.0.1",
            f"--nproc_per_node={topology['nproc_per_node']}",
        ]
    else:
        command += [
            f"--nnodes={topology['nnodes']}",
            f"--nproc_per_node={topology['nproc_per_node']}",
            f"--node_rank={topology['node_rank']}",
            f"--master_addr={topology['master_addr']}",
            f"--master_port={topology['master_port']}",
            "--max_restarts=0",
            "--rdzv_conf=timeout=720",
        ]
    return command + [str(Path(__file__).resolve()), "--worker-plan", str(plan_path.resolve())]


def finish_follower(args, repository, *, topology, resume_step, hard_deadline):
    """Verify this node's durable result without publishing or sealing anything."""

    def check():
        if time.time() >= hard_deadline - 30:
            raise TimeoutError("Follower finalization exceeded the shared deadline")

    repository.store.check = check
    result = json.loads((args.output / "node-finished.json").read_text())
    if (
        result.get("node_finished") is not True
        or result.get("checkpoint_writer") is not False
        or result.get("node_rank") != topology["node_rank"]
        or result.get("world_size") != 4
        or result.get("initial_optimizer_step") != resume_step
        or result.get("final_optimizer_step", 0) <= resume_step
        or result.get("checkpoint", {}).get("checkpoint", {}).get("step")
        != result["final_optimizer_step"]
        or result.get("checkpoint") != repository.latest()
    ):
        raise RuntimeError("Follower lacks a matching final durable checkpoint receipt")
    atomic_json(args.output / "node-success.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(Path(os.environ["GMN_RESULT_PATH"]), result)
    print(
        json.dumps(
            {
                "event": "training_node_finished",
                "node_rank": topology["node_rank"],
                "final_optimizer_step": result["final_optimizer_step"],
                "checkpoint_writer": False,
            }
        ),
        flush=True,
    )
    return result


def validate_launch_options(args):
    topology = launch_topology(args)
    resolved_global_batch(args, topology)
    nat = getattr(args, "nat", False)
    clean = getattr(args, "clean_binaural", False)
    if nat and clean:
        raise ValueError("Choose exactly one of --nat and --clean-binaural")
    source_fields = [
        getattr(args, name, None)
        for name in (
            "curriculum_source_prefix",
            "curriculum_source_run_id",
            "curriculum_source_checkpoint_sha256",
        )
    ]
    if any(value is not None for value in source_fields) and (
        not nat or not all(source_fields) or not SHA.fullmatch(source_fields[2])
    ):
        raise ValueError("Curriculum import requires Nat and a complete pinned clean source")
    batch_source = [
        getattr(args, name, None)
        for name in ("batch_source_prefix", "batch_source_run_id", "batch_source_checkpoint_sha256")
    ]
    if any(value is not None for value in batch_source):
        if (
            not all(batch_source)
            or not SHA.fullmatch(batch_source[2])
            or getattr(args, "resume", False)
            or any(source_fields)
            or getattr(args, "global_batch_size", None) is None
            or batch_source[0] == getattr(args, "checkpoint_prefix", None)
            or topology["nnodes"] != 1
        ):
            raise ValueError(
                "Batch transition requires an explicit new batch, distinct prefix and pinned source; no resume/phase import"
            )
    readiness = [
        getattr(args, name, None) for name in ("stop_on_nat_ready_key", "stop_on_nat_ready_bucket")
    ]
    if any(readiness) and (not clean or not all(readiness)):
        raise ValueError("Nat readiness early-stop requires clean binaural and both bucket/key")
    manifest = getattr(args, "naturalistic_manifest", None)
    digest = getattr(args, "naturalistic_manifest_sha256", None)
    if (
        nat
        and not getattr(args, "plan_only", False)
        and (manifest is None or not isinstance(digest, str) or not SHA.fullmatch(digest))
    ):
        raise ValueError("Nat requires --naturalistic-manifest and its pinned SHA256")
    if not nat and (manifest is not None or digest is not None):
        raise ValueError("Naturalistic manifest arguments require --nat")
    waived = getattr(args, "skip_streaming_acceptance", False)
    reason = getattr(args, "streaming_acceptance_waiver_reason", None)
    if waived and (not isinstance(reason, str) or len(reason.strip()) < 12):
        raise ValueError("Skipping streaming acceptance requires a human-readable waiver reason")
    if not waived and reason is not None:
        raise ValueError("A waiver reason requires --skip-streaming-acceptance")
    if waived and getattr(args, "streaming_smoke_report", None) is not None:
        raise ValueError("Choose a verified streaming report or an explicit waiver, not both")


def streaming_acceptance(args):
    validate_launch_options(args)
    if getattr(args, "skip_streaming_acceptance", False):
        return {
            "status": "waived_by_user",
            "hardware_acceptance_passed": False,
            "report_sha256": None,
            "reason": args.streaming_acceptance_waiver_reason.strip(),
            "scope": "separate streaming GPU preflight only; runtime validation and checkpoint integrity remain enabled",
        }
    topology = launch_topology(args)
    default_batch = 16 if getattr(args, "nat", False) else 64
    if resolved_global_batch(args, topology) != default_batch:
        raise ValueError("Changed global batch requires an explicit separate-preflight waiver")
    if (topology["nnodes"], topology["nproc_per_node"]) != (1, 4):
        raise ValueError("Changed GPU topology requires an explicit separate-preflight waiver")
    if getattr(args, "clean_binaural", False):
        raise ValueError("Clean binaural curriculum requires an explicit preflight waiver")
    if not args.streaming_smoke_report:
        raise ValueError("A successful streaming hardware report is required before launch")
    digest = verify_streaming_report(
        args.streaming_smoke_report,
        expected_index_sha256=args.index_sha256,
        temporal=args.temporal,
        nat=args.nat,
        naturalistic_manifest_sha256=args.naturalistic_manifest_sha256,
    )
    return {"status": "passed", "hardware_acceptance_passed": True, "report_sha256": digest}


def freeze_naturalistic_manifest(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    bucket: str,
    endpoint: str,
):
    from src.data.naturalistic_audio import NaturalisticAudio

    if not isinstance(expected_sha256, str) or not SHA.fullmatch(expected_sha256):
        raise ValueError("A pinned naturalistic manifest SHA256 is required")
    with source.open("rb") as handle:
        payload = handle.read(256 * 1024**2 + 1)
    if len(payload) > 256 * 1024**2 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Naturalistic manifest size/SHA256 mismatch")
    destination.write_bytes(payload)
    renderer = NaturalisticAudio(destination, require_asset_checksums=True, cache_max_gib=1.0)
    if (
        not renderer.remote_assets
        or renderer.split != "train"
        or renderer.bucket != bucket
        or renderer.endpoint != endpoint
    ):
        raise ValueError(
            "Nat requires training assets in the same private bucket/endpoint as AudioSet"
        )
    return expected_sha256


def freeze_complete_index(source: Path, destination: Path, expected_sha256: str, endpoint: str):
    from src.data.sharded_audio_datamodule import _validate_index

    if not SHA.fullmatch(expected_sha256):
        raise ValueError("A pinned completed index SHA256 is required")
    with source.open("rb") as handle:
        payload = handle.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen staging index size/SHA256 mismatch")
    value = json.loads(payload)
    _validate_index(value, endpoint, allow_partial=False)
    destination.write_bytes(payload)
    return expected_sha256


def verify_resume_state(
    path: Path,
    receipt: dict,
    index_sha256: str,
    *,
    nat: bool = False,
    naturalistic_manifest_sha256: str | None = None,
    clean_binaural: bool = False,
    global_batch: int | None = None,
):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    step = receipt["checkpoint"]["step"]
    if (
        state.get("global_step") != step
        or int(state.get("state_dict", {}).get("ema_updates", -1)) != step
        or len(state.get("optimizer_states", [])) != 1
        or not state["optimizer_states"][0].get("state")
        or len(state.get("lr_schedulers", [])) != 1
        or state["lr_schedulers"][0].get("last_epoch") != step
        or any(
            value.get("step") != step for value in state["optimizer_states"][0]["state"].values()
        )
        or not state.get("loops")
        or state.get("ShardedAudioDataModule", {}).get("index_sha256") != index_sha256
    ):
        raise ValueError(
            "Resume checkpoint is not complete matching model/EMA/optimizer/loop/data state"
        )
    from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule

    if nat and clean_binaural:
        raise ValueError("Cannot verify a checkpoint as both clean and naturalistic")
    binaural = nat or clean_binaural
    expected_batch = (16 if binaural else 64) if global_batch is None else global_batch
    if (
        type(expected_batch) is not int
        or expected_batch < 1
        or (
            global_batch is not None
            and receipt.get("owner", {}).get("global_batch") != expected_batch
        )
    ):
        raise ValueError("Invalid checkpoint global batch")
    metadata = HierarchicalWavJEPAModule.validated_checkpoint_metadata(state)
    dynamics = metadata["dynamics"]
    data_state = state.get("ShardedAudioDataModule", {})
    if (
        metadata["backbone"].get("input_channels", 1) != (2 if binaural else 1)
        or dynamics.get("masked_crops_per_recording", 0) != (8 if binaural else 0)
        or dynamics.get("masked_crop_samples") != (32160 if binaural else None)
        or data_state.get("naturalistic_sha256") != naturalistic_manifest_sha256
        or (
            clean_binaural
            and (
                data_state.get("data_mode") != "clean_binaural"
                or data_state.get("masked_crops_per_recording") != 8
                or data_state.get("masked_crop_samples") != 32160
                or receipt.get("owner", {}).get("training_phase") != "clean_binaural"
                or receipt.get("owner", {}).get("global_batch") != expected_batch
            )
        )
        or (
            nat
            and (
                not isinstance(naturalistic_manifest_sha256, str)
                or not SHA.fullmatch(naturalistic_manifest_sha256)
                or data_state.get("masked_crops_per_recording") != 8
                or data_state.get("masked_crop_samples") != 32160
                or receipt.get("owner", {}).get("naturalistic_manifest_sha256")
                != naturalistic_manifest_sha256
                or receipt.get("owner", {}).get("global_batch") != expected_batch
            )
        )
    ):
        raise ValueError("Resume checkpoint has a different channel/crop/manifest recipe")
    if (
        dynamics["lr_total_steps"],
        dynamics["lr_warmup_steps"],
        dynamics["ema_anneal_end_step"],
    ) != (375000, 100000, 100000) or state.get(
        HierarchicalWavJEPAModule.SCHEDULE_CHECKPOINT_KEY
    ) != {
        "total_steps": 375000,
        "warmup_steps": 100000,
        "ema_end_step": 100000,
    }:
        raise ValueError("Resume checkpoint has a different training schedule")
    return step


def import_clean_curriculum(
    source: CheckpointRepository,
    destination: Path,
    *,
    expected_checkpoint_sha256: str,
    index_sha256: str,
    target_manifest_sha256: str,
):
    """Restore untouched full Lightning state; permit only the verified data-phase change."""
    import torch

    from src.data.sharded_audio_datamodule import data_state_sha256

    handoff = source.clean_handoff(expected_checkpoint_sha256)
    receipt = source.download_latest(destination)
    if receipt != handoff["checkpoint"] or file_sha256(destination) != expected_checkpoint_sha256:
        raise ValueError("Clean source changed while importing its handoff")
    step = verify_resume_state(
        destination,
        receipt,
        index_sha256,
        clean_binaural=True,
        global_batch=source.owner["global_batch"],
    )
    original = torch.load(destination, map_location="cpu", weights_only=True)
    transition = {
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "source_data_state_sha256": data_state_sha256(original["ShardedAudioDataModule"]),
        "index_sha256": index_sha256,
        "target_manifest_sha256": target_manifest_sha256,
    }
    return step, receipt, transition


def import_batch_transition(
    source: CheckpointRepository,
    destination: Path,
    *,
    expected_checkpoint_sha256: str,
    index_sha256: str,
    new_global_batch: int,
    nat: bool = False,
    clean_binaural: bool = False,
    naturalistic_manifest_sha256: str | None = None,
):
    """Import unchanged complete state; only a new owned run may change its batch."""
    old_batch = source.owner["global_batch"]
    if type(new_global_batch) is not int or new_global_batch < 1 or new_global_batch == old_batch:
        raise ValueError("Batch transition must explicitly change the positive global batch")
    if source.owner.get("naturalistic_manifest_sha256") != naturalistic_manifest_sha256:
        raise ValueError("Batch transition cannot change naturalistic assets")
    expected_phase = "naturalistic" if nat else "clean_binaural" if clean_binaural else None
    if source.owner.get("training_phase") != expected_phase:
        raise ValueError("Batch transition cannot change training phase")
    receipt = source.download_latest(destination)
    if (
        receipt["checkpoint"]["sha256"] != expected_checkpoint_sha256
        or file_sha256(destination) != expected_checkpoint_sha256
    ):
        raise ValueError("Batch source changed or differs from its pinned checkpoint SHA256")
    step = verify_resume_state(
        destination,
        receipt,
        index_sha256,
        nat=nat,
        clean_binaural=clean_binaural,
        naturalistic_manifest_sha256=naturalistic_manifest_sha256,
        global_batch=old_batch,
    )
    if step >= 375000:
        raise ValueError("Batch source has already completed its optimizer schedule")
    transition = {
        "source_run_id": source.run_id,
        "source_prefix": source.prefix,
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "source_optimizer_step": step,
        "old_global_batch": old_batch,
        "new_global_batch": new_global_batch,
    }
    return step, receipt, transition


def resolved_global_batch(args, topology=None):
    topology = launch_topology(args) if topology is None else topology
    default = 16 if getattr(args, "nat", False) or getattr(args, "clean_binaural", False) else 64
    value = getattr(args, "global_batch_size", None)
    value = default if value is None else value
    world = topology["nnodes"] * topology["nproc_per_node"]
    if type(value) is not int or value < 2 * world or value % world:
        raise ValueError(
            "Global recording batch must be divisible by the GPU rank count with at least two recordings per rank"
        )
    return value


def make_config(plan: dict):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    nat = plan.get("nat", False)
    clean = plan.get("clean_binaural", False)
    binaural = nat or clean
    global_batch = plan.get("global_batch", 16 if binaural else 64)
    if type(global_batch) is not int or global_batch < 1:
        raise ValueError("Global recording batch must be a positive integer")
    experiment = (
        "audioset_clean_binaural_h100_four"
        if clean
        else "audioset_nat_h100_four"
        if binaural
        else (
            "audioset_temporal_h100_four" if plan.get("temporal", False) else "audioset_h100_four"
        )
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", "data=sharded_audio"],
        )
    OmegaConf.set_struct(cfg, False)
    work, output = Path(plan["work_dir"]), Path(plan["output"])
    cfg.paths.root_dir = cfg.paths.work_dir = str(ROOT)
    cfg.paths.output_dir = str(work / "run")
    cfg.paths.log_dir = str(work / "logs")
    cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
    cfg.logger = {
        "csv": {
            "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
            "save_dir": str(output / "metrics"),
            "name": "training",
            "version": "",
            "flush_logs_every_n_steps": 50,
        }
    }
    cfg.callbacks = {
        "timer": {
            "_target_": "scripts.segment_checkpoint.SegmentTimer",
            "training_deadline": plan["training_deadline"],
        },
        "model_checkpoint": {
            "_target_": "scripts.segment_checkpoint.DurableStepCheckpoint",
            "repository_options": plan["repository_options"],
            "segment_id": plan["segment_id"],
            "data_seed": plan["resolved_data_seed"],
            "receipt_path": str(output / "latest-checkpoint.json"),
            "transfer_timeout_seconds": plan["budget"]["transfer_timeout_seconds"],
            "hard_deadline": plan["hard_deadline"],
            # Keep one full-state recovery file in the platform's captured output
            # even when private-S3 publication fails; the large input cache stays
            # in work_dir and never enters the output artifact.
            "dirpath": str(output / "checkpoints"),
            "filename": "step_{step:07d}",
            "auto_insert_metric_name": False,
            "every_n_train_steps": plan["checkpoint_every_steps"],
            # Recheck durable publication early in every segment, including a
            # resume whose original step-250 checkpoint is already behind it.
            "first_checkpoint_step": min(plan.get("resume_step", 0) + 250, 375000),
            "async_upload": True,
        },
        "progress": {
            "_target_": "scripts.training_progress.TrainingProgress",
            "every_n_steps": 100,
            "all_ranks": True,
        },
    }
    if plan.get("operator_stop"):
        # Only rank zero reads the request. Every rank joins the callback's
        # decision broadcast before the durable checkpoint handles should_stop.
        cfg.callbacks = {
            "timer": cfg.callbacks.timer,
            "operator_stop": {
                "_target_": "scripts.operator_stop.OperatorStop",
                **plan["operator_stop"],
            },
            **{name: value for name, value in cfg.callbacks.items() if name != "timer"},
        }
    if plan.get("nat_readiness"):
        readiness = plan["nat_readiness"]
        cfg.callbacks["nat_readiness"] = {
            "_target_": "scripts.curriculum_ready_stop.NatReadyStop",
            "endpoint": readiness["endpoint"],
            "bucket": readiness["bucket"],
            "ready_key": readiness["key"],
        }
    cfg.data.clean_binaural = clean
    cfg.data.clean_to_nat_transition = plan.get("clean_to_nat_transition")
    cfg.data.index_path = str(work / "frozen-index.json")
    cfg.data.endpoint = plan["data_endpoint"]
    cfg.data.cache_dir = str(plan.get("cache_dir") or work / "shard-cache")
    cfg.data.cache_max_gib = 60
    if nat:
        manifest = work / "frozen-naturalistic-manifest.json"
        if file_sha256(manifest) != plan["naturalistic_manifest_sha256"]:
            raise ValueError("Frozen naturalistic manifest changed before training")
        cfg.data.naturalistic_manifest = str(manifest)
    cfg.data.num_workers = 1
    # Shared object storage can deliver a healthy ~625 MB shard in over two
    # minutes. All ranged retries share this bound; peers have 720s for DDP.
    cfg.data.download_timeout_seconds = 300
    cfg.data.worker_timeout_seconds = 360
    cfg.data.allow_partial_index = False
    cfg.data.allow_stream_restart_on_resume = True
    cfg.data.seed = plan["resolved_data_seed"]
    cfg.trainer.min_epochs = cfg.trainer.min_steps = 0
    topology = plan.get("distributed", {"nnodes": 1, "nproc_per_node": 4})
    cfg.trainer.devices = topology["nproc_per_node"]
    cfg.trainer.num_nodes = topology["nnodes"]
    world_size = cfg.trainer.devices * cfg.trainer.num_nodes
    if world_size not in {4, 8} or global_batch < 2 * world_size or global_batch % world_size:
        raise ValueError(
            "Global recording batch requires four/eight ranks and at least two recordings per rank"
        )
    cfg.data.batch_size = global_batch // world_size
    shard_prefetch = plan.get("shard_prefetch")
    if shard_prefetch is None:
        shard_prefetch = int((cfg.trainer.num_nodes, cfg.trainer.devices) == (1, 8))
    if type(shard_prefetch) is not int or shard_prefetch not in {0, 1}:
        raise ValueError("Shard prefetch must be 0 or 1")
    cfg.data.shard_prefetch = shard_prefetch
    activation_checkpointing = plan.get("activation_checkpointing", "auto")
    if activation_checkpointing not in {"auto", "on", "off"}:
        raise ValueError("Activation checkpointing must be auto, on, or off")
    if activation_checkpointing != "auto":
        cfg.model.backbone.config.use_gradient_checkpointing = activation_checkpointing == "on"
    elif (cfg.trainer.num_nodes, cfg.trainer.devices) == (1, 8):
        # Spare memory on the eight-H100 node removes activation recomputation.
        # Batch, crops, optimizer, EMA and the complete loss remain unchanged.
        cfg.model.backbone.config.use_gradient_checkpointing = False
    cfg.trainer.max_epochs = -1
    cfg.trainer.max_steps = 375000
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.limit_val_batches = 1.0
    cfg.trainer.enable_model_summary = False
    # Retain the conservative checkpoint-aware timeout for final serialization
    # and synchronous compatibility; async upload polls use short collectives.
    cfg.trainer.strategy.timeout = {
        "_target_": "datetime.timedelta",
        "seconds": max(
            instantiate(cfg.trainer.strategy.timeout).total_seconds(),
            minimum_checkpoint_collective_timeout(plan["budget"]["transfer_timeout_seconds"]),
        ),
    }
    cfg.ckpt_path = plan["resume_path"]
    cfg.train, cfg.test = True, False
    if (
        cfg.data.batch_size * cfg.trainer.devices * cfg.trainer.num_nodes != global_batch
        or cfg.trainer.get("accumulate_grad_batches", 1) != 1
    ):
        raise ValueError(f"Segment must preserve its global optimizer batch of {global_batch}")
    if binaural and (
        cfg.data.masked_crops_per_recording != 8
        or cfg.data.masked_crop_samples != 32160
        or cfg.model.masked_crops_per_recording != 8
        or cfg.model.masked_crop_samples != 32160
        or cfg.model.backbone.config.input_channels != 2
    ):
        raise ValueError(
            "Nat segment must retain binaural input and eight 2.01-second masked crops"
        )
    if (cfg.model.lr_total_steps, cfg.model.lr_warmup_steps, cfg.model.ema_anneal_end_step) != (
        375000,
        100000,
        100000,
    ):
        raise ValueError("Segment must preserve the complete original LR/EMA schedules")
    return cfg


def worker(plan_path: Path):
    import torch.distributed as dist

    from src.train import train

    plan = json.loads(plan_path.read_text())
    topology = plan.get("distributed", {"nnodes": 1, "nproc_per_node": 4, "node_rank": 0})
    world_size = topology["nnodes"] * topology["nproc_per_node"]
    if int(os.environ.get("WORLD_SIZE", "0")) != world_size:
        raise ValueError("Training segment rank count differs from its pinned launch topology")
    if topology["nnodes"] > 1 and (
        int(os.environ.get("RANK", "-1")) != topology["node_rank"]
        or int(os.environ.get("LOCAL_RANK", "-1")) != 0
        or int(os.environ.get("LOCAL_WORLD_SIZE", "-1")) != 1
    ):
        raise ValueError("Worker rank does not match its pinned multihost plan")
    metrics, objects = train(make_config(plan))
    trainer, model = objects["trainer"], objects["model"]
    from scripts.operator_stop import OperatorStop

    stop_reason = (
        "operator"
        if any(
            isinstance(callback, OperatorStop) and callback.stop_requested
            for callback in trainer.callbacks
        )
        else "bounded_segment"
    )
    step = trainer.global_step
    if step <= plan["resume_step"] or int(model.ema_updates) != step:
        raise RuntimeError("Segment did not advance optimizer and EMA steps together")
    callback = trainer.checkpoint_callback
    if callback.last_receipt is None or callback.last_receipt["checkpoint"]["step"] != step:
        raise RuntimeError("Final optimizer step lacks a verified durable checkpoint")
    if topology["node_rank"] > 0:
        atomic_json(
            Path(plan["output"]) / "node-finished.json",
            {
                "node_finished": True,
                "checkpoint_writer": False,
                "node_rank": topology["node_rank"],
                "world_size": 4,
                "initial_optimizer_step": plan["resume_step"],
                "final_optimizer_step": step,
                "checkpoint": callback.last_receipt,
            },
        )
    dist.barrier()
    if trainer.is_global_zero:
        atomic_json(
            Path(plan["output"]) / "segment-finished.json",
            {
                "segment_finished": True,
                "stop_reason": stop_reason,
                "ready_for_nat": False,
                "training_complete": step == 375000,
                "initial_optimizer_step": plan["resume_step"],
                "final_optimizer_step": step,
                "ema_updates": int(model.ema_updates),
                "global_batch": plan["global_batch"],
                "global_masked_crops": plan["global_batch"] * 8
                if plan["nat"] or plan.get("clean_binaural")
                else 0,
                "batch_transition": plan.get("batch_transition"),
                "training_phase": "naturalistic"
                if plan["nat"]
                else ("clean_binaural" if plan.get("clean_binaural") else "mono"),
                "naturalistic_manifest_sha256": plan["naturalistic_manifest_sha256"],
                "streaming_acceptance": plan["streaming_acceptance"],
                "model_variant": plan["model_variant"],
                "final_metrics": {
                    key: float(value)
                    for key, value in metrics.items()
                    if not hasattr(value, "numel") or value.numel() == 1
                },
                "stream_positions_checkpointed": False,
                "stream_restarted_on_resume": plan["resume_path"] is not None,
                "data_seed_policy": "base_plus_resume_step",
                "resolved_data_seed": plan["resolved_data_seed"],
                "checkpoint": callback.last_receipt,
                "budget": plan["budget"],
            },
        )
    dist.barrier()


def remember_descendants(owned):
    """Retain PID/create-time identities even if a worker later becomes orphaned."""
    import psutil

    for process in list(owned.values()):
        try:
            for child in process.children(recursive=True):
                owned[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            continue


def _owned_process_alive(process):
    """A reused PID or zombie is not a surviving owned compute process."""
    import psutil

    try:
        # psutil's is_running compares the retained PID/create-time identity.
        # status alone would accidentally wait on a different, reused PID.
        return process.is_running() and process.status() not in {
            psutil.STATUS_ZOMBIE,
            psutil.STATUS_DEAD,
        }
    except psutil.NoSuchProcess:
        return False


def wait_owned_processes(processes, *, timeout):
    """Bounded wait with a polling fallback for kernels without working pidfds.

    Some container kernels expose pidfd_open but reject it with EINVAL. psutil
    then raises instead of waiting; use its identity-checked process inspection
    without Process.wait in that case. The fallback inherits the original
    deadline, and errors other than unsupported waiting remain fail-closed.
    Root-child reaping is still performed by subprocess.Popen.wait below.
    """
    import psutil

    deadline = time.monotonic() + timeout
    try:
        return psutil.wait_procs(processes, timeout=timeout)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        print(
            json.dumps({"event": "process_wait_fallback", "errno": error.errno}),
            flush=True,
        )
    alive, gone = list(processes), []
    while True:
        remaining_processes = []
        for process in alive:
            if _owned_process_alive(process):
                remaining_processes.append(process)
            else:
                gone.append(process)
        alive = remaining_processes
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            return gone, alive
        time.sleep(min(0.05, remaining))


def stop_process_tree(process, owned, *, grace_seconds=10):
    """Stop detached torchrun rank sessions as well as the launcher process.

    Freeze discovered parents before the final tree scan so they cannot create
    more workers during cleanup. psutil checks create times before signaling,
    protecting against PID reuse. Escalation does not depend on wrapper survival.
    """
    import psutil

    suspended = set()
    while True:
        for identity, child in list(owned.items()):
            if identity in suspended:
                continue
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
            suspended.add(identity)
        remember_descendants(owned)
        if set(owned) <= suspended:
            break
    for child in owned.values():
        try:
            child.terminate()
            child.resume()
        except psutil.NoSuchProcess:
            pass
    _, alive = wait_owned_processes(list(owned.values()), timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = wait_owned_processes(alive, timeout=grace_seconds)
    # A reparented zombie has released compute resources and awaits init's reap.
    surviving = [child for child in alive if _owned_process_alive(child)]
    if surviving:
        raise RuntimeError("Owned training processes survived the hard watchdog")
    process.wait(timeout=1)


def segment_budget(args):
    return SegmentBudget(
        args.segment_hours,
        args.budget_usd,
        args.maximum_rate_usd_per_hour,
        args.finalization_reserve_seconds,
        args.transfer_timeout_seconds,
    )


def supervise(args, raw_args):
    """Bound setup and training, including torchrun's detached rank process trees."""
    import psutil

    budget = segment_budget(args)
    if args.plan_only:
        print(json.dumps(budget.report(), indent=2))
        return
    started = getattr(args, "shared_started_at", None) or time.time()
    hard_stop = time.monotonic() + started + budget.seconds - time.time() - 30
    if hard_stop <= time.monotonic():
        raise TimeoutError("Shared segment allowance expired before local launch")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *raw_args,
            "--execute-started",
            str(started),
        ],
        cwd=ROOT,
        start_new_session=True,
    )
    root = psutil.Process(process.pid)
    owned = {(root.pid, root.create_time()): root}
    try:
        while True:
            remember_descendants(owned)
            code = process.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f"Training segment exited unsuccessfully (code {code})")
                break
            # The remaining 30s let TERM/KILL finish within our own bound. The
            # platform's five-minute margin is an additional hard backstop.
            remaining = hard_stop - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, budget.seconds - 30)
            try:
                process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop_process_tree(process, owned)


def finish_clean_phase(repository, result, output: Path, hard_deadline: float):
    """An operator pause saves full state but does not seal the clean run."""
    if result.get("stop_reason") == "operator":
        result["ready_for_nat"] = False
        return

    def handoff_check():
        if time.time() >= hard_deadline - 30:
            raise TimeoutError("Clean handoff exceeded segment deadline")

    repository.store.check = handoff_check
    handoff = repository.publish_clean_handoff(result["checkpoint"], writer_finished=True)
    atomic_json(output / "clean-handoff.json", handoff)
    result["ready_for_nat"] = True
    result["clean_handoff_key"] = repository.handoff_key


def execute_segment(args):
    budget = segment_budget(args)
    topology = launch_topology(args)
    if not args.exclusive_checkpoint_writer or not args.acknowledge_stream_restart:
        raise ValueError(
            "Explicit exclusive-writer and stream-restart acknowledgements are required"
        )
    acceptance = streaming_acceptance(args)
    nat, clean = args.nat, getattr(args, "clean_binaural", False)
    temporal = args.temporal or nat or clean
    global_batch = resolved_global_batch(args, topology)
    started = args.execute_started
    hard_deadline = started + budget.seconds
    training_deadline = hard_deadline - budget.finalization_reserve_seconds

    def check():
        if time.time() >= training_deadline:
            raise TimeoutError("Segment setup exhausted its training allowance")

    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "streaming-acceptance.json", acceptance)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    frozen_sha = freeze_complete_index(
        args.index, args.work_dir / "frozen-index.json", args.index_sha256, args.data_endpoint
    )
    naturalistic_sha = None
    if nat:
        index = json.loads((args.work_dir / "frozen-index.json").read_text())
        naturalistic_sha = freeze_naturalistic_manifest(
            args.naturalistic_manifest,
            args.work_dir / "frozen-naturalistic-manifest.json",
            args.naturalistic_manifest_sha256,
            bucket=index["bucket"],
            endpoint=args.data_endpoint,
        )
    source_prefix = getattr(args, "curriculum_source_prefix", None)
    curriculum_source = (
        None
        if source_prefix is None
        else {
            "prefix": source_prefix,
            "run_id": args.curriculum_source_run_id,
            "checkpoint_sha256": args.curriculum_source_checkpoint_sha256,
        }
    )
    client = checkpoint_client(args.checkpoint_endpoint)
    resume_path, resume_step, transition, batch_transition = None, 0, None, None
    batch_source_prefix = getattr(args, "batch_source_prefix", None)
    if batch_source_prefix is not None:
        source = CheckpointRepository.from_owned_source(
            client,
            bucket=args.checkpoint_bucket,
            prefix=batch_source_prefix,
            run_id=args.batch_source_run_id,
            index_sha256=frozen_sha,
            check=check,
        )
        if source.base_data_seed != args.base_data_seed:
            raise ValueError("Batch transition must preserve the base data seed")
        resume_path = args.work_dir / "resume.ckpt"
        resume_step, receipt, batch_transition = import_batch_transition(
            source,
            resume_path,
            expected_checkpoint_sha256=args.batch_source_checkpoint_sha256,
            index_sha256=frozen_sha,
            new_global_batch=global_batch,
            nat=nat,
            clean_binaural=clean,
            naturalistic_manifest_sha256=naturalistic_sha,
        )
        atomic_json(args.output / "imported-batch-checkpoint.json", receipt)
    elif args.resume:
        # Preserve immutable provenance on subsequent segments; never infer a
        # batch change from a mutable CLI setting or rewrite an existing owner.
        existing = CheckpointRepository.from_owned_source(
            client,
            bucket=args.checkpoint_bucket,
            prefix=args.checkpoint_prefix,
            run_id=args.run_id,
            index_sha256=frozen_sha,
            check=check,
        )
        batch_transition = existing.owner.get("batch_transition")
        if curriculum_source is None:
            curriculum_source = existing.owner.get("curriculum_source")
    repository_options = dict(
        bucket=args.checkpoint_bucket,
        prefix=args.checkpoint_prefix,
        run_id=args.run_id,
        index_sha256=frozen_sha,
        base_data_seed=args.base_data_seed,
        global_batch=global_batch,
        naturalistic_manifest_sha256=naturalistic_sha,
        clean_binaural=clean,
        curriculum_source=curriculum_source,
        batch_transition=batch_transition,
    )
    repository = CheckpointRepository(client, **repository_options, check=check)
    repository.open(initialize=not args.resume)
    if args.resume:
        resume_path = args.work_dir / "resume.ckpt"
        receipt = repository.download_latest(resume_path)
        pinned_resume = getattr(args, "resume_sha256", None)
        if pinned_resume is not None and receipt["checkpoint"]["sha256"] != pinned_resume:
            raise ValueError("Resume checkpoint differs from the shared pinned SHA256")
        resume_step = verify_resume_state(
            resume_path,
            receipt,
            frozen_sha,
            nat=nat,
            naturalistic_manifest_sha256=naturalistic_sha,
            clean_binaural=clean,
            global_batch=global_batch,
        )
        atomic_json(args.output / "resumed-checkpoint.json", receipt)
        if resume_step == 375000:
            raise ValueError("Training already completed 375000 steps; no new segment is needed")
    elif curriculum_source is not None:
        source = CheckpointRepository.from_owned_source(
            client,
            bucket=args.checkpoint_bucket,
            prefix=source_prefix,
            run_id=args.curriculum_source_run_id,
            index_sha256=frozen_sha,
            check=check,
        )
        if (
            source.owner["global_batch"] != global_batch
            or source.base_data_seed != args.base_data_seed
        ):
            raise ValueError("Clean-to-Nat import must preserve global batch and base data seed")
        resume_path = args.work_dir / "resume.ckpt"
        resume_step, receipt, transition = import_clean_curriculum(
            source,
            resume_path,
            expected_checkpoint_sha256=args.curriculum_source_checkpoint_sha256,
            index_sha256=frozen_sha,
            target_manifest_sha256=naturalistic_sha,
        )
        atomic_json(args.output / "imported-clean-checkpoint.json", receipt)
    check()
    plan = {
        "segment_id": args.segment_id,
        "distributed": topology,
        "activation_checkpointing": getattr(args, "activation_checkpointing", "auto"),
        "shard_prefetch": getattr(args, "shard_prefetch", None),
        "operator_stop": {
            "request_path": str((args.output / "operator-stop.json").resolve()),
            "run_id": args.run_id,
            "segment_id": args.segment_id,
            "nonce": uuid.uuid4().hex,
        },
        "pinned_resume_sha256": getattr(args, "resume_sha256", None),
        "nat": nat,
        "clean_binaural": clean,
        "clean_to_nat_transition": transition,
        "batch_transition": batch_transition,
        "curriculum_source": curriculum_source,
        "nat_readiness": {
            "endpoint": args.data_endpoint,
            "bucket": args.stop_on_nat_ready_bucket,
            "key": args.stop_on_nat_ready_key,
        }
        if getattr(args, "stop_on_nat_ready_key", None)
        else None,
        "temporal": temporal,
        "global_batch": global_batch,
        "naturalistic_manifest_sha256": naturalistic_sha,
        "model_variant": "clean_binaural_temporal_aggregation"
        if clean
        else "nat_temporal_aggregation"
        if nat
        else ("temporal_aggregation" if temporal else "pointwise"),
        "work_dir": str(args.work_dir.resolve()),
        "cache_dir": str(
            (getattr(args, "cache_dir", None) or args.work_dir / "shard-cache").resolve()
        ),
        "output": str(args.output.resolve()),
        "repository_options": repository_options | {"endpoint": args.checkpoint_endpoint},
        "data_endpoint": args.data_endpoint,
        "index_sha256": frozen_sha,
        "streaming_smoke_report_sha256": acceptance["report_sha256"],
        "streaming_acceptance": acceptance,
        "training_deadline": training_deadline,
        "hard_deadline": hard_deadline,
        "budget": budget.report(),
        "resume_path": str(resume_path.resolve()) if resume_path else None,
        "resume_step": resume_step,
        "data_seed_policy": "base_plus_resume_step",
        "base_data_seed": args.base_data_seed,
        "resolved_data_seed": args.base_data_seed + resume_step,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "stream_positions_checkpointed": False,
    }
    plan_path = args.output / "segment-plan.json"
    atomic_json(plan_path, plan)
    command = torchrun_command(plan, plan_path)
    # torchrun creates a new session for each rank. The parent's watchdog tracks
    # descendant PID/create-time identities rather than assuming shared PGIDs.
    code = subprocess.Popen(command, cwd=ROOT, start_new_session=False).wait()
    if code:
        raise RuntimeError(f"Training segment exited unsuccessfully (code {code})")
    if topology["node_rank"] > 0:
        # Only node zero owns final publication. Other nodes have independent
        # filesystems and cannot read its local segment-finished.json.
        finish_follower(
            args,
            repository,
            topology=topology,
            resume_step=resume_step,
            hard_deadline=hard_deadline,
        )
        return
    result_path = args.output / "segment-finished.json"
    result = json.loads(result_path.read_text())
    if (
        result.get("segment_finished") is not True
        or result.get("final_optimizer_step", 0) <= resume_step
    ):
        raise RuntimeError("Worker exited without an advanced, verified final checkpoint")
    if clean:
        # Only the local global-rank-zero writer publishes; the orchestrator
        # must also confirm any remote followers exited before a phase switch.
        finish_clean_phase(repository, result, args.output, hard_deadline)
    result["wall_seconds"] = time.time() - started
    result["observed_runner_time_cost_at_supplied_rate_usd"] = (
        result["wall_seconds"] / 3600 * budget.maximum_rate_usd_per_hour
    )
    result["compute_cost_upper_bound_usd"] = budget.maximum_platform_cost_usd
    atomic_json(args.output / "segment-success.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(Path(os.environ["GMN_RESULT_PATH"]), result)
    print(
        json.dumps(
            {
                "segment_finished": True,
                "final_optimizer_step": result["final_optimizer_step"],
                "training_complete": result["training_complete"],
            }
        ),
        flush=True,
    )


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--execute-started", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--shared-started-at", type=float)
    parser.add_argument("--resume-sha256")
    parser.add_argument(
        "--activation-checkpointing",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto disables activation recomputation on eight local GPUs; other presets stay unchanged",
    )
    parser.add_argument(
        "--shard-prefetch",
        type=int,
        choices=(0, 1),
        help="Look ahead one shard without adding workers; defaults to 1 on eight local GPUs",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use the tested temporal aggregation model; omit for the pointwise baseline",
    )
    parser.add_argument("--nat", action="store_true", help="Train the binaural Nat temporal model")
    parser.add_argument(
        "--clean-binaural",
        action="store_true",
        help="Duplicate clean mono into the unchanged two-channel Nat architecture",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Explicit global recording batch; changes require a new owned prefix",
    )
    parser.add_argument("--batch-source-prefix")
    parser.add_argument("--batch-source-run-id")
    parser.add_argument("--batch-source-checkpoint-sha256")
    parser.add_argument("--curriculum-source-prefix")
    parser.add_argument("--curriculum-source-run-id")
    parser.add_argument("--curriculum-source-checkpoint-sha256")
    parser.add_argument("--stop-on-nat-ready-key")
    parser.add_argument("--stop-on-nat-ready-bucket")
    parser.add_argument("--naturalistic-manifest", type=Path)
    parser.add_argument("--naturalistic-manifest-sha256")
    parser.add_argument("--skip-streaming-acceptance", action="store_true")
    parser.add_argument("--streaming-acceptance-waiver-reason")
    parser.add_argument("--segment-hours", type=float, default=10)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--maximum-rate-usd-per-hour", type=float)
    parser.add_argument("--finalization-reserve-seconds", type=float, default=1800)
    parser.add_argument("--transfer-timeout-seconds", type=float, default=600)
    parser.add_argument("--run-id")
    parser.add_argument("--segment-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, help="Persistent node-local audio cache outside output"
    )
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-sha256")
    parser.add_argument("--data-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--checkpoint-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--checkpoint-bucket")
    parser.add_argument("--checkpoint-prefix")
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--base-data-seed", type=int, default=42)
    parser.add_argument("--streaming-smoke-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclusive-checkpoint-writer", action="store_true")
    parser.add_argument("--acknowledge-stream-restart", action="store_true")
    args = parser.parse_args(raw_args)
    if args.worker_plan:
        return worker(args.worker_plan)
    required = ["budget_usd", "maximum_rate_usd_per_hour"]
    if not args.plan_only:
        required += [
            "run_id",
            "segment_id",
            "output",
            "work_dir",
            "index",
            "index_sha256",
            "checkpoint_bucket",
            "checkpoint_prefix",
        ]
    if any(getattr(args, field) is None for field in required):
        parser.error(
            "Missing required launch fields: "
            + ", ".join(field for field in required if getattr(args, field) is None)
        )
    try:
        validate_launch_options(args)
    except ValueError as error:
        parser.error(str(error))
    if args.checkpoint_every_steps < 1:
        parser.error("checkpoint-every-steps must be positive")
    if args.execute_started is not None:
        execute_segment(args)
    else:
        supervise(args, raw_args)


if __name__ == "__main__":
    main()
