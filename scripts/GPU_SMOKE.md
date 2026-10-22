# Bounded GPU smoke

This probe runs the two-GPU BF16 correctness/resume test, then three full-base
training updates using `audioset_h100` model dimensions and batch 32 per GPU.
The latter uses **generated audio** with the exact 145,440-sample input shape;
it is not AudioSet training or a quality evaluation. No pretrained downloads
or full-base checkpoints are created. Reports include per-rank step times and
peak CUDA memory. Three updates give only a preliminary timing observation.

For Autoresearch, use one batch job with these **platform-enforced** bounds:

```json
{
  "chip": "h100",
  "chip_count": 2,
  "max_duration_minutes": 10,
  "build_timeout_minutes": 10,
  "max_restarts": 0,
  "resume": "none",
  "queue_ttl_minutes": 60,
  "dockerfile_path": "Dockerfile.gpu-smoke",
  "command": "python scripts/smoke_gpu.py --output /output/h-wavjepa-smoke",
  "checks": {
    "success": {
      "check": {
        "type": "file_exists",
        "path": "/output/h-wavjepa-smoke/smoke-success.json"
      }
    }
  }
}
```

Add the approved mission, a fresh idempotency key, and an allowlisted source
context. Include only `.project-root`, `pyproject.toml`, `uv.lock`,
`requirements.txt`, `Dockerfile.gpu-smoke`, `src`, `configs`, `tests`, `scripts`, `third_party`,
and `THIRD_PARTY_NOTICES.md`; omit `__pycache__` and `*.pyc`.
Never upload `.git`, `.venv`, `.env`, datasets, checkpoints, or existing logs.
For `create_context`, compress the tar with **zstd** before hashing/uploading:
the service rejects plain tar, gzip, and xz. A tracked-files-only context can be
prepared without copying local credentials or caches:

```bash
git archive --format=tar HEAD .project-root pyproject.toml uv.lock requirements.txt Dockerfile.gpu-smoke \
  src configs tests scripts third_party THIRD_PARTY_NOTICES.md \
  | zstd -o /tmp/h-wavjepa-smoke.tar.zst
```

The Dockerfile lives at the context root because this platform resolves build
inputs relative to its directory. It pins the official runtime image digest and derives Python
constraints from `uv.lock`, retaining the image's matched CUDA/Torch stack.

On 2026-10-22, `validate_job` quotes **$1.26 maximum GPU runtime** for two H100s
over ten minutes. The published build rate is $0.01/minute, so its separate
ten-minute cap adds at most **$0.10** (before the free build allowance).
Revalidate prices immediately before submitting. There is no submitted job or
GPU allocation merely from adding these files. Do not automatically resubmit a
failed probe; review its receipt and the remaining authorized budget first.

The build timeout includes source download, image construction, and registry
publishing; a prebuilt image uses that same timeout for import. This is separate
from the GPU runtime limit. `resume=none`, no task/checkpoint declarations, and
`max_restarts=0` avoid billable repeated work. Platform-severed attempts without
checkpoint/task progress are documented as free. Queue time is free.

Sources: [platform manual](https://autoresearch.sfcompute.com/llms.txt),
[pricing](https://autoresearch.sfcompute.com/), and
[API schema](https://autoresearch.sfcompute.com/openapi.json).
