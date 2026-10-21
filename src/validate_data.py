"""Read-only validation of real WAV manifests before scheduling training.

Run ``python -m src.validate_data --train train.jsonl --val val.jsonl --test test.jsonl``.
"""

import argparse
import json
import sys

from src.data.wav_datamodule import validate_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training JSONL manifest")
    parser.add_argument("--val", required=True, help="Validation JSONL manifest")
    parser.add_argument("--test", required=True, help="Test JSONL manifest")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--num-samples", type=int, default=1024)
    args = parser.parse_args()
    try:
        report = validate_manifests(
            args.train, args.val, args.test, args.sample_rate, args.num_samples
        )
    except ValueError as error:
        print(json.dumps({"valid": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
