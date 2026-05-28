"""CLI for pulling TritonPilot recordings into TritonAnalysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pilot_transfer import DEFAULT_INBOX, DEFAULT_PILOT_TRANSFER_URL, sync_from_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync saved TritonPilot media from the transfer server.")
    parser.add_argument(
        "base_url",
        nargs="?",
        default=DEFAULT_PILOT_TRANSFER_URL,
        help="TritonPilot transfer server URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INBOX,
        help="Local TritonAnalysis inbox directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would copy without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Recopy files even if size and mtime match.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = sync_from_pilot(
            args.base_url,
            args.output,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Pilot transfer failed: {exc}", file=sys.stderr)
        return 2

    action = "Would copy" if summary.dry_run else "Copied"
    print(f"Pilot transfer source: {summary.base_url}")
    print(f"Destination: {summary.destination}")
    print(f"Scanned: {summary.scanned} file(s)")
    print(f"{action}: {summary.copied} file(s), {summary.bytes_copied} byte(s)")
    print(f"Skipped current: {summary.skipped} file(s)")
    for path in summary.copied_paths[:20]:
        print(f"  {path}")
    if len(summary.copied_paths) > 20:
        print(f"  ... {len(summary.copied_paths) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
