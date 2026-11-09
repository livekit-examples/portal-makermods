"""One Seeed reBot B601-DM with overhead and arm cameras.

    uv run --group rebot robot-rebot

Uses the same explicit control, park, and home RPCs as `robot`.
See tutorial/13-rebot.md for wiring, calibration, and Dream-1 setup.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from robot import main
from utils.rebot import build_follower


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true", help="stream sensors and log actions; never drive motors")
    parser.add_argument("--trace", type=Path, help="new JSONL file for read-only action/state comparisons")
    args = parser.parse_args()
    if args.trace and not args.read_only:
        parser.error("--trace requires --read-only")
    if args.read_only:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = args.trace or Path(f"data/rebot/actions-{stamp}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as trace:
            print(f"[robot] READ ONLY: motor commands blocked; action trace: {path.resolve()}")
            asyncio.run(main(partial(build_follower, read_only=True, trace=trace), "rebot"))
    else:
        asyncio.run(main(build_follower, "rebot"))


if __name__ == "__main__":
    run()
