"""One Seeed reBot B601-DM with overhead and arm cameras.

    uv run --group rebot robot-rebot

Uses the same explicit control, park, and home RPCs as `robot`.
See tutorial/13-rebot.md for wiring, calibration, and Dream-1 setup.
"""
import argparse
import asyncio

from robot import main
from utils.rebot import build_follower


def run() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(main(build_follower, "rebot"))


if __name__ == "__main__":
    run()
