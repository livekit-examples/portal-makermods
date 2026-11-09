"""An ACT checkpoint as a Portal operator. Keys here: `c` takes or releases control, `p` parks or powers up."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import DataProcessorPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.common import (portal_config, HardwareThread, claim_or_release, env_int, load_env, mint_token, pace,
                          request_park, required_env)
from utils.hotkeys import Hotkeys
from livekit.portal import Operator, OperatorConfig, frame_bytes_to_numpy_rgb

IDENTITY = "policy-act"


def dataset_fields(root: Path, cfg) -> tuple[list[str], list[str]]:
    """Use training metadata order: normalizer vectors and model outputs are positional."""
    features = json.loads((root / "meta" / "info.json").read_text())["features"]
    fields = []
    for feature, schema in (("observation.state", cfg.state_schema), ("action", cfg.action_schema)):
        names = features[feature]["names"]
        expected = {field.name for field in schema}
        if not isinstance(names, list) or len(names) != len(expected) or set(names) != expected:
            raise ValueError(f"Dataset {feature} names do not match portal.yaml: {names}")
        fields.append(names)
    return fields[0], fields[1]


def build_batch(obs, cameras, state_keys, task):
    batch = {"observation.state": torch.tensor(
        [float(obs.state[key]) for key in state_keys], dtype=torch.float32), "task": task}
    for name in cameras:
        frame = obs.frames[name]
        rgb = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)
        batch[f"observation.images.{name}"] = (
            torch.from_numpy(np.array(rgb, copy=True)).float().permute(2, 0, 1).contiguous() / 255.0
        )
    return batch


def action_dict(tensor, keys):
    values = tensor.detach().cpu().reshape(-1).tolist()
    if len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("ACT returned an invalid action vector")
    return dict(zip(keys, map(float, values)))


async def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained_model directory")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Training dataset with meta/info.json")
    parser.add_argument("--task", default=os.environ.get("PORTAL_DATA_TASK", "MakerMods Metal bimanual manipulation"))
    args = parser.parse_args()

    url, room = required_env("LIVEKIT_URL"), required_env("LIVEKIT_ROOM")
    cfg = portal_config(OperatorConfig, room)
    fps = env_int("PORTAL_FPS", cfg.fps)
    state_keys, action_keys = dataset_fields(args.dataset_root.expanduser(), cfg)
    checkpoint = str(args.checkpoint.expanduser().resolve())
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    cameras = tuple(key.removeprefix("observation.images.") for key in policy.config.input_features
                    if key.startswith("observation.images."))
    missing = set(cameras) - set(cfg.video_tracks)
    if missing:
        raise ValueError(f"Checkpoint cameras missing from portal.yaml: {sorted(missing)}")
    # Replan on each observation; reset the ensemble after a human correction.
    policy.config.temporal_ensemble_coeff = 0.01
    policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    preprocessor = DataProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}})
    postprocessor = DataProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    def infer(obs):
        with torch.inference_mode():
            batch = preprocessor(build_batch(obs, cameras, state_keys, args.task))
            return action_dict(postprocessor(policy.select_action(batch)), action_keys)

    op = Operator(cfg)
    worker = HardwareThread("act-inference")
    latest_obs = None
    control_changed = True

    def on_observation(obs):
        nonlocal latest_obs
        latest_obs = obs

    def on_active_operator_changed(identity):
        nonlocal control_changed
        control_changed = True
        print(f"[policy-act] active operator: {identity}")

    op.on_observation(on_observation)
    op.on_active_operator_changed(on_active_operator_changed)
    hotkeys = Hotkeys(global_tap=False)  # this terminal only; works over SSH
    try:
        await op.connect(url, mint_token(IDENTITY, room))
        me = op.local_identity()
        hotkeys.start()
        print(f"[policy-act] ready on {device}; c takes control, p parks")
        async for _ in pace(fps):
            keys = hotkeys.drain()
            if "p" in keys:
                await request_park(op, "policy-act")
            if "c" in keys:
                await claim_or_release(op, "policy-act")
            obs = latest_obs
            if op.active_operator() != me or obs is None or any(c not in obs.frames for c in cameras):
                continue
            if control_changed:
                # Drop the actions ACT queued before the handoff.
                control_changed = False
                await worker.run(policy.reset)
            try:
                action = await worker.run(infer, obs)
            except Exception:
                logging.exception("ACT inference failed")
                continue
            # A handoff during inference invalidates this result, even if control came back.
            if not control_changed:
                op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[policy-act] stopping")
    finally:
        hotkeys.stop()
        worker.close()
        try:
            await op.disconnect()
        finally:
            op.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
