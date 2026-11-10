# 7. HITL recording

A correction is useful training data only if it is paired with the observation
that prompted it. The human and policy both send an `in_reply_to_ts_us` anchor:
the human uses the camera view, while a policy (Dream-1 from chapter 2, or ACT)
uses its input observation.

## 1. Record continuously

[`src/recorder.py`](../src/recorder.py)
connects an `Observer`, waits for robot clock synchronization, and starts a
Portal `RrdSink`.

Start the robot first, then choose the same rig in a separate recorder process:

```bash
uv run recorder --profile widowx  # dual Trossen WidowX AI: 14 values, four cameras
uv run recorder --profile yam     # dual I2RT YAM: 14 values, four cameras
uv run recorder --profile rebot  # single Seeed B601-DM: seven values, two cameras
uv run recorder --profile metal  # original Metal pair
```

Run just the command for your room. `--profile` overrides `PORTAL_PROFILE`,
including settings loaded from `.env.local`. Without the flag, the recorder
continues to use `PORTAL_PROFILE`, falling back to Metal when unset. Both
state/action schemas and all camera names come from `portal.yaml` through
`portal_config`; the recorder does not pad or convert an arm's data. Check
the startup log for the selected profile, field counts and camera names.

For Trossen the cameras are overhead, left wrist, right wrist, and scene;
for ReBot they remain overhead and arm. Grippers stay in the profile's units
(normalized 0–1 for Trossen/ReBot, millimetres for Metal). Each profile's
recordings must be converted into its own compatible dataset.

To preview without writing a file, add `--observe-only` to the same command.
The Rerun layout shows only the arms present in that profile and all its
cameras. Ctrl-C or SIGTERM flushes the recorder before closing the archive.

The recording path still uses the standard Portal sink:

```python
sink = RrdSink("data/sessions", jpeg_quality=95)
obs.record_to(sink)
```

The observer receives state, frames, active-operator actions, and keypoints.
It writes continuously to `<room>-<timestamp>.rrd`, including time outside
marked takes. Recording continues when control changes or teleop restarts.

An action’s `sender` identifies its operator. Use that recorded identity rather
than looking up the current gate later, when it may have changed. The archive
records accepted action traffic; the robot applies the latest accepted action
per hardware tick, so it is not a log of every motor write.

## 2. Preserve both timestamps

| Timeline | Purpose |
|---|---|
| `robot_time` | State and frames at capture time; actions at their observation anchors |
| `action_time` | Each action at its send time on the robot’s synchronized clock |

Several actions can respond to one frame during a video stall. Reading only
`robot_time` can collapse them into one value per entity. Conversion reads
`action_time` so those responses remain distinct. Unanchored actions fall back
to their send time on `robot_time`.

## 3. Mark takes inside the archive

The teleoperator sends a keypoint when you press `r` or `[`:

```python
await op.send_keypoint("recording", {"task_description": task},
                       timestamp_us=view.anchor())
```

`recording` opens a take, `idle` closes it, and `discard` rejects it. The task
label belongs to the take, so task switching is blocked while recording.
Control handoff is allowed: one take can contain policy motion and a human
correction.

## 4. Convert completed takes

Use the [conversion steps in chapter 2](02-dream-1.md#4-convert-and-inspect),
choosing a new dataset ID for each round.

For each action, conversion selects the newest state and frames at or before
its anchor. The default allows 200 ms of staleness; `--max-stale-ms 0` requires
exact timestamps. Rows without required samples inside the limit are skipped.

Discarded and unclosed takes are skipped. If there are no `recording` markers,
the whole archive becomes one episode. Archives combined into one dataset must
have compatible schemas. Field order comes from the archive.

Keep the archives so you can rebuild the dataset without collecting again.
The [hardware-free checks](../test/README.md) verify archive alignment.

---

Previous: [6. Operator loop](06-operator-loop.md) · [Tutorial index](../README.md#the-tutorial) · Next: [8. Handoff](08-handoff.md)
