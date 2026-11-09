# 2. Run Dream-1 and record corrections

This chapter's `dream-1` app uses the Metal pair. For the single-ReBot
`dream-1-tasks` app, use [chapter 13](13-rebot.md).

Dream-1 runs on Dream Labs' servers and needs no training dataset. Run it on
your task, correct it with the leaders, and record the corrected attempts for ACT.
A **take** is one attempt, marked with `r` to begin and `r` to end.

Complete [chapter 1](01-first-recording.md) first. Use the same task, camera
positions, and host configuration.

## 1. Configure access

On the **robot host**, create an API key at
[platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys) and add it to `.env`:

```dotenv
DREAM_API_KEY=dl_live_...
```

Keep the robot, recorder, and teleoperator from chapter 1 running. If you stopped
them, restart them in that order, in separate terminals. Keep other policies stopped.

## 2. Check targets without moving

In a fourth terminal on the robot host:

```bash
uv run dream-1 --task "fold the towel" --dry-run
```

1. In teleop, press `c` until `policy-dream-1` is selected. Without a teleop
   host, press `c` in the Dream-1 terminal instead; `c` there again releases control.
2. Check the targets printed by Dream-1. They should be near the current arm poses.
   Dry-run prints targets but sends no motion commands.
3. Press `c` to select the human again.
4. Stop Dream-1 with Ctrl-C. If the targets look wrong, check the camera views
   and arm readings before enabling motion.

## 3. Run and correct the policy

In the fourth terminal:

```bash
uv run dream-1 --task "fold the towel"
```

In teleop:

1. Select the human with `c`, then reset the towel and arms.
2. Press `r` to start a take.
3. Press `c` and confirm `policy-dream-1` is selected.
4. Keep the leaders near the followers. When the policy needs help, press `c`,
   confirm the human is selected, and guide the leaders to correct the motion.
5. Press `c` to let the policy continue. Repeat step 4 when needed.
6. Press `r` when the task is complete. To reject an attempt, press `[` before
   ending the take instead.

Repeat these steps to collect varied successful takes. A `NOT RECORDED` message
means the boundary did not reach an observer; fix the recorder connection
before continuing. One take checks the pipeline but is not enough to expect a
useful trained policy.

Recording continues through each handoff. Dream-1 starts a fresh plan when it
regains control. Target limits reduce command jumps; keep the leaders aligned
before taking over.

## 4. Convert and inspect

Finish the take and select the human. Stop Dream-1 with Ctrl-C, quit teleop with
`x`, stop the recorder with Ctrl-C, then stop the robot with Ctrl-C: it parks the followers first (they walk back to
the pose they started in and switch off), then exits.

The recorder saves `.rrd` archives under `data/sessions`. Keep them so you can
rebuild the dataset later. On the robot host, check the conversion report:

```bash
uv run livekit-portal convert data/sessions --output lerobotv3 --dry-run
```

Confirm it includes completed episodes and usable rows, then create the dataset:

```bash
uv run livekit-portal convert data/sessions --output lerobotv3 --repo-id local/portal-makermods
```

The dataset is saved to `data/local/portal-makermods`. Discarded and unfinished
takes are excluded. Conversion does not append to an existing dataset; use a
new repo ID for later rounds, as shown in chapter 3.

Continue to **[3. Train ACT](03-train-act.md)**. To change Dream-1's task while
it runs, or to pause it and hand over control by voice, see
[chapter 10](10-voice-tasks.md).

For implementation details, see [`src/policies/dream_1/run.py`](../src/policies/dream_1/run.py).

---

Previous: [1. Set up the rig](01-first-recording.md) · [Tutorial index](../README.md#the-tutorial) · Next: [3. Train ACT and improve it with corrections](03-train-act.md)
