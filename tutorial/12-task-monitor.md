# 12. Watch Dream-1 finish, or get stuck

[Chapter 10](10-voice-tasks.md) lets you change Dream-1's task and pause it by
voice, but you still have to watch the arms to know when to say "stop". Here you
add a task monitor: a process that watches the overhead camera while Dream-1
drives and judges, with a vision-language model, how far along the task is. When
it judges the task done, or Dream-1 stuck, it takes control from Dream-1 so the
arms hold, and the voice agent tells you:

> **You:** Put the blue cube in the box, and go.\
> **Agent:** On it: Dream-1 is putting the blue cube in the box.\
> **You:** How's it going?\
> **Agent:** The monitor estimates it's about forty percent done, and rising.\
> **Agent:** *(unprompted)* The monitor thinks the blue cube is in the box, so it paused Dream-1. What's next?\
> **You:** Put the red cube in the box, and go.\
> **Agent:** *(a minute later)* Dream-1 seems stuck on the red cube: no progress in forty-five seconds, so it's paused. Try again, a different task, or teleop?

The monitor scores with [TOPReward](https://huggingface.co/docs/lerobot/topreward),
LeRobot's zero-shot reward model. It needs no training and no labels, but its
estimate is a model's judgment of a video, not a measurement. Treat a verdict as
a convenience: the monitor is not a safety system, and it never parks the arms.

Complete [chapter 10](10-voice-tasks.md) first: this chapter uses the same
`dream-1-tasks` operator and voice agent.
That operator now uses the single ReBot rig from [chapter 13](13-rebot.md).
Set `PORTAL_PROFILE=rebot` on the monitor host to subscribe to the same
seven-value/two-camera contract. The scoring camera remains `overhead_camera`.

## 1. How it fits

| Process | Identity | What it does |
|---|---|---|
| `uv run dream-1-tasks` | `policy-dream-1-tasks` (Portal operator) | Drives the arms; answers `dream.status` with its task |
| `uv run --group monitor task-monitor` | `task-monitor` (Portal observer) | Scores Dream-1's runs; clears the control gate on a verdict |
| `uv run --group voice voice-agent` | `voice-agent` (not a Portal peer) | Answers `monitor.report`; speaks the estimate and the verdicts |

```text
cameras ──▶ robot ──overhead frames──▶ task monitor ──monitor.report──▶ voice agent ──▶ you
                ▲                           │
                └── set_active_operator(None) on a verdict: the arms hold
```

The monitor is a Portal observer, like the recorder, so it can never send
actions. A **run** starts when it sees `policy-dream-1-tasks` take control; it
asks Dream-1 for the task over `dream.status`. The run ends on a verdict, or
when anyone else takes control (a pause, "teleop", or a park), which needs no
verdict. On a verdict the monitor clears the robot's control gate, exactly as a
pause does: no operator drives and the followers hold under power. Only
"continue" (or `c`) gives Dream-1 control again, and that starts a new run.

Every few seconds, and at once on a verdict, the monitor calls `monitor.report`
on the `voice-agent` participant with a JSON report:

| Field | Value |
|---|---|
| `state` | `idle` (no run yet), `watching`, `success`, `stuck`, or `stopped` (ended without a verdict) |
| `run`, `task`, `elapsed_s` | which run, Dream-1's task, seconds since Dream-1 took control |
| `progress`, `trend` | the newest estimate, 0 to 1, and `rising`, `flat`, or `falling` over the last 10 seconds |
| `done_for_s`, `hold_s` | while it looks done: how long it has, and how long it must before the verdict |
| `reason`, `released` | why the verdict, and whether the monitor took control from Dream-1 |
| `dry_run` | the monitor reports verdicts but never takes control |

The voice agent keeps the newest report and gives it to the LLM with the
instructions before every reply, so "how is it going?" always gets the newest
estimate, never one from earlier in the conversation. The agent also announces
each `success` or `stuck` on a run it saw being watched. Without a voice agent in the room, the
monitor still pauses Dream-1; it logs once that nobody takes the reports.

## 2. Install the monitor

TOPReward runs Qwen3-VL, an open vision-language model, on the monitor's host.
The default, `Qwen/Qwen3-VL-8B-Instruct`, is about 17 GB of weights in bfloat16:
use an NVIDIA GPU with 24 GB, or an Apple silicon Mac with 32 GB or more. The
weights pass through system memory as they load, so leave that much free RAM
too. The first start downloads the model from Hugging Face.

```bash
uv sync --group monitor --group voice   # voice too, if the agent runs on this host
```

On a smaller GPU, set `MONITOR_VLM=Qwen/Qwen3-VL-4B-Instruct` (about 9 GB) or
`Qwen/Qwen3-VL-2B-Instruct` (about 4.5 GB). Smaller models judge less reliably;
calibrate again after a change. The monitor needs only the overhead camera's
video, so it can run on any host that reaches the LiveKit server.

## 3. Calibrate with a dry run

The estimate's scale depends on the model, the camera, and the scene, so check
the thresholds on your rig before letting the monitor pause anything. With the
robot and `dream-1-tasks` running (start the robot first: the monitor connects
only once it can sync to the robot's clock):

```bash
uv run --group monitor task-monitor --dry-run
```

Then let Dream-1 attempt the task a few times ("continue", and "stop" when it is
done or clearly failing). Each score is a line:

```text
[task-monitor] run 2: Dream-1 drives on 'put the blue cube in the box'
[task-monitor] run 2 at 6s: 21% done (log P(True) -1.71), first score
[task-monitor] run 2 at 18s: 47% done (log P(True) -0.98), rising
[task-monitor] run 2 at 31s: 91% done (log P(True) -0.21), rising; above 85% for 0s of 10s
[task-monitor] run 2 at 37s: 93% done (log P(True) -0.17), rising; above 85% for 6s of 10s
[task-monitor] run 2 at 41s: 92% done (log P(True) -0.19), flat; above 85% for 10s of 10s
[task-monitor] run 2: success: an estimated 92% done, and above 85% for 10 seconds; dry run, so Dream-1 keeps control
```

A dry run reports its verdicts, so the voice agent announces them, but it never
takes control, and it keeps scoring after a verdict so you see the whole curve.
Choose `MONITOR_SUCCESS` above what runs reach before the task is done, and
below what they reach once it is. If finished runs never get there, try a larger
model or a camera position that shows the goal clearly. Then set
`MONITOR_SUCCESS_HOLD_S` to how long Dream-1 still needs once the goal looks
reached, for example to let go of the object and move clear.

## 4. Run it

```bash
uv run --group monitor task-monitor
```

Start the voice agent as in chapter 10 and talk to it:

| Say | The agent |
|---|---|
| "How's it going?", "is it done?", "how far along is it?" | answers from the monitor's newest report |
| *(nothing: the monitor judges the task done)* | says so, and asks for the next task |
| *(nothing: the monitor judges Dream-1 stuck)* | says why, and offers "continue", another task, or teleop |
| "Continue" after a verdict | calls `dream.resume`: a new run starts from where the arms are |

The arms hold where they are after a verdict. Park them with "park the arms"
or `p` when you are done.

## 5. How it decides

**The clip.** While a run lasts, the monitor keeps the overhead camera's frames
in memory, two per second, shrunk to 448 pixels wide. When it holds 64, it drops
every other one and halves the rate, so the clip always covers the whole run in
bounded memory.

**A score.** Every `MONITOR_EVERY_S` seconds, or as fast as the model allows, it
picks 16 frames spread from the start of the run to now and asks TOPReward's
question: the video, then "The above video shows a robot manipulation trajectory
that completes the following task: *put the blue cube in the box* Decide whether
the above statement is True or not. The answer is:". TOPReward's reward is
log P("True"), whose scale varies from scene to scene. The monitor reads the
same forward pass for "False" too and uses P(True) / (P(True) + P(False)), a
0 to 1 estimate. The progress it logs and reports is the newest score. Whether
it is still rising, and its trend, use the median of the last three scores, so
one odd score moves neither.

**Done** when every score has been at or above `MONITOR_SUCCESS` (0.85) for
`MONITOR_SUCCESS_HOLD_S` (10 seconds). Dream-1 keeps driving while the monitor
waits, so it can finish the task, and a score below the line starts the wait
again: the object may have slipped. While it waits, the arms standing still do
not count as stuck.

**Stuck** when either:

- the estimate has not risen by `MONITOR_GAIN` (5 points) in `MONITOR_STUCK_S`
  (45 seconds), or
- no joint has moved more than about 1° (0.02 rad), and no grip more than 2% of
  its range, in `MONITOR_STILL_S` (15 seconds). This one needs no model.

The settings are in the [reference](09-reference.md#task-monitor). A run the
monitor joins halfway, because it started after Dream-1 took control, is judged
from what it saw.

## 6. How the code works

[`src/task_monitor.py`](../src/task_monitor.py) has four parts:

- `Clip` holds a run's frames and samples them evenly.
- `Run` holds one run's scores and arm poses and decides once: `add_score` and
  `add_pose` each return a new verdict, or `None`.
- `Scorer` loads LeRobot's `TOPRewardModel` and its encoder, which tokenizes the
  frames and the prompt for Qwen3-VL, and reads the logits for both answers.
  It runs on a worker thread, off the event loop.
- `Monitor.tick` runs four times a second: it follows the control gate, starts
  and ends runs, feeds the clip, starts a score when the last one is done,
  releases control on a verdict, and sends reports without waiting on them.

The voice agent's side is in [`src/voice_agent.py`](../src/voice_agent.py):
`PolicyLink` registers `monitor.report` on its connection to the robot's room,
`describe_report` turns a report into the sentence the LLM reads,
`VoiceControl.llm_node` adds that sentence to the instructions of each LLM call
(never to the stored conversation), and `on_verdict` makes the agent speak
without being asked.

The [hardware-free checks](../test/README.md) cover the clip, the verdicts, and
the gate with a stub model (`test_task_monitor`), and the monitor, a real
operator, and the voice agent's connection over a local server
(`test_monitor_voice`).

## Troubleshooting

- **`cannot read Dream-1's task`.** The monitor asks `dream-1-tasks` for its
  task when a run starts. It watches only `uv run dream-1-tasks`, not `uv run dream-1`.
- **The monitor does not start: `TimeSyncTimeout`.** Start the robot first, with
  the same `LIVEKIT_URL` and `LIVEKIT_ROOM`.
- **`no voice agent took the report`.** Start the voice agent, with the same
  `LIVEKIT_ROOM`; it joins that room as `voice-agent`. The monitor keeps trying.
- **The agent says no task monitor is running.** Its newest report is more than
  15 seconds old: check the monitor's terminal.
- **Done too early, or never.** Calibrate `MONITOR_SUCCESS` with `--dry-run`
  (section 3). If the estimate is right but Dream-1 still had to finish, such as
  letting go of the object, raise `MONITOR_SUCCESS_HOLD_S`. A goal the overhead camera cannot see, such as an object hidden by
  the arm, cannot be judged from that camera; `MONITOR_CAMERA` picks another.
- **Stuck while Dream-1 works.** Slow tasks plateau between steps: raise
  `MONITOR_STUCK_S`. Pauses longer than `MONITOR_STILL_S` with the arms still
  also count as stuck.
- **Scores come slowly.** Each score is one model forward pass; the log shows
  each one. Use a GPU, or a smaller `MONITOR_VLM`.
- **Out of memory loading the model.** Use a smaller `MONITOR_VLM`, or set
  `MONITOR_DEVICE=cpu` to check the setup, slowly.

---

Previous: [10. Steer Dream-1 by voice](10-voice-tasks.md) · [Tutorial index](../README.md#the-tutorial)
