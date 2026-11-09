# 10. Steer Dream-1 by voice

Dream-1 follows a one-line task prompt. In [chapter 2](02-dream-1.md) that prompt
is fixed when the process starts, and `c` is the only way to hand over control.
Here `dream-1-tasks` drives one Seeed ReBot B601-DM with two cameras. Complete
[chapter 13](13-rebot.md) first. The voice agent lets you change its task,
pause and resume Dream-1, send the arm home, and park it by talking:

> **You:** Put the blue cube in the box.\
> **Agent:** Task set; say "continue" when you're ready for me to start.\
> **You:** Continue. *(Dream-1 takes control and reaches for the blue cube.)*\
> **You:** Wait, stop! Put the red cube in the box instead. *(The arms hold as soon as "wait" is heard.)*\
> **Agent:** I've switched the task to the red cube; say "continue" when you're ready.\
> **You:** Continue. *(Dream-1 starts on the red cube from where the arms are.)*\
> **You:** Reset the arms. *(They walk back to their rest pose and hold there, powered.)*\
> **You:** Continue. *(Dream-1 starts its task again from the rest pose.)*\
> **You:** Park the arms. *(They walk back to their rest pose and switch off.)*

**A new task never starts motion.** Changing the task while Dream-1 drives pauses
it first; only "continue" (or `c`) hands the arms back. Voice is not an emergency
stop: a spoken "stop" takes effect a second or two after you say it. To intervene
faster, release control with `c` in the Dream-1 terminal or park with `p`.

Keep `robot-rebot` running in the ReBot room. Set `PORTAL_PROFILE=rebot` on
any recorder or task-monitor host. The handoff RPCs below remain available for
a compatible human operator, but the supplied Metal teleoperator and webcam
hand teleop use different kinematics and units; do not run them in this room.

## 1. How it fits

| Process | Identity | What it does |
|---|---|---|
| `uv run dream-1-tasks` | `policy-dream-1-tasks` (Portal operator) | Single ReBot Dream-1 loop with a switchable task; answers the `dream.*` RPCs |
| `uv run --group voice voice-agent` | `voice-agent` (not a Portal peer) | Turns what you say into those RPCs; never drives the arms |

```text
you ──speech──▶ voice agent ──RPC dream.*──▶ dream-1-tasks ──set_active_operator──▶ robot
```

The voice agent joins the robot's room as an ordinary LiveKit participant and
calls RPC methods on `policy-dream-1-tasks`, which changes its own prompt or
moves the robot's control gate, as `c` does. The agent has no Portal role, so it
never appears in teleop's `c` ring and cannot send actions.

"Home" and "park" are the two requests that go to the robot itself, so they work
whoever drives and whether or not `dream-1-tasks` runs. For "park" the agent
calls the robot's `park` RPC with the payload `"park"`: the robot takes control
from everyone, walks both arms back to their rest pose, and switches them off, as
`p` does. That payload only ever parks: on parked arms it does nothing. Powering
them up stays a `p` key in a terminal, so voice can switch the motors off but
never on.

For "home" or "reset the arms" it calls the robot's `home` RPC: the same walk back
to the rest pose, but the motors stay on and the arms hold there. Nobody drives
afterwards, so "continue" starts Dream-1 from the rest pose and "switch to
teleop" hands the arms to the leaders. On parked arms it does nothing.

| Method | Payload | Effect |
|---|---|---|
| `dream.status` | `{}` | None; reports the task and who drives |
| `dream.set_task` | `{"task": "put the red cube in the box"}` | Pauses Dream-1 if it drives, then changes the prompt |
| `dream.pause` | `{}` | Dream-1 releases control if it has it; the robot holds the arms |
| `dream.resume` | `{}` | Dream-1 takes control from whoever has it, starting a new episode |
| `dream.to_teleop` | `{}` | Control passes to `teleoperator`; the followers track the leaders |
| `dream.to_hand_teleop` | `{}` | Control passes to `hand-teleop` (`HAND_TELEOP_IDENTITY`); each arm holds until a hand grips it |
| `dream.stop_hand_teleop` | `{}` | If the hand teleop drives, no operator does afterwards and the robot holds the arms; otherwise nothing |

Every reply is the same JSON status: `task`, `presets`, `driving`,
`active_operator`, `teleoperator_connected`, `hand_teleop` (its identity), and
`hand_teleop_connected`. A pause leaves no operator active, so the robot sends no
commands and the followers hold their last target under power. A pause while
either teleop drives changes nothing. Both handoffs go only to an operator that
is connected; otherwise Dream-1 keeps what it has.

## 2. Install the voice agent

The agent follows LiveKit's
[agent-starter-python](https://github.com/livekit-examples/agent-starter-python):
speech-to-text, the LLM, and text-to-speech run on
[LiveKit Inference](https://docs.livekit.io/agents/models/), which needs a
LiveKit Cloud project. Run it on any host with a microphone and speaker, such as
the teleop host:

```bash
uv sync --group voice
sudo apt install libportaudio2   # Linux: audio for console mode
```

If `LIVEKIT_URL` is your Cloud project, nothing else is needed. With a
self-hosted server, such as `livekit-server --dev`, keep the robot's room there
and add a Cloud project's key for Inference only:

```dotenv
LIVEKIT_INFERENCE_API_KEY=YOUR_CLOUD_API_KEY
LIVEKIT_INFERENCE_API_SECRET=YOUR_CLOUD_API_SECRET
```

## 3. Start the task-switching Dream-1

Keep `robot-rebot` running. On the **policy host**, configure Dream as in
[chapter 13](13-rebot.md), give the session's tasks, then check the targets
without moving:

```bash
uv run dream-1-tasks --dry-run \
  --task "put the blue cube in the box" \
  --task "put the red cube in the box"
```

The first task is the starting one. Without `--task`, the list comes from
`PORTAL_DATA_TASKS`, the same setting teleop reads, then `PORTAL_DATA_TASK`.
With none of these, Dream-1 starts with no task and does not move, even when
handed control, until you give it one.
In this terminal, `1`–`9` switch task (pausing Dream-1 if it drives), `c` takes
or releases control, and `p` parks or powers up the arms. When the targets look
right, stop it with Ctrl-C and start it again without `--dry-run`.

## 4. Talk to it

On the host with the microphone:

```bash
uv run --group voice voice-agent console
```

The agent greets you with the current task and who is driving. Then:

| Say | The agent calls |
|---|---|
| "Put the green cube in the bowl", "do the second one" | `dream.set_task`, then asks you to say "continue" |
| "Continue", "go", "start", "back to Dream-1" | `dream.resume` |
| "Stop", "wait", "hold on", "pause" | `dream.pause` |
| "Switch to teleop", "I'll take over" | `dream.to_teleop` |
| "Hand teleop", "I'll use my hands" | `dream.to_hand_teleop` |
| "Disable hand teleop", "turn off hand teleop", "hands off" | `dream.stop_hand_teleop` |
| "Home the arms", "reset the arms" | the robot's `home` |
| "Park the arms", "put the arms away" | the robot's `park`, payload `"park"` |
| "What's happening?", "who's driving?" | `dream.status` |
| "How's it going?", "is it done?" | `dream.status`, plus the task monitor's estimate ([chapter 12](12-task-monitor.md)) |
| "Put the red cube in the box and go" | `dream.set_task`, then `dream.resume` |

The teleop and hand-teleop requests work only when a compatible operator is
connected. With this repository's single-ReBot setup, use task, pause, resume,
home, and park commands. A plain "stop" releases Dream-1 only; it does not
revoke a human operator's control.

"Stop", "wait", "hold on", "pause", and "freeze" are matched in the live
transcript and pause Dream-1 before you finish the sentence. The LLM still reads
the whole sentence, so "wait, stop, the red cube instead" both pauses and changes
the task. A false match only pauses.

To talk from a browser instead, run the agent in a room of its own and join that
room from any LiveKit frontend:

```bash
uv run --group voice voice-agent connect --room portal-voice
lk token create --join --room portal-voice --identity you   # a token for the frontend
```

Keep the conversation out of the robot's room: the agent listens to the first
participant it finds, and there that is a Portal peer, not you. The agent still
reaches Dream-1 through its own connection to `LIVEKIT_ROOM`.

## 5. Record the corrections

Recording remains separate from this policy. A ReBot-compatible recording
operator must start/end takes and label them. The existing Metal teleoperator
cannot be used for ReBot corrections. With a compatible operator, end the take
before changing a task, then begin a new take with the matching task label.
A spoken task changes Dream-1's prompt, not the label of an open take.

## 6. How the code works

[`src/policies/dream_1_tasks/run.py`](../src/policies/dream_1_tasks/run.py)
reuses chapter 2's scheduling and frame encoder. `RebotMapping` supplies
single-arm actions, custom ReBot IK, and padding for the model's absent left
arm/view (see chapter 13). `TaskControl` holds the task and
answers the RPCs. `set_task` releases control before it changes the prompt, and
sets `changed`. The loop then treats a task change like a handoff:

```python
if control_changed or tasks.changed:
    control_changed = tasks.changed = False
    ...
    await worker.run(client.reset_episode)
```

A new episode clears Dream-1's memory of the old task and its chunk blending, so
a chunk planned for the blue cube is never played under the red cube's prompt.
The arms ramp from where they are, as after a handoff.

[`src/voice_agent.py`](../src/voice_agent.py) keeps the starter's pipeline:
AssemblyAI speech-to-text with keyterms (`Dream-1`, `teleop`, `hand teleop`, `park`), Gemma 4 as the
LLM, Fish Audio speech with expressive mode, the LiveKit turn detector with
preemptive generation, and ai-coustics noise cancellation on the room's
microphone. On top of it, `VoiceControl` has nine tools: `status`,
`change_task`, `pause`, `resume`, `switch_to_teleop`, `switch_to_hand_teleop`,
`stop_hand_teleop`, `home_arms`, and `park_arms`. Each calls
one RPC through `PolicyLink`, the agent's own connection to the robot's room, and
returns the result as a sentence, so the agent reports only what happened. The
fast pause is a `user_input_transcribed` handler that matches stop words in
interim transcripts. `VOICE_STT`, `VOICE_LLM`, `VOICE_TTS`, and
`VOICE_TTS_VOICE` swap models; see the [reference](09-reference.md#voice-agent).

The [hardware-free checks](../test/README.md) cover the task and control rules
and the RPCs between the agent's connection and a real operator.

## Troubleshooting

- **The agent says Dream-1 is not in the room.** Start `uv run dream-1-tasks`,
  and check both use the same `LIVEKIT_URL` and `LIVEKIT_ROOM`.
- **The agent says the robot is not in the room.** "Home" and "park" go to the
  robot itself: start `uv run --group rebot robot-rebot` with the same `LIVEKIT_URL` and `LIVEKIT_ROOM`.
- **"Continue" after a park moves nothing.** Parked arms ignore actions. Press
  `p` in the robot, teleop, or `dream-1-tasks` terminal to power them up first.
- **"The teleoperator is not connected."** Connect a ReBot-compatible
  teleoperator; the supplied Metal leader app is not compatible.
- **"The hand teleop is not connected."** The supplied webcam hand app is for
  Metal. This RPC requires a ReBot-compatible hand operator. If you set
  `HAND_TELEOP_IDENTITY`, set it where `dream-1-tasks` runs too.
- **The agent cannot reach its models.** LiveKit Inference needs a Cloud
  project's key: `LIVEKIT_API_KEY` for a Cloud `LIVEKIT_URL`, otherwise
  `LIVEKIT_INFERENCE_API_KEY` and `LIVEKIT_INFERENCE_API_SECRET`.
- **`ai_coustics ... Unable to authorize model use`.** Noise cancellation runs
  on LiveKit Cloud rooms only. In a room on a self-hosted server it logs this and
  passes the audio through; remove the `noise_cancellation` line to silence it.
- **Console mode finds no microphone.** List devices with
  `voice-agent console --list-devices` and pick one with `--input-device`.
  The CLI may print that console mode is moving to `lk agent console`; it still works.
- **"Continue" does nothing.** Dream-1 sends nothing until both ReBot cameras and a complete state
  publish, as in chapter 13. Ask "what's happening?" and check the
  `dream-1-tasks` terminal.

---

Previous: [9. Configuration and troubleshooting](09-reference.md) · [Tutorial index](../README.md#the-tutorial) · Next: [12. Watch Dream-1 finish, or get stuck](12-task-monitor.md)
