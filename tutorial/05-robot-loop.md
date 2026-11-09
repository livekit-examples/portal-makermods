# 5. Robot loop

Open [`src/robot.py`](../src/robot.py).
The loop publishes both followers' state and three camera feeds, then applies
the latest accepted action. Hardware calls live in
[`src/utils/metal.py`](../src/utils/metal.py).

`robot-rebot` reuses this loop with one B601-DM and two cameras, supplied by
[`src/utils/rebot.py`](../src/utils/rebot.py). It selects the `rebot` profile
from `portal.yaml`; see [chapter 13](13-rebot.md). Connection and cleanup also
run on the hardware worker, so device calls stay serialized throughout.

## 1. Receive an action

Portal calls `on_action` for actions that pass the active-operator gate.
The callback saves their values; the hardware loop applies the latest values
on its next tick. This avoids issuing a motor transaction for every packet
when actions arrive in a burst.

The robot does not need to know whether an action came from a human or a policy.
All operators use the same schema and gate.

## 2. Publish observations and apply the action

The main loop contains this sequence:

```python
ts_us = robot.now_us()
observation = await hardware.run(followers.get_observation)
state, frames = split_state_frames(observation, tracks)
if state:
    robot.send_state(state, timestamp_us=ts_us)
for name, frame in frames.items():
    robot.send_video_frame(name, frame, timestamp_us=ts_us)
if latest_action:
    await hardware.run(followers.send_action, latest_action)
```

State is published only when both arms read: half a row would be invented
data. State and frames share one timestamp. That lets Portal match them into an
observation and lets the recorder align responses later. Frames are RGB
`uint8` arrays with shape `(height, width, 3)`.

`HardwareThread` serializes device calls while leaving the asyncio loop free
for network callbacks. Device calls finish before hardware disconnects.

## 3. Handle handoffs and faults

The robot starts with no active operator. On a gate change it clears its cached
action, so the previous operator’s command is not reused while waiting for the
new operator. If the active operator disconnects, it also releases the gate.

The followers ramp each joint target at most `METAL_MAX_STEP_RAD` per command,
starting from the measured pose and again after any pause in commands, and clamp
every joint to the soft limits. An arm is never commanded before it has
been read. This limits command changes during takeover; it is not a guarantee of
physical motor speed.

A failed arm read is logged and counted instead of ending the loop; that tick
publishes no state. A camera hiccup serves the last good frame.

A Metal arm has no brakes and drops when disabled, so a fault never disables
the motors. The one switch-off is a park, which `p` asks for over a Portal RPC
and every robot exit runs first: walk each follower back to the pose read at
startup, check it arrived, switch it off, and reopen it read-only so state keeps
flowing. A park that cannot finish leaves the arms holding. Startup opens both
arms or neither: if one fails, the other is switched off where it opened. See
[parking and power](09-reference.md#parking-and-power).

---

Previous: [4. The wire contract](04-wire-contract.md) · [Tutorial index](../README.md#the-tutorial) · Next: [6. Operator loop](06-operator-loop.md)
