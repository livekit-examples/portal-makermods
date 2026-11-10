# 8. Handoff

HITL uses two operators in one room: the human and the policy. The robot’s
active-operator gate accepts only one sender at a time.

## 1. Select the operator

The teleoperator’s `cycle_active_operator` builds a ring containing itself and
the other connected operators. Pressing `c` advances through that ring:

```python
ring = [me, *sorted(op.operators())]
current = op.active_operator()
target = ring[(ring.index(current) + 1) % len(ring)] if current in ring else me
await op.set_active_operator(target)
```

With teleop and ACT connected, the sequence is:

```text
no operator → teleoperator → policy-act → teleoperator → ...
```

The call is an RPC to the robot. If it fails, teleop reports the error; press
`c` again to retry. Neither operator claims control automatically.

`uv run dream-1-tasks` ([chapter 10](10-voice-tasks.md)) makes the same call when
asked: its `dream.pause`, `dream.resume`, `dream.to_teleop`,
`dream.to_hand_teleop`, and `dream.stop_hand_teleop` RPCs, which the voice agent
calls, release control, take it, pass it to `teleoperator` or `hand-teleop`, or
take it from `hand-teleop` and leave no operator active.

## 2. Keep recording across handoffs

The [ACT walkthrough](03-train-act.md#4-record-corrections) gives the keypresses
for a complete correction. A single take can include both policy and human actions:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/handoff-take-dark.svg">
  <img alt="One take from r to r: control passes teleop, policy, teleop for a correction, policy, each switch a press of c, while the recorder writes continuously." src="images/handoff-take-light.svg" width="100%">
</picture>

The recorder stays connected throughout. It captures accepted actions from
both senders, so the corrected trajectory can become another training episode.
Keep each leader near its follower’s pose before takeover to reduce the change
in commanded target.

## 3. Clear old commands

The robot clears its action cache when the gate changes and releases the gate
when the active operator disconnects. ACT resets its action history when it
becomes active again. If control changes while inference is running, that
inference result is discarded.

The follower’s step limiter bounds changes to commanded targets. Clearing the
gate stops accepting new operator actions; it does not disable motor torque.

---

Previous: [7. HITL recording](07-hitl-recording.md) · [Tutorial index](../README.md#the-tutorial) · Next: [9. Configuration and troubleshooting](09-reference.md)

## Simulation handoff

[Chapter 14](14-mujoco-sim.md) uses the same gate for a browser operator and
a policy on two simulated WidowX arms. Both start idle. The browser can select
an operator, release control, or reset the scene. Reset and handoff invalidate
old targets and observations; a command timeout releases control without
automatically choosing a replacement.
