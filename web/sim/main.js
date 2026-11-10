import { Room, RoomEvent, Track } from "livekit-client";
import "./style.css";

const $ = (id) => document.getElementById(id);
let socket,
  room,
  identity,
  snapshot,
  limits = {},
  targets = {},
  selectedSide = "left";
let initializedGeneration = null;
let videoReady = new Set();
let cameraCount = 4;
let connected = false;
let view = { azimuth: 135, elevation: -25, distance: 2.45 };
let viewDirty = false;
let errorUntil = 0;
let taskDirty = false;
let taskPending = null;
let taskTimer = null;
let labels = {
  joint_0: "Joint 0 · Base",
  joint_1: "Joint 1 · Shoulder",
  joint_2: "Joint 2 · Elbow",
  joint_3: "Joint 3 · Wrist pitch",
  joint_4: "Joint 4 · Wrist yaw",
  joint_5: "Joint 5 · Wrist roll",
  gripper: "Gripper",
};

function tell(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  if (error) errorUntil = Date.now() + 5000;
}
function send(data) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(data));
}
function ownControl() {
  return (
    connected &&
    snapshot?.active === identity &&
    snapshot?.fresh &&
    !snapshot?.busy &&
    videoReady.size === cameraCount
  );
}
function setConnection(text, on = false) {
  $("connection").textContent = text;
  $("connection-dot").classList.toggle("on", on);
}
function displayValue(key, value) {
  return key.includes("gripper")
    ? `${Math.round(value * 100)}%`
    : value.toFixed(2);
}
function buildJoints() {
  $("joints").replaceChildren();
  for (const [key, range] of Object.entries(limits)) {
    if (!key.startsWith(selectedSide + "_")) continue;
    const joint = key.slice(selectedSide.length + 1).replace(".pos", "");
    const row = document.createElement("div");
    row.className = "joint";
    const heading = document.createElement("div");
    heading.className = "joint-label";
    const label = document.createElement("label");
    label.htmlFor = key;
    label.textContent = labels[joint] || joint;
    const output = document.createElement("output");
    output.id = `${key}-value`;
    output.htmlFor = key;
    const input = document.createElement("input");
    input.type = "range";
    input.id = key;
    input.min = range[0];
    input.max = range[1];
    input.step = 0.005;
    input.value = targets[key] ?? 0;
    input.disabled = !ownControl();
    output.textContent = displayValue(key, Number(input.value));
    input.addEventListener("input", () => {
      if (!ownControl()) return;
      targets[key] = Number(input.value);
      output.textContent = displayValue(key, targets[key]);
    });
    heading.append(label, output);
    row.append(heading, input);
    $("joints").append(row);
  }
}
function updateControls() {
  const fresh =
    connected &&
    snapshot?.fresh &&
    !snapshot?.busy &&
    videoReady.size === cameraCount;
  $("take").disabled = !fresh || snapshot?.active === identity;
  $("hold").disabled = !connected || !snapshot?.active;
  $("handoff").disabled = !fresh || !$("policy").value;
  $("home").disabled = !fresh || snapshot?.recording;
  $("reset").disabled = !fresh || snapshot?.recording;
  $("setup").disabled = !fresh || snapshot?.recording;
  $("record").disabled = !fresh;
  $("discard").disabled = !fresh || !snapshot?.recording;
  $("task").disabled = !!snapshot?.recording;
  for (const input of $("joints").querySelectorAll("input"))
    input.disabled = !ownControl();
}
function renderState(data) {
  snapshot = data;
  const mine = data.active === identity;
  if (!mine || initializedGeneration !== data.generation) {
    if (data.state) targets = { ...data.state };
    initializedGeneration = data.generation;
    for (const [key, value] of Object.entries(targets)) {
      if ($(key)) {
        $(key).value = value;
        $(`${key}-value`).textContent = displayValue(key, value);
      }
    }
  }
  $("operator-name").textContent = mine
    ? "You · Browser teleop"
    : data.active || "Nobody · Holding";
  $("operator-dot").classList.toggle("on", !!data.active);
  const operators = (data.operators || [])
    .filter((name) => name !== identity && !name.startsWith("browser-"))
    .sort();
  const signature = operators.join("|");
  if ($("policy").dataset.signature !== signature) {
    const previous = $("policy").value;
    $("policy").replaceChildren();
    if (!operators.length)
      $("policy").add(new Option("No policy connected", ""));
    for (const name of operators)
      $("policy").add(
        new Option(
          name === "policy-sim-demo" ? "Demo · waist sweep" : name,
          name,
        ),
      );
    if (operators.includes(previous)) $("policy").value = previous;
    $("policy").dataset.signature = signature;
  }
  $("sim-clock").textContent =
    `Episode ${data.episode ?? "—"} · ${(data.sim_time || 0).toFixed(1)} s`;
  $("rate").textContent =
    `${Math.min(9.99, data.realtime_factor || 0).toFixed(2)}× real time`;
  $("record-status").textContent = data.recording ? "Recording" : "Idle";
  if (data.dream_task) {
    if (taskPending === data.dream_task.task) taskPending = null;
    if (
      !taskDirty && !taskPending && !data.recording &&
      document.activeElement !== $("task") && data.dream_task.task
    )
      $("task").value = data.dream_task.task;
    $("task-status").textContent = taskDirty
      ? "Editing task…"
      : taskPending
        ? "Setting task…"
        : data.dream_task.message;
  }
  $("record").textContent = data.recording ? "End take" : "● Start take";
  updateControls();
  if (Date.now() > errorUntil)
    tell(
      data.busy
        ? "Resetting the scene; control is released."
        : !data.fresh
          ? "Waiting for fresh observations…"
          : videoReady.size < cameraCount
            ? "Connecting camera streams…"
            : mine
              ? "You have control. Adjust joint targets; Hold releases control."
              : data.active
                ? `Watching ${data.active}. Take control to intervene.`
                : "Ready. Take control or hand off to a connected policy.",
    );
}
async function connect() {
  $("login").close();
  setConnection("Connecting…");
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`,
  );
  socket.addEventListener("message", async ({ data: raw }) => {
    const data = JSON.parse(raw);
    if (data.type === "error") {
      if (taskPending) {
        taskPending = null;
        taskDirty = true;
      }
      tell(data.message, true);
      return;
    }
    if (data.type === "state") {
      renderState(data);
      return;
    }
    if (data.type !== "ready") return;
    identity = data.identity;
    $("embodiment").textContent = `${data.arm_label} · MuJoCo · Human-in-the-loop`;
    $("arm-caption").textContent = data.profile === "yam" ? "DUAL YAM" : "DUAL WIDOWX";
    document.title = `${data.arm_label} · Portal Robot Lab`;
    const offset = data.profile === "yam" ? 1 : 0;
    labels = { gripper: "Gripper" };
    ["Base", "Shoulder", "Elbow", "Wrist pitch", "Wrist yaw", "Wrist roll"].forEach((name, i) => {
      labels[`joint_${i + offset}`] = `Joint ${i + offset} · ${name}`;
    });
    if (data.task !== null) $("task").value = data.task;
    else send({ type: "task", task: $("task").value });
    $("setup").replaceChildren();
    for (const [value, label] of Object.entries(data.setups))
      $("setup").add(new Option(label, value));
    limits = data.limits;
    cameraCount = data.cameras.length;
    buildJoints();
    $("session-info").textContent = identity;
    room = new Room({ adaptiveStream: false });
    room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (track.kind !== Track.Kind.Video || participant.identity !== "robot")
        return;
      const element = $(publication.trackName);
      if (!element) return;
      track.attach(element);
      element.addEventListener(
        "loadeddata",
        () => {
          videoReady.add(publication.trackName);
          element.parentElement.classList.add("has-video");
          updateControls();
        },
        { once: true },
      );
      element
        .play()
        .catch(() => tell("Click a camera to allow video playback.", true));
      element.onclick = () => element.play();
    });
    room.on(RoomEvent.TrackUnsubscribed, (track, publication) => {
      track.detach();
      videoReady.delete(publication.trackName);
      $(publication.trackName)?.parentElement.classList.remove("has-video");
      if (snapshot?.active === identity)
        send({ type: "control", identity: null });
      updateControls();
    });
    room.on(RoomEvent.Disconnected, () => {
      connected = false;
      videoReady.clear();
      send({ type: "control", identity: null });
      setConnection("Video disconnected");
      updateControls();
    });
    room.on(RoomEvent.Reconnecting, () => {
      connected = false;
      send({ type: "control", identity: null });
      setConnection("Reconnecting video…");
      updateControls();
    });
    room.on(RoomEvent.Reconnected, () => {
      connected = true;
      setConnection("Connected", true);
      updateControls();
    });
    try {
      await room.connect(data.url, data.token);
      connected = true;
      setConnection("Connected", true);
      updateControls();
    } catch (error) {
      tell(`Video connection failed: ${error.message}`, true);
      socket.close();
    }
  });
  socket.addEventListener("close", () => {
    connected = false;
    room?.disconnect();
    videoReady.clear();
    setConnection("Disconnected");
    updateControls();
    tell(
      "Connection closed. Reload to reconnect; control has been released.",
      true,
    );
  });
}
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: $("access-code").value }),
    });
    if (!response.ok) throw new Error(await response.text());
    $("access-code").value = "";
    await connect();
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});
$("login").addEventListener("cancel", (event) => event.preventDefault());
$("take").onclick = () => send({ type: "control", identity });
$("hold").onclick = () => send({ type: "control", identity: null });
$("handoff").onclick = () =>
  send({ type: "control", identity: $("policy").value, task: $("task").value });
function submitTask() {
  clearTimeout(taskTimer);
  const task = $("task").value.trim().replace(/\s+/g, " ");
  if (!task || task.length > 200) {
    tell("Enter one task of 1–200 characters.", true);
    return;
  }
  send({ type: "task", task });
  taskPending = task;
  taskDirty = false;
}
$("task").addEventListener("input", () => {
  taskDirty = true;
  clearTimeout(taskTimer);
  taskTimer = setTimeout(submitTask, 600);
});
$("task").addEventListener("change", submitTask);
$("task").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    submitTask();
  }
});
$("policy").onchange = updateControls;
$("home").onclick = () => send({ type: "motion", command: "home" });
$("reset").onclick = () => send({ type: "motion", command: `reset:${$("setup").value}` });
$("record").onclick = () =>
  send({
    type: "record",
    mark: snapshot?.recording ? "idle" : "recording",
    task: $("task").value,
  });
$("discard").onclick = () =>
  send({ type: "record", mark: "discard", task: $("task").value });
for (const side of ["left", "right"])
  $(side + "-tab").onclick = () => {
    selectedSide = side;
    for (const name of ["left", "right"]) {
      $(name + "-tab").classList.toggle("selected", name === side);
      $(name + "-tab").setAttribute("aria-pressed", name === side);
    }
    buildJoints();
  };
function releaseOnLeave() {
  if (snapshot?.active === identity) send({ type: "control", identity: null });
}
window.addEventListener("blur", releaseOnLeave);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseOnLeave();
});
window.addEventListener("pagehide", () => {
  releaseOnLeave();
  socket?.close();
  room?.disconnect();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") releaseOnLeave();
});
let drag = null;
$("scene-view").addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY };
  $("scene-view").setPointerCapture(event.pointerId);
});
$("scene-view").addEventListener("pointermove", (event) => {
  if (!drag) return;
  view.azimuth = (view.azimuth - (event.clientX - drag.x) * 0.4 + 360) % 360;
  view.elevation = Math.max(
    -85,
    Math.min(-5, view.elevation + (event.clientY - drag.y) * 0.3),
  );
  drag = { x: event.clientX, y: event.clientY };
  viewDirty = true;
});
for (const name of ["pointerup", "pointercancel"])
  $("scene-view").addEventListener(name, () => {
    drag = null;
  });
$("scene-view").addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    view.distance = Math.max(
      0.5,
      Math.min(3, view.distance + event.deltaY * 0.002),
    );
    viewDirty = true;
  },
  { passive: false },
);
$("view-reset").onclick = () => {
  view = { azimuth: 135, elevation: -25, distance: 2.45 };
  viewDirty = true;
};
setInterval(() => {
  if (ownControl() && !document.hidden && document.hasFocus())
    send({
      type: "action",
      values: targets,
      anchor: snapshot.anchor,
      generation: snapshot.generation,
    });
  if (viewDirty && connected) {
    send({ type: "view", view });
    viewDirty = false;
  }
}, 1000 / 30);
try {
  const response = await fetch("/api/session");
  if (response.ok) await connect();
  else $("login").showModal();
} catch {
  $("login").showModal();
}
