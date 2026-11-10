"""Keep Filament on a process main thread while Portal's asyncio loop stays free."""
import multiprocessing
import traceback
from types import SimpleNamespace

from utils.mujoco_sim import Simulation, validate_action


def _serve(connection, cfg, kwargs):
    sim = None
    try:
        sim = Simulation(cfg, **kwargs)
        connection.send((True, (sim.limits, sim.version)))
        while True:
            command, kwargs = connection.recv()
            if command == "close":
                break
            connection.send((True, sim.tick(**kwargs)))
    except EOFError:
        pass
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        if sim:
            sim.close()
        connection.close()


class SimulationProcess:
    """Synchronous proxy, called only by the existing serialized HardwareThread.

    Spawn instead of fork: no inherited graphics state, Portal sockets or threads.
    Camera arrays are copied through the pipe; no views of reusable GPU buffers
    escape. A failed worker stops this robot rather than silently restarting it.
    """
    def __init__(self, cfg, **kwargs):
        # Portal's native config is not a process-safe Python value. Snapshot only
        # the simulator's inputs from the already-loaded portal.yaml schema.
        cfg = SimpleNamespace(fps=cfg.fps, video_tracks=list(cfg.video_tracks),
                              action_schema=[SimpleNamespace(name=f.name) for f in cfg.action_schema],
                              state_schema=[SimpleNamespace(name=f.name) for f in cfg.state_schema])
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, cfg, kwargs), name="filament-sim")
        self.process.start()
        child.close()
        try:
            self.limits, self.version = self._receive(timeout=120)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout=30):
        if not self.connection.poll(timeout):
            raise RuntimeError("simulation worker timed out")
        try:
            ok, value = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(f"simulation worker exited (code {self.process.exitcode})") from error
        if not ok:
            raise RuntimeError(f"simulation worker failed:\n{value}")
        return value

    def validate_action(self, values):
        return validate_action(values, self.limits)

    def tick(self, **kwargs):
        self.connection.send(("tick", kwargs))
        return self._receive()

    def close(self):
        if self.connection.closed:
            return
        try:
            self.connection.send(("close", {}))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()
