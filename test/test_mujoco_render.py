"""Four real camera renders; requires a graphics context (MUJOCO_GL=egl on Linux)."""
import time
import numpy as np
from livekit.portal import RobotConfig
from utils.common import HardwareThread, portal_config
from utils.mujoco_sim import Simulation
from utils.sim_process import SimulationProcess


async def check(profile, renderer):
    worker = HardwareThread('render-check')
    sim = None
    try:
        factory = SimulationProcess if renderer == 'filament' else Simulation
        sim = await worker.run(lambda: factory(portal_config(RobotConfig, 'render-test', profile),
                                              width=640, height=480, arms=profile, renderer=renderer))
        state, frames, _ = await worker.run(sim.tick)
        assert len(state) == 14 and len(frames) == 4
        for name, frame in frames.items():
            assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, name
            assert frame.std() > 5, f'{name} is blank'
        original = {name: frame.copy() for name, frame in frames.items()}
        started = time.perf_counter()
        for _ in range(5):
            _, moved, status = await worker.run(lambda: sim.tick(view=dict(azimuth=100, elevation=-35, distance=2.)))
        assert status['renderer'] == renderer
        assert not np.array_equal(moved['scene_camera'], original['scene_camera'])
        for name in original:
            assert np.array_equal(frames[name], original[name]), 'readback buffers were reused'
        print(f'PASS: {profile}/{renderer}: four cameras, orbit, stable frame ownership; '
              f'{(time.perf_counter()-started)*200:.1f} ms per four-camera tick', flush=True)
    finally:
        if sim:
            await worker.run(sim.close)
        worker.close()


async def main():
    for renderer in ('filament', 'classic'):
        for profile in ('widowx', 'yam'):
            await check(profile, renderer)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
