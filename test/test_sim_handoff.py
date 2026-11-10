"""Live transport check against a disposable robot-sim + sim-policy session.

SIM_TEST_URL=http://127.0.0.1:8080 SIM_WEB_TOKEN=... python -m test.test_sim_handoff
This takes control, moves an arm, and resets the scene. It does not start servers.
"""
import asyncio
import os
import time

import aiohttp


async def main():
    url = os.environ['SIM_TEST_URL'].rstrip('/')
    code = os.environ['SIM_WEB_TOKEN']
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:
        async with session.post(url + '/api/login', json={'code':code}) as response:
            assert response.status == 200, await response.text()
        async with session.ws_connect(url + '/ws') as ws:
            ready = await ws.receive_json(timeout=15)
            assert ready['type'] == 'ready', ready
            me = ready['identity']
            waist = 'left_joint_1.pos' if ready['profile'] == 'yam' else 'left_joint_0.pos'
            assert set(ready['setups']) == {'workbench'}
            latest = None

            async def until(predicate, timeout=10):
                nonlocal latest
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    message = await ws.receive_json(timeout=max(.1, deadline-time.monotonic()))
                    assert message['type'] != 'error', message
                    if message['type'] == 'state':
                        latest = message
                        if predicate(message):
                            return message
                raise AssertionError('timed out waiting for state')

            async def drive(values, duration):
                end = time.monotonic() + duration
                while time.monotonic() < end:
                    await ws.send_json({'type':'action', 'values':values, 'anchor':latest['anchor'],
                                        'generation':latest['generation']})
                    await until(lambda x: x['fresh'])

            await until(lambda x: x['fresh'] and 'policy-sim-demo' in x['operators'])
            assert latest['active'] is None, 'peers must start idle'
            assert len(latest['state']) == 14
            print('PASS: synchronized four-camera observations; peers start idle')
            await ws.send_json({'type':'control', 'identity':me})
            await until(lambda x: x['active'] == me and x['fresh'])
            target = latest['state'] | {waist:.35}
            await drive(target, 1.5)
            assert abs(latest['state'][waist] - .35) < .04, latest
            print('PASS: browser bridge sends anchored Portal actions that move the arm')

            await ws.send_json({'type':'control', 'identity':'policy-sim-demo'})
            await until(lambda x: x['active'] == 'policy-sim-demo')
            start = latest['state'][waist]
            await drive(target | {waist:-1.}, 1.5)  # inactive browser must be rejected
            assert latest['active'] == 'policy-sim-demo', latest
            assert latest['state'][waist] > start + .05, latest
            print('PASS: policy drives; inactive browser actions cannot override it')

            await ws.send_json({'type':'control', 'identity':me})
            await until(lambda x: x['active'] == me)
            await drive(latest['state'] | {waist:.1}, 1.5)
            assert abs(latest['state'][waist] - .1) < .04
            print('PASS: human takeover interrupts the policy')
            # Stop all commands but leave the browser socket connected.
            await until(lambda x: x['active'] is None, timeout=3)
            print('PASS: command timeout releases control')

            old = latest.copy()
            await ws.send_json({'type':'motion', 'command':'reset'})
            await until(lambda x: x.get('episode',0) > old['episode'] and not x['busy'] and x['fresh'])
            assert latest['active'] is None
            await ws.send_json({'type':'action', 'values':target, 'anchor':old['anchor'], 'generation':old['generation']})
            await until(lambda x: x['sim_time'] > .5)
            assert abs(latest['state'][waist]) < .03
            print('PASS: reset restores scene, releases gate, rejects old command')
            await ws.send_json({'type':'control', 'identity':'policy-sim-demo'})
            await until(lambda x: x['active'] == 'policy-sim-demo')
            old_episode = latest['episode']
            await ws.send_json({'type':'motion', 'command':'reset:workbench'})
            await until(lambda x: x.get('episode', 0) > old_episode and not x['busy'] and x['fresh'])
            assert latest['active'] is None and latest['setup'] == 'workbench', latest
            print('PASS: workbench reset releases policy control')
            await ws.send_json({'type':'control', 'identity':me})
            await until(lambda x: x['active'] == me)
            await drive(latest['state'], .3)
        async with session.ws_connect(url + '/ws') as ws2:
            ready2 = await ws2.receive_json(timeout=15)
            assert ready2['identity'] != me
            for _ in range(30):
                state = await ws2.receive_json(timeout=5)
                if state.get('fresh') and state.get('active') is None:
                    break
            else:
                raise AssertionError('browser disconnect did not release control')
        print('PASS: browser disconnect releases its operator; reconnect does not reclaim')


if __name__ == '__main__':
    asyncio.run(main())
