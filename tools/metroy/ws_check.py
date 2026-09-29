"""Start a MetroY capture through the running service and count what the browser's WebSocket receives."""
import asyncio
import json
import struct
import sys

import httpx
import websockets

B = "http://127.0.0.1:8765/api/capture"


async def main(secs: float, tracking: str, surface: str):
    httpx.post(f"{B}/connect", json={"driver": "metroy_usb", "settings": {"tracking": tracking, "surface": surface}},
               timeout=30).raise_for_status()
    httpx.post(f"{B}/start", timeout=30).raise_for_status()
    live = fused = 0
    live_pts = []
    async with websockets.connect("ws://127.0.0.1:8765/api/capture/stream", max_size=None) as ws:
        loop = asyncio.get_event_loop()
        end = loop.time() + secs
        while loop.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, bytes):
                _, _, n, flags = struct.unpack("<IIII", msg[:16])
                if flags & 8:
                    live += 1
                    live_pts.append(n)
                else:
                    fused += n
    st = httpx.get(f"{B}/status").json()
    httpx.post(f"{B}/stop", timeout=30)
    httpx.post(f"{B}/discard", timeout=30)
    print(f"{tracking}/{surface}: live frames received {live} ({live / secs:.1f}/s, ~{int(sum(live_pts) / max(1, live))} pts each), "
          f"fused points streamed {fused}; session tracking={st['tracking']} fused_frames={st['fused_frames']}")


asyncio.run(main(float(sys.argv[1]), sys.argv[2], sys.argv[3]))
