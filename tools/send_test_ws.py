import asyncio, websockets, json, base64, time
import cv2

WS_URL = 'ws://127.0.0.1:9000/ws/PerimetralesMultiCam'
FRAMES = 50
DELAY = 0.05  # seconds between sends

async def main():
    async with websockets.connect(WS_URL) as ws:
        cap = cv2.VideoCapture(0)
        sent = 0
        while sent < FRAMES:
            ret, frame = cap.read()
            if not ret:
                # generate dummy frame
                import numpy as np
                frame = (255 * np.random.rand(480,640,3)).astype('uint8')
            _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64 = base64.b64encode(enc.tobytes()).decode('utf-8')
            payload = {'data': {'image': f'data:image/jpeg;base64,{b64}', 'roi_activate': False, 'camera_id': 1}}
            t0 = time.time()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            t1 = time.time()
            print(f'response len {len(resp)} time {t1-t0:.3f}s')
            sent += 1
            await asyncio.sleep(DELAY)
        cap.release()

if __name__ == '__main__':
    asyncio.run(main())
