import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# Local imports
try:
    from sort import Sort
except ImportError:
    print("Warning: sort.py dependencies not available. Using placeholder tracker.")
    class Sort:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, detections):
            return detections

app = FastAPI(title="Campus AI Dashboard")
app.mount("/static", StaticFiles(directory="."), name="static")

# Configuration
DEFAULT_VIDEO_SOURCE = "people.mp4"
YOLO_WEIGHTS = "yolov8n.pt"
CONF_THRESHOLD = 0.3
MASK_PATH = "mask.png"

# Classes
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


class DetectionPipeline:
    def __init__(self, name: str, video_source=DEFAULT_VIDEO_SOURCE):
        self.name = name
        self.video_source = video_source
        self.latest_frame_jpeg: Optional[bytes] = None
        self.metrics_lock = threading.Lock()
        self.current_fps: float = 0.0
        self.up_total: int = 0
        self.down_total: int = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.online = True
        self.start_time = datetime.now(timezone.utc)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        raise NotImplementedError


class PeopleCounterPipeline(DetectionPipeline):
    def _run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Warning: {self.video_source} not found. Running in offline mode.")
            self.online = False
            # Generate empty frames while offline
            while not self._stop.is_set():
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    self.latest_frame_jpeg = buf.tobytes()
                time.sleep(0.1)
            return
        
        self.online = True

        mask = cv2.imread(MASK_PATH)
        use_mask = mask is not None

        model = YOLO(YOLO_WEIGHTS)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        totalCountUp: Deque[int] = deque(maxlen=10000)
        totalCountDown: Deque[int] = deque(maxlen=10000)

        prev_time = time.time()
        while not self._stop.is_set():
            success, img = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            img_infer = img
            if use_mask and mask.shape[:2] == img.shape[:2]:
                img_infer = cv2.bitwise_and(img, mask)

            results = model(img_infer, stream=True)
            detections = np.empty((0, 5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls < len(CLASS_NAMES) and CLASS_NAMES[cls] == "person" and conf > CONF_THRESHOLD:
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

            resultsTracker = tracker.update(detections)

            for result in resultsTracker:
                x1, y1, x2, y2, track_id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                if 103 < cx < 296 and 161 - 15 < cy < 161 + 15:
                    if int(track_id) not in totalCountUp:
                        totalCountUp.append(int(track_id))

                if 527 < cx < 735 and 489 - 15 < cy < 489 + 15:
                    if int(track_id) not in totalCountDown:
                        totalCountDown.append(int(track_id))

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            with self.metrics_lock:
                self.up_total = len(totalCountUp)
                self.down_total = len(totalCountDown)
                self.current_fps = 0.9 * self.current_fps + 0.1 * fps if self.current_fps > 0 else fps

            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self.latest_frame_jpeg = buf.tobytes()

        cap.release()


class EmptyPipeline(DetectionPipeline):
    """Empty pipeline - generates blank frames, no actual processing"""
    def _run(self):
        self.online = False
        while not self._stop.is_set():
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self.latest_frame_jpeg = buf.tobytes()
            time.sleep(0.1)


PIPELINES: Dict[str, DetectionPipeline] = {
    "people": PeopleCounterPipeline("people", DEFAULT_VIDEO_SOURCE),
    "violence": EmptyPipeline("violence", ""),
    "riding": EmptyPipeline("riding", ""),
}
for p in PIPELINES.values():
    p.start()


@app.get("/")
async def index():
    model_names = list(PIPELINES.keys())
    try:
        with open("dashboard_complete_empty.html", "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("const MODELS = [];", "const MODELS = " + json.dumps(model_names) + ";")
        return HTMLResponse(html)
    except Exception:
        return HTMLResponse("<html><body>Error: dashboard_complete_empty.html not found</body></html>")


def get_pipeline(name: Optional[str]) -> DetectionPipeline:
    if not name:
        return PIPELINES["people"]
    return PIPELINES.get(name, PIPELINES["people"])


@app.get("/metrics/current")
async def metrics_current(model: Optional[str] = Query(default=None)):
    pipe = get_pipeline(model)
    with pipe.metrics_lock:
        return {
            "model": pipe.name,
            "ts": datetime.now(timezone.utc).isoformat(),
            "up_total": getattr(pipe, 'up_total', 0),
            "down_total": getattr(pipe, 'down_total', 0),
            "fps": round(pipe.current_fps, 2),
            "online": pipe.online,
        }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket, model: Optional[str] = None):
    await ws.accept()
    pipe = get_pipeline(model)
    try:
        while True:
            await asyncio.sleep(0.5)
            with pipe.metrics_lock:
                payload = {
                    "model": pipe.name,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "up_total": getattr(pipe, 'up_total', 0),
                    "down_total": getattr(pipe, 'down_total', 0),
                    "fps": round(pipe.current_fps, 2),
                    "online": pipe.online,
                }
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return


@app.get("/stream.mjpg")
async def stream_mjpeg(model: Optional[str] = Query(default=None)):
    pipe = get_pipeline(model)

    async def frame_generator():
        boundary = b"--frame"
        while True:
            await asyncio.sleep(0.05)
            frame = pipe.latest_frame_jpeg
            if frame is None:
                continue
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame.jpg")
async def frame_jpg(model: Optional[str] = Query(default=None)):
    pipe = get_pipeline(model)
    frame = pipe.latest_frame_jpeg
    if frame is None:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        ok, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buf.tobytes() if ok else b""
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    return Response(content=frame, media_type="image/jpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)