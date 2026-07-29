#!/usr/bin/env python3
# EPAJ_UI_PATCH_V1
# -*- coding: utf-8 -*-

"""
RK3588 + 双 IMX219 + YOLOv8 RKNN 实时检测

左相机：
    CAMERA1 -> /dev/video22

右相机：
    CAMERA2 -> /dev/video31

模型：
    /root/yolov8_rk3588/int8_best.rknn

浏览器：
    http://板子IP:8080/

说明：
1. 两路摄像头均以 1920x1080 NV12 连续采集。
2. 采集线程始终只保留最新帧，避免延迟不断累积。
3. 使用一个 RKNNLite 实例依次处理左右相机，稳定性优先。
4. 显示线程可以保持约30FPS，检测框使用最近一次推理结果。
5. 当前仅做双路目标检测，不做双目深度计算。
"""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = Path(
    "/root/yolov8_rk3588/int8_best.rknn"
)

DEFAULT_IMAGE_INFER_SCRIPT = Path(
    "/root/yolov8_rk3588/"
    "rknn_model_zoo-main/examples/yolov8/python/"
    "yolov8_rknn_infer.py"
)

DEFAULT_RESULT_DIR = Path(
    "/root/yolov8_rk3588/result_dual_camera"
)


def load_yolo_module(script_path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到原图片推理代码：{script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "yolov8_image_infer",
        str(script_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载推理模块：{script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_functions = [
        "letterbox",
        "postprocess_nine_outputs",
        "postprocess_single_output",
        "restore_boxes",
        "draw_detections",
        "initialize_rknn",
    ]

    missing = [
        name
        for name in required_functions
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "原图片推理代码缺少函数："
            + ", ".join(missing)
        )

    return module


def read_exact(stream, size):
    """从管道中读取完整一帧数据。"""
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            return None

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


class FrameSlot:
    """保存某一路摄像头的最新帧。"""

    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = -1
        self.timestamp = 0.0
        self.error = None

    def update(self, frame, timestamp):
        with self.lock:
            self.frame = frame
            self.sequence += 1
            self.timestamp = timestamp

    def get_latest(self):
        with self.lock:
            return (
                self.frame,
                self.sequence,
                self.timestamp,
            )

    def set_error(self, message):
        with self.lock:
            self.error = message

    def get_error(self):
        with self.lock:
            return self.error


class DetectionSlot:
    """保存某一路相机最近一次推理结果。"""

    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()

        self.boxes = None
        self.class_ids = None
        self.scores = None

        self.frame_sequence = -1
        self.timestamp = 0.0
        self.inference_ms = 0.0
        self.inference_fps = 0.0
        self.detection_count = 0

        self.last_update_time = None

    def update(
        self,
        boxes,
        class_ids,
        scores,
        frame_sequence,
        inference_ms,
    ):
        now = time.monotonic()

        with self.lock:
            if self.last_update_time is not None:
                delta = max(
                    now - self.last_update_time,
                    1e-6,
                )

                instant_fps = 1.0 / delta

                if self.inference_fps <= 0:
                    self.inference_fps = instant_fps
                else:
                    self.inference_fps = (
                        self.inference_fps * 0.85
                        + instant_fps * 0.15
                    )

            self.last_update_time = now

            self.boxes = boxes
            self.class_ids = class_ids
            self.scores = scores

            self.frame_sequence = frame_sequence
            self.timestamp = now
            self.inference_ms = inference_ms

            self.detection_count = (
                0 if boxes is None else len(boxes)
            )

    def get_latest(self):
        with self.lock:
            return {
                "boxes": self.boxes,
                "class_ids": self.class_ids,
                "scores": self.scores,
                "frame_sequence": self.frame_sequence,
                "timestamp": self.timestamp,
                "inference_ms": self.inference_ms,
                "inference_fps": self.inference_fps,
                "detection_count": self.detection_count,
            }


class V4L2CaptureWorker(threading.Thread):
    """通过 v4l2-ctl 连续读取一路 NV12 摄像头。"""

    def __init__(
        self,
        name,
        device,
        width,
        height,
        fps,
        warmup,
        buffers,
        frame_slot,
        stop_event,
    ):
        super().__init__(
            name=f"capture-{name}",
            daemon=True,
        )

        self.camera_name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.buffers = buffers
        self.frame_slot = frame_slot
        self.stop_event = stop_event

        self.frame_size = (
            width * height * 3 // 2
        )

        self.process = None
        self.log_path = Path(
            f"/tmp/{name.lower()}_camera_v4l2.log"
        )

    def stop(self):
        process = self.process

        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def run(self):
        command = [
            "v4l2-ctl",
            "-d",
            self.device,
            (
                "--set-fmt-video="
                f"width={self.width},"
                f"height={self.height},"
                "pixelformat=NV12"
            ),
            (
                "--set-parm="
                f"{self.fps}"
            ),
            f"--stream-mmap={self.buffers}",
            f"--stream-skip={self.warmup}",
            "--stream-poll",
            "--stream-to=-",
        ]

        print(
            f"[{self.camera_name}] 采集命令："
            + " ".join(command)
        )

        try:
            self.log_path.unlink(missing_ok=True)

            with self.log_path.open(
                "w",
                encoding="utf-8",
            ) as log_file:

                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    bufsize=0,
                )

                if self.process.stdout is None:
                    raise RuntimeError(
                        "无法读取 v4l2-ctl 输出"
                    )

                while not self.stop_event.is_set():
                    frame_bytes = read_exact(
                        self.process.stdout,
                        self.frame_size,
                    )

                    if frame_bytes is None:
                        if self.stop_event.is_set():
                            break

                        return_code = self.process.poll()

                        log_file.flush()

                        log_text = ""

                        if self.log_path.exists():
                            log_text = self.log_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )[-3000:]

                        raise RuntimeError(
                            "摄像头数据流中断；"
                            f"退出码={return_code}\n"
                            f"{log_text}"
                        )

                    raw = np.frombuffer(
                        frame_bytes,
                        dtype=np.uint8,
                    )

                    if raw.size != self.frame_size:
                        raise RuntimeError(
                            "NV12帧大小错误："
                            f"实际={raw.size}，"
                            f"预期={self.frame_size}"
                        )

                    nv12 = raw.reshape(
                        (
                            self.height * 3 // 2,
                            self.width,
                        )
                    )

                    bgr = cv2.cvtColor(
                        nv12,
                        cv2.COLOR_YUV2BGR_NV12,
                    )

                    self.frame_slot.update(
                        bgr,
                        time.monotonic(),
                    )

        except Exception as exc:
            if not self.stop_event.is_set():
                message = (
                    f"{self.camera_name} 采集失败：{exc}"
                )

                self.frame_slot.set_error(message)
                print(message, file=sys.stderr)

        finally:
            self.stop()


class InferenceWorker(threading.Thread):
    """
    一个RKNN实例轮流处理左右最新帧。

    这样可避免同一个RKNNLite对象被多个线程同时调用。
    """

    def __init__(
        self,
        yolo,
        rknn,
        left_frame_slot,
        right_frame_slot,
        left_detection_slot,
        right_detection_slot,
        stop_event,
    ):
        super().__init__(
            name="dual-yolo-inference",
            daemon=True,
        )

        self.yolo = yolo
        self.rknn = rknn

        self.sources = [
            (
                "LEFT",
                left_frame_slot,
                left_detection_slot,
            ),
            (
                "RIGHT",
                right_frame_slot,
                right_detection_slot,
            ),
        ]

        self.stop_event = stop_event
        self.error = None
        self.printed_output_shapes = False

    def infer_one(self, frame):
        input_image, ratio, padding = (
            self.yolo.letterbox(
                frame,
                self.yolo.IMAGE_SIZE,
                color=(0, 0, 0),
            )
        )

        input_image = cv2.cvtColor(
            input_image,
            cv2.COLOR_BGR2RGB,
        )

        input_tensor = np.expand_dims(
            input_image,
            axis=0,
        )

        start = time.perf_counter()

        outputs = self.rknn.inference(
            inputs=[input_tensor]
        )

        inference_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if outputs is None:
            raise RuntimeError(
                "RKNN inference 返回 None"
            )

        if not self.printed_output_shapes:
            print("\n========== RKNN 模型输出 ==========")
            print(f"输出数量：{len(outputs)}")

            for index, output in enumerate(outputs):
                print(
                    f"output[{index}]："
                    f"{np.asarray(output).shape}"
                )

            self.printed_output_shapes = True

        if len(outputs) == 9:
            boxes, class_ids, scores = (
                self.yolo.postprocess_nine_outputs(
                    outputs
                )
            )

        elif len(outputs) == 1:
            boxes, class_ids, scores = (
                self.yolo.postprocess_single_output(
                    outputs
                )
            )

        else:
            shapes = [
                np.asarray(output).shape
                for output in outputs
            ]

            raise RuntimeError(
                "不支持当前模型输出："
                f"数量={len(outputs)}，"
                f"形状={shapes}"
            )

        boxes = self.yolo.restore_boxes(
            boxes,
            ratio,
            padding,
            frame.shape,
        )

        return (
            boxes,
            class_ids,
            scores,
            inference_ms,
        )

    def run(self):
        last_sequences = {
            "LEFT": -1,
            "RIGHT": -1,
        }

        try:
            while not self.stop_event.is_set():
                processed = False

                for (
                    camera_name,
                    frame_slot,
                    detection_slot,
                ) in self.sources:

                    if self.stop_event.is_set():
                        break

                    frame, sequence, _ = (
                        frame_slot.get_latest()
                    )

                    if frame is None:
                        continue

                    if sequence == last_sequences[camera_name]:
                        continue

                    # 直接处理当前最新帧，中间旧帧自动丢弃
                    last_sequences[camera_name] = sequence

                    (
                        boxes,
                        class_ids,
                        scores,
                        inference_ms,
                    ) = self.infer_one(frame)

                    detection_slot.update(
                        boxes=boxes,
                        class_ids=class_ids,
                        scores=scores,
                        frame_sequence=sequence,
                        inference_ms=inference_ms,
                    )

                    processed = True

                if not processed:
                    time.sleep(0.002)

        except Exception as exc:
            self.error = f"YOLO推理线程失败：{exc}"
            print(self.error, file=sys.stderr)
            self.stop_event.set()


class OutputStore:
    """保存网页所需的最新JPEG和状态。"""

    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.version = 0
        self.status = {
            "state": "starting",
        }

    def update(self, jpeg, status):
        with self.condition:
            self.jpeg = jpeg
            self.status = status
            self.version += 1
            self.condition.notify_all()

    def wait_for_jpeg(
        self,
        previous_version,
        timeout=2.0,
    ):
        with self.condition:
            self.condition.wait_for(
                lambda: self.version != previous_version,
                timeout=timeout,
            )

            return self.jpeg, self.version

    def get_status(self):
        with self.condition:
            return dict(self.status)


def put_text_with_outline(
    image,
    text,
    position,
    scale=1.45,
):
    """
    在1080P原图上绘制大号文字。

    网页预览会把1920宽图像缩小到约640宽，
    因此原图文字必须放大约3倍，否则浏览器中会非常小。
    """
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )


def atomic_imwrite(path, image, quality=92):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.stem + ".tmp" + path.suffix
    )

    success = cv2.imwrite(
        str(temporary),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError(
            f"保存图片失败：{temporary}"
        )

    os.replace(temporary, path)


class PreviewWorker(threading.Thread):
    """以目标显示帧率生成左右合并画面。"""

    def __init__(
        self,
        yolo,
        left_frame_slot,
        right_frame_slot,
        left_detection_slot,
        right_detection_slot,
        output_store,
        result_dir,
        display_fps,
        preview_width,
        jpeg_quality,
        save_interval,
        stop_event,
    ):
        super().__init__(
            name="dual-preview",
            daemon=True,
        )

        self.yolo = yolo

        self.left_frame_slot = left_frame_slot
        self.right_frame_slot = right_frame_slot

        self.left_detection_slot = (
            left_detection_slot
        )
        self.right_detection_slot = (
            right_detection_slot
        )

        self.output_store = output_store
        self.result_dir = result_dir

        self.display_fps = display_fps
        self.preview_width = preview_width
        self.jpeg_quality = jpeg_quality
        self.save_interval = save_interval

        self.stop_event = stop_event
        self.error = None

        self.measured_display_fps = 0.0
        self.last_display_time = None

    def draw_camera(
        self,
        frame,
        frame_sequence,
        frame_timestamp,
        detection,
        camera_title,
        device_text,
    ):
        if frame is None:
            height = int(
                self.preview_width * 9 / 16
            )

            blank = np.zeros(
                (
                    height,
                    self.preview_width,
                    3,
                ),
                dtype=np.uint8,
            )

            put_text_with_outline(
                blank,
                f"{camera_title}: waiting...",
                (20, 40),
            )

            return blank, None

        result = frame.copy()

        result = self.yolo.draw_detections(
            result,
            detection["boxes"],
            detection["class_ids"],
            detection["scores"],
        )

        now = time.monotonic()

        capture_age_ms = (
            max(0.0, now - frame_timestamp)
            * 1000.0
        )

        if detection["timestamp"] > 0:
            detection_age_ms = (
                max(
                    0.0,
                    now - detection["timestamp"],
                )
                * 1000.0
            )
        else:
            detection_age_ms = -1.0

        lines = [
            f"{camera_title}  {device_text}",
            (
                f"Capture seq: {frame_sequence}  "
                f"age: {capture_age_ms:.0f} ms"
            ),
            (
                f"YOLO: {detection['inference_fps']:.1f} FPS  "
                f"{detection['inference_ms']:.1f} ms"
            ),
            (
                f"Detections: "
                f"{detection['detection_count']}  "
                f"result age: {detection_age_ms:.0f} ms"
            ),
        ]

        # 半透明黑色状态面板，提高复杂背景下的可读性
        panel = result.copy()
        panel_width = min(result.shape[1] - 1, 1050)
        panel_height = min(result.shape[0] - 1, 245)

        cv2.rectangle(
            panel,
            (0, 0),
            (panel_width, panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            panel,
            0.60,
            result,
            0.40,
            0,
            result,
        )

        y = 52

        for line in lines:
            put_text_with_outline(
                result,
                line,
                (12, y),
                scale=1.35,
            )
            y += 50

        original_height, original_width = (
            result.shape[:2]
        )

        preview_height = int(
            round(
                original_height
                * self.preview_width
                / original_width
            )
        )

        preview = cv2.resize(
            result,
            (
                self.preview_width,
                preview_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return preview, result

    def run(self):
        frame_interval = (
            1.0 / max(self.display_fps, 1.0)
        )

        next_time = time.monotonic()
        last_save_time = 0.0

        try:
            self.result_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            while not self.stop_event.is_set():
                now = time.monotonic()

                if now < next_time:
                    time.sleep(next_time - now)

                current_time = time.monotonic()
                next_time += frame_interval

                if (
                    current_time - next_time
                    > frame_interval
                ):
                    next_time = (
                        current_time + frame_interval
                    )

                left_frame, left_sequence, left_ts = (
                    self.left_frame_slot.get_latest()
                )

                right_frame, right_sequence, right_ts = (
                    self.right_frame_slot.get_latest()
                )

                left_detection = (
                    self.left_detection_slot.get_latest()
                )

                right_detection = (
                    self.right_detection_slot.get_latest()
                )

                (
                    left_preview,
                    left_full,
                ) = self.draw_camera(
                    frame=left_frame,
                    frame_sequence=left_sequence,
                    frame_timestamp=left_ts,
                    detection=left_detection,
                    camera_title="LEFT / CAMERA1",
                    device_text="/dev/video22",
                )

                (
                    right_preview,
                    right_full,
                ) = self.draw_camera(
                    frame=right_frame,
                    frame_sequence=right_sequence,
                    frame_timestamp=right_ts,
                    detection=right_detection,
                    camera_title="RIGHT / CAMERA2",
                    device_text="/dev/video31",
                )

                target_height = max(
                    left_preview.shape[0],
                    right_preview.shape[0],
                )

                def pad_height(image):
                    difference = (
                        target_height - image.shape[0]
                    )

                    if difference <= 0:
                        return image

                    return cv2.copyMakeBorder(
                        image,
                        0,
                        difference,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

                left_preview = pad_height(
                    left_preview
                )
                right_preview = pad_height(
                    right_preview
                )

                combined = np.hstack(
                    (
                        left_preview,
                        right_preview,
                    )
                )

                if self.last_display_time is not None:
                    delta = max(
                        current_time
                        - self.last_display_time,
                        1e-6,
                    )

                    instant_fps = 1.0 / delta

                    if self.measured_display_fps <= 0:
                        self.measured_display_fps = (
                            instant_fps
                        )
                    else:
                        self.measured_display_fps = (
                            self.measured_display_fps
                            * 0.9
                            + instant_fps
                            * 0.1
                        )

                self.last_display_time = current_time

                success, encoded = cv2.imencode(
                    ".jpg",
                    combined,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        self.jpeg_quality,
                    ],
                )

                if not success:
                    raise RuntimeError(
                        "实时JPEG编码失败"
                    )

                status = {
                    "state": "running",
                    "time": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "display_fps": round(
                        self.measured_display_fps,
                        2,
                    ),
                    "left": {
                        "device": "/dev/video22",
                        "capture_sequence": (
                            left_sequence
                        ),
                        "inference_fps": round(
                            left_detection[
                                "inference_fps"
                            ],
                            2,
                        ),
                        "inference_ms": round(
                            left_detection[
                                "inference_ms"
                            ],
                            2,
                        ),
                        "detections": (
                            left_detection[
                                "detection_count"
                            ]
                        ),
                    },
                    "right": {
                        "device": "/dev/video31",
                        "capture_sequence": (
                            right_sequence
                        ),
                        "inference_fps": round(
                            right_detection[
                                "inference_fps"
                            ],
                            2,
                        ),
                        "inference_ms": round(
                            right_detection[
                                "inference_ms"
                            ],
                            2,
                        ),
                        "detections": (
                            right_detection[
                                "detection_count"
                            ]
                        ),
                    },
                }

                self.output_store.update(
                    encoded.tobytes(),
                    status,
                )

                if (
                    self.save_interval > 0
                    and (
                        current_time
                        - last_save_time
                        >= self.save_interval
                    )
                ):
                    if left_full is not None:
                        atomic_imwrite(
                            self.result_dir
                            / "latest_left.jpg",
                            left_full,
                        )

                    if right_full is not None:
                        atomic_imwrite(
                            self.result_dir
                            / "latest_right.jpg",
                            right_full,
                        )

                    atomic_imwrite(
                        self.result_dir
                        / "latest_pair.jpg",
                        combined,
                    )

                    last_save_time = current_time

        except Exception as exc:
            self.error = f"预览线程失败：{exc}"
            print(self.error, file=sys.stderr)
            self.stop_event.set()


def create_http_handler(output_store):
    class CameraHandler(BaseHTTPRequestHandler):
        server_version = (
            "RK3588-Dual-YOLO/1.0"
        )

        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_index()

            elif self.path.startswith(
                "/stream.mjpg"
            ):
                self.send_stream()

            elif self.path.startswith(
                "/snapshot.jpg"
            ):
                self.send_snapshot()

            elif self.path.startswith(
                "/status.json"
            ):
                self.send_status()

            else:
                self.send_error(
                    404,
                    "Not Found",
                )

        def send_index(self):
            html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RK3588 双摄 YOLO 实时检测</title>
<style>
body {
    margin: 0;
    background: #101010;
    color: #eeeeee;
    font-family: Arial, sans-serif;
    text-align: center;
}
h2 {
    margin: 16px 0 8px;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 1px;
}
#status {
    margin: 10px;
    font-family: Consolas, monospace;
    font-size: 21px;
    font-weight: 700;
    line-height: 1.55;
    white-space: pre-wrap;
}
img {
    max-width: 98vw;
    max-height: 82vh;
    border: 1px solid #555;
}
a {
    color: #74b9ff;
}
</style>
</head>
<body>
<h2>RK3588 双 IMX219 YOLOv8 实时检测</h2>
<div id="status">正在连接……</div>
<img src="/stream.mjpg" alt="双摄实时检测">
<p>
<a href="/snapshot.jpg" target="_blank">
打开当前合并截图
</a>
</p>
<script>
async function updateStatus() {
    try {
        const response = await fetch(
            '/status.json',
            {cache: 'no-store'}
        );
        const data = await response.json();

        const left = data.left || {};
        const right = data.right || {};

        document.getElementById('status').textContent =
            `显示：${data.display_fps ?? '-'} FPS | ` +
            `左：${left.inference_fps ?? '-'} FPS / ` +
            `${left.detections ?? '-'} 个 | ` +
            `右：${right.inference_fps ?? '-'} FPS / ` +
            `${right.detections ?? '-'} 个`;
    } catch (error) {
        document.getElementById('status').textContent =
            '状态连接失败';
    }
}

setInterval(updateStatus, 1000);
updateStatus();
</script>
</body>
</html>
"""

            body = html.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)

        def send_stream(self):
            self.send_response(200)
            self.send_header(
                "Cache-Control",
                "no-cache, no-store",
            )
            self.send_header(
                "Pragma",
                "no-cache",
            )
            self.send_header(
                "Connection",
                "close",
            )
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace;"
                " boundary=frame",
            )
            self.end_headers()

            previous_version = -1

            try:
                while True:
                    jpeg, version = (
                        output_store.wait_for_jpeg(
                            previous_version,
                            timeout=2.0,
                        )
                    )

                    if jpeg is None:
                        continue

                    if version == previous_version:
                        continue

                    previous_version = version

                    self.wfile.write(
                        b"--frame\r\n"
                    )
                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )
                    self.wfile.write(
                        (
                            f"Content-Length: "
                            f"{len(jpeg)}\r\n\r\n"
                        ).encode("ascii")
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):
                return

        def send_snapshot(self):
            jpeg, _ = output_store.wait_for_jpeg(
                -1,
                timeout=2.0,
            )

            if jpeg is None:
                self.send_error(
                    503,
                    "No frame available",
                )
                return

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "image/jpeg",
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.send_header(
                "Content-Length",
                str(len(jpeg)),
            )
            self.end_headers()
            self.wfile.write(jpeg)

        def send_status(self):
            body = json.dumps(
                output_store.get_status(),
                ensure_ascii=False,
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)

    return CameraHandler


class DualCameraHTTPServer(
    ThreadingHTTPServer
):
    allow_reuse_address = True
    daemon_threads = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "RK3588 双 IMX219 YOLOv8 RKNN "
            "实时检测"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--infer-script",
        type=Path,
        default=DEFAULT_IMAGE_INFER_SCRIPT,
    )

    parser.add_argument(
        "--left",
        default="/dev/video22",
        help="左相机 CAMERA1",
    )

    parser.add_argument(
        "--right",
        default="/dev/video31",
        help="右相机 CAMERA2",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
    )

    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--buffers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--preview-width",
        type=int,
        default=640,
        help=(
            "网页中每一路画面的宽度；"
            "采集和推理仍使用1920x1080"
        ),
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )

    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
        help="每隔多少秒覆盖保存最新结果",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )

    return parser.parse_args()


def validate_environment(args):
    if not args.model.exists():
        raise FileNotFoundError(
            f"模型不存在：{args.model}"
        )

    if not args.infer_script.exists():
        raise FileNotFoundError(
            "图片推理代码不存在："
            f"{args.infer_script}"
        )

    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError(
            "找不到 v4l2-ctl"
        )

    for name, device in [
        ("左相机", args.left),
        ("右相机", args.right),
    ]:
        if not Path(device).exists():
            raise FileNotFoundError(
                f"{name}设备不存在：{device}"
            )

    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            "分辨率必须大于0"
        )

    if not 0 < args.conf < 1:
        raise ValueError(
            "conf必须位于0到1之间"
        )

    if not 0 < args.nms < 1:
        raise ValueError(
            "nms必须位于0到1之间"
        )

    args.jpeg_quality = max(
        20,
        min(100, args.jpeg_quality),
    )


def get_board_ips():
    try:
        output = subprocess.check_output(
            ["hostname", "-I"],
            text=True,
            timeout=2,
        )

        return [
            item
            for item in output.strip().split()
            if item
        ]

    except Exception:
        return []


def main():
    args = parse_args()
    validate_environment(args)

    print("========== 双摄运行配置 ==========")
    print(f"Python：{sys.executable}")
    print(f"OpenCV：{cv2.__version__}")
    print(f"模型：{args.model}")
    print(f"左相机：{args.left}")
    print(f"右相机：{args.right}")
    print(
        f"采集分辨率："
        f"{args.width}x{args.height}"
    )
    print(f"采集目标帧率：{args.camera_fps}")
    print(f"模型输入：{args.input_size}x{args.input_size}")
    print(f"类别：100- O-O-S")
    print(f"结果目录：{args.result_dir}")

    yolo = load_yolo_module(
        args.infer_script
    )

    yolo.MODEL_PATH = args.model
    yolo.CLASS_NAMES = ["100- O-O-S"]
    yolo.IMAGE_SIZE = args.input_size
    yolo.CONF_THRESHOLD = args.conf
    yolo.NMS_THRESHOLD = args.nms

    print("\n========== 初始化 RKNN NPU ==========")
    rknn = yolo.initialize_rknn()
    print("RKNN NPU 初始化成功")

    stop_event = threading.Event()

    left_frame_slot = FrameSlot("LEFT")
    right_frame_slot = FrameSlot("RIGHT")

    left_detection_slot = DetectionSlot("LEFT")
    right_detection_slot = DetectionSlot("RIGHT")

    output_store = OutputStore()

    left_capture = V4L2CaptureWorker(
        name="LEFT",
        device=args.left,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        warmup=args.warmup,
        buffers=args.buffers,
        frame_slot=left_frame_slot,
        stop_event=stop_event,
    )

    right_capture = V4L2CaptureWorker(
        name="RIGHT",
        device=args.right,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        warmup=args.warmup,
        buffers=args.buffers,
        frame_slot=right_frame_slot,
        stop_event=stop_event,
    )

    inference_worker = InferenceWorker(
        yolo=yolo,
        rknn=rknn,
        left_frame_slot=left_frame_slot,
        right_frame_slot=right_frame_slot,
        left_detection_slot=left_detection_slot,
        right_detection_slot=right_detection_slot,
        stop_event=stop_event,
    )

    preview_worker = PreviewWorker(
        yolo=yolo,
        left_frame_slot=left_frame_slot,
        right_frame_slot=right_frame_slot,
        left_detection_slot=left_detection_slot,
        right_detection_slot=right_detection_slot,
        output_store=output_store,
        result_dir=args.result_dir,
        display_fps=args.display_fps,
        preview_width=args.preview_width,
        jpeg_quality=args.jpeg_quality,
        save_interval=args.save_interval,
        stop_event=stop_event,
    )

    handler = create_http_handler(
        output_store
    )

    server = DualCameraHTTPServer(
        (args.host, args.port),
        handler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="dual-camera-http",
        daemon=True,
    )

    def request_stop(signum=None, frame=None):
        stop_event.set()

    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        left_capture.start()
        right_capture.start()
        inference_worker.start()
        preview_worker.start()
        server_thread.start()

        print("\n========== 浏览器访问地址 ==========")

        board_ips = get_board_ips()

        if board_ips:
            for ip_address in board_ips:
                print(
                    f"http://{ip_address}:{args.port}/"
                )
        else:
            print(
                f"http://板子IP:{args.port}/"
            )

        print("\n按 Ctrl+C 停止程序。")

        while not stop_event.wait(1.0):
            errors = [
                left_frame_slot.get_error(),
                right_frame_slot.get_error(),
                inference_worker.error,
                preview_worker.error,
            ]

            errors = [
                item
                for item in errors
                if item
            ]

            if errors:
                raise RuntimeError(
                    "\n".join(errors)
                )

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        print("\n正在停止双摄程序……")

        stop_event.set()

        left_capture.stop()
        right_capture.stop()

        server.shutdown()
        server.server_close()

        left_capture.join(timeout=3)
        right_capture.join(timeout=3)
        inference_worker.join(timeout=3)
        preview_worker.join(timeout=3)

        rknn.release()

        print("双摄采集已停止")
        print("RKNN NPU资源已释放")
        print(f"结果目录：{args.result_dir}")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"\n程序运行失败：{exc}",
            file=sys.stderr,
        )

        raise SystemExit(1)
