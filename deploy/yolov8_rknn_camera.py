#!/usr/bin/env python3
"""
TL3588F-EVM / RK3588 + IMX219 + YOLOv8 RKNN 实时检测

采集链路：
    /dev/video-camera0
    -> v4l2-ctl 连续输出 NV12
    -> OpenCV NV12 转 BGR
    -> YOLOv8 RKNN NPU 推理
    -> 浏览器 MJPEG 实时显示

默认模型：
    /root/yolov8_rk3588/int8_best.rknn

默认结果：
    /root/yolov8_rk3588/result_camera/latest.jpg

浏览器：
    http://板子IP:8080/
"""

from __future__ import annotations

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
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL_PATH = Path(
    "/root/yolov8_rk3588/int8_best.rknn"
)

IMAGE_INFER_SCRIPT = Path(
    "/root/yolov8_rk3588/"
    "rknn_model_zoo-main/examples/yolov8/python/"
    "yolov8_rknn_infer.py"
)

DEFAULT_DEVICE = "/dev/video-camera0"

DEFAULT_RESULT_DIR = Path(
    "/root/yolov8_rk3588/result_camera"
)

V4L2_LOG_PATH = Path(
    "/tmp/yolov8_camera_v4l2.log"
)


def load_image_infer_module():
    """
    动态加载原来的图片推理代码。

    这样可以直接复用：
    - letterbox
    - RKNN初始化
    - 9输出后处理
    - 单输出后处理
    - NMS
    - 坐标恢复
    - 检测框绘制
    """
    if not IMAGE_INFER_SCRIPT.exists():
        raise FileNotFoundError(
            f"找不到原图片推理代码：{IMAGE_INFER_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "yolov8_image_infer",
        str(IMAGE_INFER_SCRIPT),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载Python模块：{IMAGE_INFER_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class LatestFrameStore:
    """采集线程和推理线程之间只传递最新一帧。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.error: str | None = None

    def update(self, frame: np.ndarray) -> None:
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()

    def set_error(self, message: str) -> None:
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def wait_next(
        self,
        last_sequence: int,
        timeout: float = 5.0,
    ) -> tuple[np.ndarray | None, int]:
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.sequence != last_sequence
                    or self.error is not None
                ),
                timeout=timeout,
            )

            if self.error:
                raise RuntimeError(self.error)

            if (
                self.frame is None
                or self.sequence == last_sequence
            ):
                return None, last_sequence

            return self.frame.copy(), self.sequence


class ResultStore:
    """保存浏览器需要的最新JPEG和状态信息。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.version = 0
        self.status: dict[str, Any] = {
            "state": "starting",
        }

    def update(
        self,
        jpeg: bytes,
        status: dict[str, Any],
    ) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.status = status
            self.version += 1
            self.condition.notify_all()

    def wait_jpeg(
        self,
        last_version: int,
        timeout: float = 2.0,
    ) -> tuple[bytes | None, int]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.version != last_version,
                timeout=timeout,
            )

            return self.jpeg, self.version

    def get_status(self) -> dict[str, Any]:
        with self.condition:
            return dict(self.status)


def read_exact(
    stream,
    byte_count: int,
) -> bytes | None:
    """
    从连续NV12字节流中读取完整一帧。

    pipe/read并不保证一次返回完整帧，因此必须循环读取。
    """
    chunks: list[bytes] = []
    remaining = byte_count

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            return None

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


class V4L2CaptureWorker(threading.Thread):
    """
    使用已经验证成功的 v4l2-ctl 路径连续采集NV12。

    这样不依赖OpenCV VideoCapture是否支持
    V4L2 multiplanar NV12。
    """

    def __init__(
        self,
        frame_store: LatestFrameStore,
        device: str,
        width: int,
        height: int,
        warmup: int,
        buffers: int,
    ) -> None:
        super().__init__(
            name="v4l2-capture",
            daemon=True,
        )

        self.frame_store = frame_store
        self.device = device
        self.width = width
        self.height = height
        self.warmup = warmup
        self.buffers = buffers

        self.frame_size = (
            self.width * self.height * 3 // 2
        )

        self.stop_event = threading.Event()
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def stop(self) -> None:
        self.stop_event.set()

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            self.process.terminate()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def run(self) -> None:
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
            f"--stream-mmap={self.buffers}",
            f"--stream-skip={self.warmup}",
            "--stream-poll",
            "--stream-to=-",
        ]

        try:
            V4L2_LOG_PATH.unlink(missing_ok=True)

            self.log_file = V4L2_LOG_PATH.open(
                "w",
                encoding="utf-8",
            )

            print("\n========== 摄像头采集命令 ==========")
            print(" ".join(command))
            print(f"单帧大小：{self.frame_size} 字节")

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=self.log_file,
                bufsize=0,
            )

            if self.process.stdout is None:
                raise RuntimeError(
                    "无法读取v4l2-ctl标准输出"
                )

            while not self.stop_event.is_set():
                frame_bytes = read_exact(
                    self.process.stdout,
                    self.frame_size,
                )

                if frame_bytes is None:
                    return_code = self.process.poll()

                    error_text = ""

                    if self.log_file:
                        self.log_file.flush()

                    if V4L2_LOG_PATH.exists():
                        error_text = V4L2_LOG_PATH.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )[-4000:]

                    raise RuntimeError(
                        "摄像头字节流中断。\n"
                        f"v4l2-ctl退出码：{return_code}\n"
                        f"日志：\n{error_text}"
                    )

                raw = np.frombuffer(
                    frame_bytes,
                    dtype=np.uint8,
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

                self.frame_store.update(bgr)

        except Exception as exc:
            if not self.stop_event.is_set():
                self.frame_store.set_error(
                    f"摄像头采集失败：{exc}"
                )

        finally:
            self.stop()

            if self.log_file:
                self.log_file.close()


def create_http_handler(
    result_store: ResultStore,
):
    class CameraHTTPHandler(BaseHTTPRequestHandler):
        server_version = "RK3588-YOLO-Camera/1.0"

        def log_message(
            self,
            format_string: str,
            *args,
        ) -> None:
            # 避免每一帧都打印HTTP日志
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_index()

            elif self.path.startswith("/stream.mjpg"):
                self.send_stream()

            elif self.path.startswith("/snapshot.jpg"):
                self.send_snapshot()

            elif self.path.startswith("/status.json"):
                self.send_status()

            else:
                self.send_error(404, "Not Found")

        def send_index(self) -> None:
            html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RK3588 YOLO 实时检测</title>
<style>
body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    text-align: center;
}
h2 {
    margin: 14px 0 8px;
}
#status {
    margin-bottom: 10px;
    font-family: monospace;
}
img {
    max-width: 96vw;
    max-height: 84vh;
    border: 1px solid #555;
}
a {
    color: #70b7ff;
}
</style>
</head>
<body>
<h2>RK3588 YOLOv8 实时检测</h2>
<div id="status">正在连接……</div>
<img src="/stream.mjpg" alt="实时检测画面">
<p><a href="/snapshot.jpg" target="_blank">打开当前检测截图</a></p>
<script>
async function updateStatus() {
    try {
        const response = await fetch('/status.json', {
            cache: 'no-store'
        });
        const data = await response.json();

        document.getElementById('status').textContent =
            `FPS=${data.fps ?? '-'} | ` +
            `NPU=${data.npu_ms ?? '-'} ms | ` +
            `检测数=${data.detections ?? '-'} | ` +
            `帧=${data.capture_sequence ?? '-'}`;
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

        def send_stream(self) -> None:
            self.send_response(200)
            self.send_header(
                "Cache-Control",
                "no-cache, no-store, must-revalidate",
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
                "multipart/x-mixed-replace; "
                "boundary=frame",
            )
            self.end_headers()

            last_version = -1

            try:
                while True:
                    jpeg, version = (
                        result_store.wait_jpeg(
                            last_version,
                            timeout=2.0,
                        )
                    )

                    if jpeg is None:
                        continue

                    if version == last_version:
                        continue

                    last_version = version

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )
                    self.wfile.write(
                        (
                            f"Content-Length: {len(jpeg)}"
                            "\r\n\r\n"
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

        def send_snapshot(self) -> None:
            jpeg, _ = result_store.wait_jpeg(
                -1,
                timeout=2.0,
            )

            if jpeg is None:
                self.send_error(
                    503,
                    "No camera frame available",
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

        def send_status(self) -> None:
            body = json.dumps(
                result_store.get_status(),
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

    return CameraHTTPHandler


def start_http_server(
    result_store: ResultStore,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    handler = create_http_handler(result_store)

    server = ThreadingHTTPServer(
        (host, port),
        handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        name="mjpeg-http",
        daemon=True,
    )
    thread.start()

    return server


def get_board_ips() -> list[str]:
    try:
        text = subprocess.check_output(
            ["hostname", "-I"],
            text=True,
            timeout=2,
        )

        return [
            item
            for item in text.strip().split()
            if item
        ]

    except Exception:
        return []


def draw_runtime_information(
    image: np.ndarray,
    fps: float,
    npu_ms: float,
    detection_count: int,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"NPU: {npu_ms:.1f} ms",
        f"Detections: {detection_count}",
    ]

    y = 30

    for text in lines:
        cv2.putText(
            image,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 30


def save_latest_image(
    image: np.ndarray,
    result_dir: Path,
) -> Path:
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        result_dir / "latest.tmp.jpg"
    )

    output_path = (
        result_dir / "latest.jpg"
    )

    success = cv2.imwrite(
        str(temporary_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if not success:
        raise RuntimeError(
            f"结果保存失败：{temporary_path}"
        )

    os.replace(
        temporary_path,
        output_path,
    )

    return output_path


def run_inference_loop(
    args: argparse.Namespace,
    yolo,
    rknn,
    frame_store: LatestFrameStore,
    result_store: ResultStore,
    stop_event: threading.Event,
) -> None:
    last_sequence = -1
    last_result_time: float | None = None
    smoothed_fps = 0.0
    first_output = True
    last_save_time = 0.0

    while not stop_event.is_set():
        original_image, sequence = (
            frame_store.wait_next(
                last_sequence,
                timeout=5.0,
            )
        )

        if original_image is None:
            continue

        last_sequence = sequence

        input_image, ratio, padding = (
            yolo.letterbox(
                original_image,
                yolo.IMAGE_SIZE,
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

        inference_start = time.perf_counter()

        outputs = rknn.inference(
            inputs=[input_tensor]
        )

        npu_ms = (
            time.perf_counter()
            - inference_start
        ) * 1000.0

        if outputs is None:
            raise RuntimeError(
                "RKNN推理失败：outputs为None"
            )

        if first_output:
            print("\n========== 模型输出 ==========")
            print(f"输出数量：{len(outputs)}")

            for index, output in enumerate(outputs):
                print(
                    f"output[{index}]："
                    f"{np.asarray(output).shape}"
                )

            first_output = False

        if len(outputs) == 9:
            boxes, class_ids, scores = (
                yolo.postprocess_nine_outputs(
                    outputs
                )
            )

        elif len(outputs) == 1:
            boxes, class_ids, scores = (
                yolo.postprocess_single_output(
                    outputs
                )
            )

        else:
            output_shapes = [
                np.asarray(output).shape
                for output in outputs
            ]

            raise RuntimeError(
                "不支持当前模型输出结构："
                f"输出数量={len(outputs)}，"
                f"形状={output_shapes}"
            )

        boxes = yolo.restore_boxes(
            boxes,
            ratio,
            padding,
            original_image.shape,
        )

        result_image = yolo.draw_detections(
            original_image.copy(),
            boxes,
            class_ids,
            scores,
        )

        detection_count = (
            0
            if boxes is None
            else len(boxes)
        )

        current_time = time.perf_counter()

        if last_result_time is not None:
            instantaneous_fps = (
                1.0
                / max(
                    current_time - last_result_time,
                    1e-6,
                )
            )

            if smoothed_fps <= 0:
                smoothed_fps = instantaneous_fps
            else:
                smoothed_fps = (
                    smoothed_fps * 0.9
                    + instantaneous_fps * 0.1
                )

        last_result_time = current_time

        draw_runtime_information(
            result_image,
            smoothed_fps,
            npu_ms,
            detection_count,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            result_image,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )

        if not success:
            raise RuntimeError(
                "OpenCV实时JPEG编码失败"
            )

        status = {
            "state": "running",
            "fps": round(smoothed_fps, 2),
            "npu_ms": round(npu_ms, 2),
            "detections": int(detection_count),
            "capture_sequence": int(sequence),
            "device": args.device,
            "model": str(args.model),
            "resolution": (
                f"{args.width}x{args.height}"
            ),
            "classes": list(yolo.CLASS_NAMES),
            "time": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        result_store.update(
            encoded.tobytes(),
            status,
        )

        now = time.monotonic()

        if (
            args.save_interval > 0
            and (
                now - last_save_time
                >= args.save_interval
            )
        ):
            output_path = save_latest_image(
                result_image,
                args.result_dir,
            )

            last_save_time = now

            print(
                "\r"
                f"帧={sequence:<8d} "
                f"FPS={smoothed_fps:>5.1f} "
                f"NPU={npu_ms:>6.1f}ms "
                f"检测={detection_count:<3d} "
                f"保存={output_path}",
                end="",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RK3588 + IMX219 + YOLOv8 RKNN "
            "摄像头实时检测"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="RKNN模型路径",
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="摄像头设备节点",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="采集宽度，默认1920",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="采集高度，默认1080",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="启动时丢弃的曝光稳定帧数",
    )

    parser.add_argument(
        "--buffers",
        type=int,
        default=4,
        help="V4L2 mmap缓冲区数量",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="模型输入尺寸，默认640",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值",
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS IoU阈值",
    )

    parser.add_argument(
        "--classes",
        default="",
        help=(
            "类别名称，使用英文逗号分隔；"
            "不填写则使用原图片代码中的CLASS_NAMES"
        ),
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="网页监听地址",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="网页端口，默认8080",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        help="实时网页JPEG质量，默认82",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="最新结果保存目录",
    )

    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
        help=(
            "latest.jpg保存间隔，单位秒；"
            "设置0表示不保存"
        ),
    )

    parser.add_argument(
        "--no-web",
        action="store_true",
        help="关闭浏览器实时画面，仅运行推理和保存结果",
    )

    return parser.parse_args()


def validate_environment(
    args: argparse.Namespace,
) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"模型不存在：{args.model}"
        )

    if not IMAGE_INFER_SCRIPT.exists():
        raise FileNotFoundError(
            f"图片推理代码不存在：{IMAGE_INFER_SCRIPT}"
        )

    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError(
            "系统中找不到v4l2-ctl"
        )

    device_path = Path(args.device)

    if not device_path.exists():
        raise FileNotFoundError(
            f"摄像头设备不存在：{args.device}"
        )

    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            "width和height必须大于0"
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
        10,
        min(100, args.jpeg_quality),
    )


def main() -> int:
    args = parse_args()
    validate_environment(args)

    yolo = load_image_infer_module()

    # 覆盖原图片推理代码中的运行参数
    yolo.MODEL_PATH = args.model
    yolo.IMAGE_SIZE = args.input_size
    yolo.CONF_THRESHOLD = args.conf
    yolo.NMS_THRESHOLD = args.nms

    if args.classes.strip():
        yolo.CLASS_NAMES = [
            item.strip()
            for item in args.classes.split(",")
            if item.strip()
        ]

    print("========== 运行配置 ==========")
    print(f"Python：{sys.executable}")
    print(f"OpenCV：{cv2.__version__}")
    print(f"模型：{yolo.MODEL_PATH}")
    print(f"摄像头：{args.device}")
    print(
        f"采集：{args.width}x{args.height} NV12"
    )
    print(
        f"模型输入：{yolo.IMAGE_SIZE}x"
        f"{yolo.IMAGE_SIZE}"
    )
    print(f"置信度：{yolo.CONF_THRESHOLD}")
    print(f"NMS阈值：{yolo.NMS_THRESHOLD}")
    print(f"类别：{yolo.CLASS_NAMES}")
    print(f"结果：{args.result_dir}")

    frame_store = LatestFrameStore()
    result_store = ResultStore()
    stop_event = threading.Event()

    capture_worker = V4L2CaptureWorker(
        frame_store=frame_store,
        device=args.device,
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        buffers=args.buffers,
    )

    http_server: ThreadingHTTPServer | None = None
    rknn = None

    def request_stop(
        signum=None,
        frame=None,
    ) -> None:
        stop_event.set()
        capture_worker.stop()

    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        print("\n========== 初始化NPU ==========")
        rknn = yolo.initialize_rknn()
        print("NPU初始化成功")

        if not args.no_web:
            http_server = start_http_server(
                result_store,
                args.host,
                args.port,
            )

            board_ips = get_board_ips()

            print("\n========== 浏览器实时画面 ==========")

            if board_ips:
                for ip_address in board_ips:
                    print(
                        f"http://{ip_address}:{args.port}/"
                    )
            else:
                print(
                    f"http://板子IP:{args.port}/"
                )

        capture_worker.start()

        print("\n按 Ctrl+C 停止程序。")

        run_inference_loop(
            args=args,
            yolo=yolo,
            rknn=rknn,
            frame_store=frame_store,
            result_store=result_store,
            stop_event=stop_event,
        )

    finally:
        stop_event.set()
        capture_worker.stop()

        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()

        if rknn is not None:
            rknn.release()

        print("\n程序已退出，NPU资源已释放。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print("\n用户终止程序。")
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\n运行失败：{exc}",
            file=sys.stderr,
        )

        print(
            f"V4L2日志：{V4L2_LOG_PATH}",
            file=sys.stderr,
        )

        raise SystemExit(1)
