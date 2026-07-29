# RK3588 + 双 IMX219 + YOLOv8 RKNN 双摄实时检测复现手册（小白版）

> 适用平台：创龙 TL3588F-EVM（RK3588）  
> 摄像头：树莓派 Camera Module v2.1（IMX219）×2  
> 当前项目映射：CAMERA1 为左相机，CAMERA2 为右相机  
> 当前验证节点：左 `/dev/video22`，右 `/dev/video31`  
> 模型示例：`/root/yolov8_rk3588/int8_best.rknn`，类别 `100- O-O-S`

---

## 0. 先看这三条规则

1. **看清终端属于谁再执行命令。**
   - `ao@ao-vm:...$`：Ubuntu 编译主机，简称 **Host**。
   - `root@RK3588-Tronlong:~#`：RK3588 板端，简称 **Target**。
   - `C:\Users\...>` 或 PowerShell：Windows 电脑，简称 **PC**。
2. **插拔 MIPI 摄像头排线前必须断电。**带电插拔可能造成摄像头、接口或板卡损坏。
3. **写 boot 分区之前必须备份并回读校验。**不要直接照抄 `dd`，本包已经提供安全脚本。

---

## 1. 复现成功的最终标准

完成本手册后，应同时满足：

- `dmesg` 中出现两条 IMX219 Model ID：
  - `imx219 1-0010: Model ID 0x0219`
  - `imx219 3-0010: Model ID 0x0219`
- Media 拓扑正确：
  - CAMERA1 → `rkcif-mipi-lvds2` → `rkisp0-vir0` → `/dev/video22`
  - CAMERA2 → `rkcif-mipi-lvds3` → `rkisp0-vir1` → `/dev/video31`
- 两路都能抓取 `1920×1080 NV12` 图像。
- 浏览器打开 `http://板子IP:8080/`，可以并排看到左右画面和 YOLO 检测框。
- 程序停止后，NPU、摄像头和端口均正常释放。

---

## 2. 为什么系统中有这么多节点

数据链路不是“摄像头直接等于 `/dev/videoX`”，而是：

```text
IMX219传感器
  → MIPI CSI/DPHY：接收高速串行图像数据
  → RKCIF：接收、分流RAW数据
  → RKISP：去马赛克、曝光、白平衡、降噪、颜色转换
  → rkisp_mainpath：输出正常的NV12彩色图像
  → /dev/video22 或 /dev/video31
```

`/dev/media0～3` 是 Media Controller 管线控制节点；`/dev/video0～40` 中还包含 RAW、统计、参数、selfpath、HDMI 输入等功能节点。YOLO 应优先使用 `rkisp_mainpath`，当前验证为 `/dev/video22` 和 `/dev/video31`。

---

## 3. 硬件准备与接线

### 3.1 硬件清单

- TL3588F-EVM 评估板及稳定电源。
- 可启动的 Linux SD 卡。
- IMX219 Camera Module v2.1 ×2。
- 与评估板匹配的 FFC 排线 ×2。
- 网线或可用的网络连接。
- Ubuntu x86_64 编译环境，已经解压 Linux SDK。

### 3.2 接线规则

- CAMERA1 接口上的 IMX219 固定作为**左相机**。
- CAMERA2 接口上的 IMX219 固定作为**右相机**。
- 两颗摄像头以后进行双目标定后，不得互换位置，不得改变基线距离和夹角。
- 复现阶段先保证两颗摄像头都出图；真正测距前再制作刚性支架。

### 3.3 IQ 文件

板端应存在：

```bash
ls -l /etc/iqfiles/imx219_CMK-OT1980-PX1_SHG102.json
```

若不存在，从产品资料中复制到 `/etc/iqfiles/`，再执行：

```bash
dos2unix /etc/iqfiles/imx219_CMK-OT1980-PX1_SHG102.json
sync
reboot
```

**为什么：**IQ 文件保存 IMX219 的 ISP 调参参数；缺失时可能出现颜色异常、曝光异常或 ISP 处理质量差。

---

## 4. 文件包结构

```text
rk3588_dual_imx219_repro/
├── README.md
├── dts/
│   └── tl3588f-evm-dual-imx219.dts
├── host/
│   └── build_dual_imx219_boot.sh
├── board/
│   ├── flash_dual_imx219_sd.sh
│   ├── find_dual_camera_nodes.sh
│   └── collect_dual_camera_diagnostics.sh
└── python/
    ├── test_dual_capture.py
    └── dual_yolov8_rknn_camera.py
```

---

## 5. 第一阶段：Ubuntu 主机编译双摄 boot.img

### 5.1 确认当前是 Ubuntu 主机

**Host 执行：**

```bash
whoami
hostname
uname -m
pwd
```

必须看到 `x86_64`。不要在 RK3588 板端编译完整 SDK。

### 5.2 检查 SDK

当前项目默认 SDK 路径：

```bash
cd /home/ao/RK3588SDK/rk3588_linux_release
ls -l build.sh
ls -l device/rockchip/rk3588/tl3588_evm_defconfig
ls -l kernel/arch/arm64/boot/dts/rockchip/tl3588f-evm.dts
```

### 5.3 一键编译

在文件包根目录执行：

```bash
cd rk3588_dual_imx219_repro
chmod +x host/build_dual_imx219_boot.sh
SDK_ROOT=/home/ao/RK3588SDK/rk3588_linux_release \
    ./host/build_dual_imx219_boot.sh
```

脚本会做以下工作：

1. 检查主机架构和 SDK 路径。
2. 把双摄 DTS 复制到内核设备树目录。
3. 备份 `tl3588_evm_defconfig`。
4. 将 `RK_KERNEL_DTS_NAME` 改为 `tl3588f-evm-dual-imx219`。
5. 执行 `./build.sh lunch:tl3588_evm_defconfig`。
6. 执行 `./build.sh kernel`。
7. 生成 `output/firmware/boot-dual-imx219.img` 和 SHA256 文件。

### 5.4 为什么要修改 DTS

官方单摄 DTS 通常只打开一个接口：

```c
CAMERA1_ENABLE_IMX219 = 1
CAMERA2_ENABLE_IMX219 = 0
```

双摄版本需要同时建立两条独立链路：

```text
CAMERA1 / i2c1 / MIPI2 / CIF2 / ISP vir0
CAMERA2 / i2c3 / MIPI3 / CIF3 / ISP vir1
```

两颗 IMX219 都使用 I²C 地址 `0x10` 并不冲突，因为它们位于不同的 I²C 总线上。

### 5.5 编译结果检查

```bash
cd /home/ao/RK3588SDK/rk3588_linux_release
ls -lh kernel/boot.img
ls -lh output/firmware/boot-dual-imx219.img
sha256sum output/firmware/boot-dual-imx219.img
```

当前项目曾验证镜像大小约 31 MiB。重新编译后的 SHA256 可能不同，**以本次实际值为准**。

---

## 6. 第二阶段：上传镜像和脚本到 RK3588

### 6.1 检查网络

**Host 执行：**

```bash
ping -c 4 192.168.10.66
ssh root@192.168.10.66
```

板端 IP 可能由 DHCP 改变，可在板端执行 `hostname -I` 查看。

### 6.2 上传 boot.img

**Host 执行：**

```bash
ssh root@192.168.10.66 'mkdir -p /userdata/camera_boot /userdata/camera_backup'

scp /home/ao/RK3588SDK/rk3588_linux_release/output/firmware/boot-dual-imx219.img \
    root@192.168.10.66:/userdata/camera_boot/
```

### 6.3 上传板端脚本和 Python 代码

```bash
scp board/*.sh root@192.168.10.66:/root/
scp python/*.py root@192.168.10.66:/root/yolov8_rk3588/
```

### 6.4 对比 SHA256

**Host 执行：**

```bash
sha256sum /home/ao/RK3588SDK/rk3588_linux_release/output/firmware/boot-dual-imx219.img
ssh root@192.168.10.66 'sha256sum /userdata/camera_boot/boot-dual-imx219.img'
```

两边必须完全一致。SHA256 用于确认传输后文件没有损坏。

---

## 7. 第三阶段：安全写入 SD 卡 boot 分区

### 7.1 确认登录到了板端

```bash
ssh root@192.168.10.66
whoami
hostname
findmnt -no SOURCE /
```

当前项目应显示：

```text
root
RK3588-Tronlong
/dev/mmcblk1p6
```

### 7.2 执行安全写入脚本

**Target 执行：**

```bash
chmod +x /root/flash_dual_imx219_sd.sh
/root/flash_dual_imx219_sd.sh
```

确认信息无误后输入：

```text
WRITE_SD_BOOT
```

脚本会：

- 检查根分区必须是 `/dev/mmcblk1p6`。
- 检查目标必须是块设备 `/dev/mmcblk1p3`。
- 备份完整 64 MiB boot 分区到 `/userdata/camera_backup/`。
- 写入双摄 boot.img。
- 从分区回读相同字节数并比较 SHA256。

### 7.3 为什么不能直接随便执行 dd

RK3588 同时有 SD 卡和 eMMC：

- `/dev/mmcblk1`：当前项目 SD 卡。
- `/dev/mmcblk0`：eMMC。

目标写错可能导致系统无法启动。安全脚本利用启动来源、块设备、分区大小、备份和回读校验降低风险。

### 7.4 重启

只有看到“写入和校验成功”后执行：

```bash
sync
reboot
```

---

## 8. 第四阶段：验证两颗 IMX219 和 Media 管线

重新 SSH 登录：

```bash
ssh root@192.168.10.66
```

执行：

```bash
chmod +x /root/find_dual_camera_nodes.sh
/root/find_dual_camera_nodes.sh
```

### 8.1 传感器成功标准

```bash
dmesg | grep -E 'imx219 [0-9]+-0010: Model ID'
```

应出现两条：

```text
imx219 1-0010: Model ID 0x0219
imx219 3-0010: Model ID 0x0219
```

### 8.2 当前已验证节点

```text
左相机 CAMERA1：/dev/video22，rkisp0-vir0
右相机 CAMERA2：/dev/video31，rkisp0-vir1
```

视频编号可能在修改系统配置后变化，因此新环境第一次必须运行节点识别脚本，不能只凭记忆猜 `/dev/videoX`。

### 8.3 为什么使用 mainpath

`rkisp_mainpath` 是 ISP 的主输出，适合：

- 完整分辨率图像；
- NV12 输出；
- OpenCV、录像和 AI 推理；
- 后续双目标定和测距。

`rkcif` 的 `/dev/video0～21` 多为 RAW 或辅助节点，不建议新手直接用于 YOLO。

---

## 9. 第五阶段：先做双路抓图，不要直接上模型

### 9.1 进入 Python 环境

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate
python --version
python -c 'import cv2,numpy; print(cv2.__version__)'
```

### 9.2 双路同时抓图

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate
python -u test_dual_capture.py
```

输出目录：

```text
/root/yolov8_rk3588/dual_capture_test/left.jpg
/root/yolov8_rk3588/dual_capture_test/right.jpg
/root/yolov8_rk3588/dual_capture_test/pair.jpg
```

### 9.3 下载查看

**PC PowerShell 执行：**

```powershell
scp root@192.168.10.66:/root/yolov8_rk3588/dual_capture_test/pair.jpg .
```

### 9.4 为什么先抓图再跑 YOLO

这是工程中的“分层排错”：

- 抓图失败：检查摄像头、DTS、ISP、节点和格式。
- 抓图成功、模型失败：检查 RKNN 环境、模型和后处理。
- 两者都成功、网页失败：检查 JPEG、端口或网络。

不要把摄像头、模型、网页三个问题混在一起调。

---

## 10. 第六阶段：验证 RKNN 模型环境

### 10.1 模型和运行库

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate

ls -lh /root/yolov8_rk3588/int8_best.rknn
python -c 'from rknnlite.api import RKNNLite; print("RKNNLite OK")'
```

### 10.2 先跑图片推理

```bash
cd /root/yolov8_rk3588/rknn_model_zoo-main/examples/yolov8/python
/root/yolov8_rk3588/rknn310/bin/python yolov8_rknn_infer.py
```

图片推理成功后再进入双摄实时检测。这样可以确认模型输出数量、类别数、阈值和后处理逻辑正确。

---

## 11. 第七阶段：双摄实时 YOLO 检测

### 11.1 语法检查

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate
python -m py_compile dual_yolov8_rknn_camera.py
```

### 11.2 启动

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate
python -u dual_yolov8_rknn_camera.py
```

浏览器访问：

```text
http://192.168.10.66:8080/
```

状态接口：

```bash
curl -s http://127.0.0.1:8080/status.json
```

结果文件：

```text
/root/yolov8_rk3588/result_dual_camera/latest_left.jpg
/root/yolov8_rk3588/result_dual_camera/latest_right.jpg
/root/yolov8_rk3588/result_dual_camera/latest_pair.jpg
```

### 11.3 停止

正常停止：

```text
Ctrl+C
```

异常断开后清理：

```bash
pkill -f dual_yolov8_rknn_camera.py || true
pkill -f 'v4l2-ctl.*video22' || true
pkill -f 'v4l2-ctl.*video31' || true
```

---

## 12. 参数为什么这样设置，应该怎样调

### 12.1 摄像头分辨率 `1920×1080`

默认：

```text
--width 1920 --height 1080
```

原因：

- 保留货架边缘和远处空位细节；
- IMX219 当前管线验证支持该分辨率；
- 后续双目标定与证据图片统一使用 1080P。

代价：NV12 每帧约 3,110,400 字节；双路 30FPS 原始吞吐约 186 MB/s，还未包含颜色转换和复制。

调低场景：CPU 过载、网页卡顿、只想先验证流程。可测试 `1280×720` 或 `640×480`，但重新标定时必须使用最终分辨率。

### 12.2 采集帧率 `30 FPS`

默认：

```text
--camera-fps 30
```

作用：保持画面流畅，减少机器人或手持移动时的跳帧。采集 30FPS 不等于两路 YOLO 都能各跑 30FPS；当前基线版本使用一个 RKNN 上下文轮流处理两路。

### 12.3 像素格式 `NV12`

代码固定为：

```text
pixelformat=NV12
```

原因：

- RKISP 主路径原生支持；
- 每像素平均 1.5 字节，比 BGR 的 3 字节节省带宽；
- 适合后续连接 RGA、VPU 或零拷贝优化。

当前 Python 基线仍使用 CPU 将 NV12 转为 BGR。极限优化时应改为 RGA 预处理或 DMA-BUF 流水线。

### 12.4 模型输入 `640×640`

默认：

```text
--input-size 640
```

必须与模型转换/训练配置一致。代码使用 letterbox 等比例缩放，避免直接拉伸 16:9 图像造成目标形状变形。

不要在不重新转换模型的情况下随意改成 320 或 1280。

### 12.5 置信度阈值 `CONF=0.25`

默认：

```text
--conf 0.25
```

- 阈值下降：召回率提高，但误检增加。
- 阈值上升：误检减少，但漏检增加。

建议调参：

```bash
# 漏检明显
python -u dual_yolov8_rknn_camera.py --conf 0.15

# 误检明显
python -u dual_yolov8_rknn_camera.py --conf 0.40
```

正式阈值应根据验证集 PR 曲线和真实门店视频确定，不应只凭一张图。

### 12.6 NMS 阈值 `0.45`

默认：

```text
--nms 0.45
```

NMS 用于删除高度重叠的重复框：

- 太低：相邻目标可能被错误合并。
- 太高：同一目标可能保留多个重复框。

常见测试范围为 `0.35～0.60`，每次只改变一个参数。

### 12.7 V4L2 缓冲数 `4`

默认：

```text
--buffers 4
```

缓冲过少可能因处理抖动导致采集饥饿；缓冲过多会增加内存和排队延迟。4 是稳定性与延迟的折中。实时系统必须“只取最新帧”，不要把旧帧无限排队。

### 12.8 预热帧 `10`

默认：

```text
--warmup 10
```

丢弃启动后的前 10 帧，让自动曝光和自动白平衡趋于稳定。暗光环境可增加到 20～30；调试启动速度时可减小。

### 12.9 网页预览宽度 `640`

默认每路预览宽度：

```text
--preview-width 640
```

采集和推理仍使用 1080P，只把浏览器画面缩小。这样可降低 JPEG 编码、网络和浏览器渲染负载。

CPU 占用过高时：

```bash
python -u dual_yolov8_rknn_camera.py \
    --preview-width 480 \
    --jpeg-quality 70
```

### 12.10 JPEG 质量 `80`

质量越高，图像越清晰，但编码耗时和网络带宽越大。实时监控 70～85 通常足够；保存证据图可单独用 90 以上。

### 12.11 保存间隔 `1秒`

```text
--save-interval 1.0
```

当前程序每秒覆盖保存最新结果，不会无限产生图片。性能测试时可关闭磁盘写入：

```bash
python -u dual_yolov8_rknn_camera.py --save-interval 0
```

---

## 13. 当前代码能说明什么，不能说明什么

当前程序是**可复现基线**，不是 RK3588 极限性能版本。它包含：

- 两个 `v4l2-ctl` 子进程；
- CPU NV12→BGR；
- CPU letterbox、后处理、绘框和 JPEG；
- 一个 RKNN 上下文轮流处理左右图；
- Python 线程和数组复制。

因此看到的总 FPS 是整条链路瓶颈，不等于 NPU 纯模型极限。

性能测试应分三层：

1. **纯模型**：固定输入，只测 NPU 推理 P50/P95/P99 和吞吐。
2. **AI 管线**：加入预处理、后处理和双摄采集。
3. **完整产品**：加入深度、网页、编码、保存、温度和长期稳定性。

---

## 14. EPAJ 优化路线（后续创新）

可将系统优化定义为 EPAJ：

- **E - Engine-aware mapping**：ISP 负责成像，RGA 负责缩放/颜色转换，NPU 负责 YOLO，VPU/MPP 负责编码，CPU 只做调度和业务逻辑。
- **P - Pipeline parallelism**：采集、预处理、推理、后处理和编码同时流水执行；队列长度限制为 1～2。
- **A - Adaptive scheduling**：根据温度、负载和场景变化动态调整 YOLO、深度和右相机复核频率。
- **J - Joint stereo-semantic**：主要在左图运行 YOLO，右图用于立体匹配，只在检测 ROI 中计算深度，避免两路全图重复推理和全图稠密视差。

建议优化顺序：

```text
Python基线跑通
→ 测各阶段耗时
→ 关闭保存/JPEG找瓶颈
→ RGA替代CPU预处理
→ C/C++ RKNN Runtime
→ DMA-BUF/少拷贝
→ 左图YOLO + ROI深度
→ 多NPU上下文吞吐测试
```

---

## 15. 常见故障排查

### 15.1 只识别到一颗 IMX219

```bash
dmesg | grep -Ei 'imx219|i2c|csi|mipi|rkcif|rkisp' | tail -n 300
```

判断缺少的是：

- `1-0010`：检查 CAMERA1、i2c1、排线方向和复位 GPIO。
- `3-0010`：检查 CAMERA2、i2c3、排线方向和复位 GPIO。

断电后重新插排线，不要带电操作。

### 15.2 两颗传感器都有，但没有 `/dev/video22` 或 `/dev/video31`

```bash
/root/find_dual_camera_nodes.sh
```

检查是否有两个 `rkisp_mainpath`。不要假设视频编号永久固定，以脚本输出为准。

### 15.3 `Device or resource busy`

```bash
fuser -v /dev/video22 /dev/video31
pkill -f dual_yolov8_rknn_camera.py || true
pkill -f 'v4l2-ctl.*video22' || true
pkill -f 'v4l2-ctl.*video31' || true
```

### 15.4 帧大小不是 3,110,400 字节

1080P NV12 理论大小：

```text
1920 × 1080 × 1.5 = 3,110,400字节
```

大小不同通常表示实际格式或分辨率没有设置成功。检查：

```bash
v4l2-ctl -d /dev/video22 --get-fmt-video
v4l2-ctl -d /dev/video31 --get-fmt-video
```

### 15.5 RKNN 初始化失败

```bash
cd /root/yolov8_rk3588
source rknn310/bin/activate
python -c 'from rknnlite.api import RKNNLite; print("OK")'
ls -lh int8_best.rknn
```

确认板端安装的是 `rknn-toolkit-lite2`，不是主机转换用的 `rknn-toolkit2`。

### 15.6 模型运行但没有框

依次检查：

- 类别名和训练类别数量。
- 模型输出是 9 输出还是 1 输出。
- `CONF_THRESHOLD` 是否过高。
- RGB/BGR 是否反了。
- letterbox 与模型训练/转换逻辑是否一致。
- 模型在静态图片上是否能正确检测。

### 15.7 网页打不开

```bash
ss -lntp | grep 8080
curl -s http://127.0.0.1:8080/status.json
hostname -I
```

端口占用时：

```bash
python -u dual_yolov8_rknn_camera.py --port 8081
```

### 15.8 画面卡顿或延迟越来越大

实时系统应丢弃旧帧，而不是排队。进一步操作：

```bash
python -u dual_yolov8_rknn_camera.py \
    --preview-width 480 \
    --jpeg-quality 70 \
    --save-interval 0
```

同时用 `top` 观察 CPU。若显示流畅但 YOLO 低，说明推理/后处理是瓶颈；若关闭网页后明显变快，说明 JPEG/网络是瓶颈。

### 15.9 写入 boot 后无法启动

若仍能从其他介质启动，恢复最近备份：

```bash
ls -lt /userdata/camera_backup/boot-sd-before-dual-*.img

dd if=/userdata/camera_backup/boot-sd-before-dual-时间戳.img \
   of=/dev/mmcblk1p3 bs=4M status=progress conv=fsync
sync
reboot
```

恢复前必须重新确认目标分区。

### 15.10 一键收集诊断信息

```bash
chmod +x /root/collect_dual_camera_diagnostics.sh
/root/collect_dual_camera_diagnostics.sh
```

把生成的 `/userdata/dual_camera_diagnostics_时间戳.txt` 发给负责人。

---

## 16. 详细步骤清单

### A. 硬件

- [ ] 板卡完全断电。
- [ ] CAMERA1 插入左 IMX219。
- [ ] CAMERA2 插入右 IMX219。
- [ ] 两条 FFC 排线方向正确并锁紧。
- [ ] 上电后散热、网络、电源正常。

### B. Ubuntu 编译主机

- [ ] `uname -m` 为 `x86_64`。
- [ ] SDK 路径存在。
- [ ] DTS 已复制到内核目录。
- [ ] `RK_KERNEL_DTS_NAME="tl3588f-evm-dual-imx219"`。
- [ ] `./build.sh lunch:tl3588_evm_defconfig` 成功。
- [ ] `./build.sh kernel` 返回 0。
- [ ] `kernel/boot.img` 存在。
- [ ] `output/firmware/boot-dual-imx219.img` 存在。
- [ ] 已记录 SHA256。

### C. 上传和写入

- [ ] Host 可以 ping/SSH 板端。
- [ ] 镜像上传到 `/userdata/camera_boot/`。
- [ ] Host 与 Target SHA256 一致。
- [ ] `findmnt -no SOURCE /` 为 `/dev/mmcblk1p6`。
- [ ] 已备份 `/dev/mmcblk1p3`。
- [ ] 写入后回读 SHA256 一致。
- [ ] 才执行重启。

### D. 驱动和节点

- [ ] `imx219 1-0010` 存在。
- [ ] `imx219 3-0010` 存在。
- [ ] `/dev/media0～3` 存在。
- [ ] 左 mainpath 已确认。
- [ ] 右 mainpath 已确认。
- [ ] 两路支持 `NV12` 和 `1920×1080`。

### E. 双路抓图

- [ ] Python 虚拟环境可激活。
- [ ] OpenCV、NumPy 可导入。
- [ ] `test_dual_capture.py` 返回成功。
- [ ] `left.jpg` 正常。
- [ ] `right.jpg` 正常。
- [ ] 左右没有接反。

### F. RKNN 和模型

- [ ] `int8_best.rknn` 存在。
- [ ] RKNNLite 可导入。
- [ ] 静态图片推理成功。
- [ ] 类别为 `100- O-O-S`。
- [ ] 模型输出数量已确认。

### G. 双摄实时检测

- [ ] Python 语法检查成功。
- [ ] 左右采集线程启动。
- [ ] NPU 初始化成功。
- [ ] 浏览器能看到两路画面。
- [ ] 检测框位置正常。
- [ ] `status.json` 可访问。
- [ ] Ctrl+C 后程序和摄像头资源正常释放。

### H. 记录实验结果

- [ ] 记录镜像 SHA256。
- [ ] 记录内核版本。
- [ ] 记录实际视频节点。
- [ ] 记录模型 SHA256。
- [ ] 记录 CONF/NMS/输入尺寸。
- [ ] 记录左右 YOLO FPS 和推理时间。
- [ ] 记录 CPU、内存、温度。
- [ ] 保存一张左右合并结果图。
- [ ] 保存诊断日志。

---

## 17. 建议的实验记录模板

```text
日期：
负责人：
板卡：TL3588F-EVM
内核：
启动介质：SD / eMMC
boot.img SHA256：
模型 SHA256：
左节点：
右节点：
分辨率：1920x1080
采集FPS：30
模型输入：640x640
CONF：0.25
NMS：0.45
左推理FPS：
右推理FPS：
显示FPS：
CPU占用：
内存占用：
最高温度：
连续运行时间：
问题与解决方法：
```

---

## 18. 当前基线的下一步

复现完成后再开展：

1. 制作固定双目支架并确定基线。
2. 采集棋盘格图像，完成双目标定。
3. 立体校正并检查水平极线。
4. 先用降采样图计算视差。
5. 将左图 YOLO 框映射为深度 ROI。
6. 输出“类别 + 置信度 + 距离”。
7. 分层测量 RK3588 的 NPU、CPU、ISP、内存和编码瓶颈。
8. 按 EPAJ 路线逐步迁移到 RGA、C++ 和少拷贝流水线。
