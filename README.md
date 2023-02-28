# YOLOv7-and-ByteTrack-Test
* 以下基于pytorch在pc平台执行即可，mot17和mot20(多目标追踪数据集)

## 目标检测(仅对行人检测)
基于rk的yolov7 github（https://github.com/airockchip/yolov7），验证目标检测指标和运行速度，结果可视化。
实验：
1. yolov7_tiny_relu(rk官方模型)

2. yolov7_tiny_silu
    * 对rk官方模型改ReLU激活函数，改为SiLU
    * 改动参考https://github.com/airockchip/yolov7/commit/b61b954c0d410470ef2d158d1c946ee5650b96dc

3. yolov7_tiny_relu(大核maxpool)
    * rk的yolov7中SPP是利用多个3x3 MaxPool堆叠代替大核MaxPool，因此需测试大核MaxPool情况下yolov7
    * 改动参考https://github.com/airockchip/yolov7/commit/f03cf65e4d955d47bea8b2a9c0223221f5e41053#diff-cfb1ff087a99a34369673c9f34bdcd22f2d429ab3599a89f386c5de1fd9a2566

## 多目标跟踪
基于yolov7_tiny_relu的bytetrack，验证多目标跟踪指标和速度，结果可视化。
对ByteTrack文件夹下的文件进行改动，接入yolov7。
实验：
1. yolov7_tiny_relu(rk官方模型)

2. yolov7_tiny_silu

3. yolov7_tiny_relu(大核maxpool)