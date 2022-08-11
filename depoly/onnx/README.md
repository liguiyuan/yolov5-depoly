

## onnx inference



**1.clone yolov5 project**

```bash
git clone https://github.com/ultralytics/yolov5.git
```



**2.download yolov5 inference model, such as `yolov5s.pt`.**

url: https://github.com/ultralytics/yolov5/releases



**3.convert `pt` model to `onnx` model**

references: https://github.com/ultralytics/yolov5/issues/251

First, used this cmd to convert (PyTroch>=1.9.0+): 

```bash
python export.py --weights yolov5s.pt --include onnx
```

If your PyTroch==1.7.1, may be you need to use `--opset 12` to convert, such as:

```bash
python export.py --weights yolov5s.pt --include onnx --opset 12
```



If convert success, output:

> ONNX: export success, saved as yolov5s.onnx (29.3 MB)
> ONNX: run --dynamic ONNX model inference with: 'python detect.py --weights yolov5s.onnx'



Second, we can use onnx model to inference:

```bash
python detect.py --weights yolov5s.onnx --source ./data/images/bus.jpg
```

<img src="https://github.com/liguiyuan/yolov5-depoly/blob/main/docs/bus.jpg"/>









