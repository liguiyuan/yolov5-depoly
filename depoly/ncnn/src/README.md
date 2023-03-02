## ncnn inference

#### 1. convert `pt` model to `ncnn` model

references: [ultralytics/yolov5#251](https://github.com/ultralytics/yolov5/issues/251)

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



#### 2. onnx model simplify

run cmd:

```bash
python export.py --weights yolov5s.pt --include onnx --img 640 --train --simplify
```

--include  : the type of the model

--train  : not directly output classification results, but to output three feature maps.

--simplify  : simplify model, not need `onnxsim` step.



#### 3. convert onnx model to ncnn model

```bash
./onnx2ncnn yolov5s_6_0.onnx yolov5s_6_0.param yolov5s_6_0.bin
```



#### 4.ncnn optimize

```bash
./ncnnoptimize yolov5s_6_0.param yolov5s_6_0.bin yolov5s_6_0-opt.param yolov5s_6_0-opt.bin 65536
```



#### 5. modify ncnn param file

open `yolov5s_6_0.param` file, find all the `Reshape` layers,  change `0=6400, 0=1600, 0=400` to 0=-1, 0=-1, 0=-1, then save file.

<img src="https://github.com/liguiyuan/yolov5-depoly/blob/main/depoly/ncnn/images/modify0.png"/>



<img src="https://github.com/liguiyuan/yolov5-depoly/blob/main/depoly/ncnn/images/modify1.png"/>



#### 6. run demo

run cmd:

```bash
make
```

```bash
./yolov5 ../images/000000070254.jpg
```

<img src="https://github.com/liguiyuan/yolov5-depoly/blob/main/depoly/ncnn/images/result.jpg"/>





















