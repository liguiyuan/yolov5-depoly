## yolov5 v6.0



**1.Export tensorrt model**

```bash
python export.py --weights yolov5s.pt --include engine --device 0
```

It will generate `yolov5s.engine` model.



**2.Run engine model**

When we depoly tensorrt model, we can make sure engine is available.

```bash
python detect.py --weights yolov5s.engine --source ./data/images/bus.jpg
```



**3.Tensorrt depoly**



```bash
git clone https://github.com/wang-xinyu/tensorrtx.git
```

```bash
cp tensorrtx/yolov5/gen_wts.py yolov5/
```

```bash
cd yolov5/
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
cp yolov5s.wts ../tensorrtx/yolov5/
```

```bash
cd ../tensorrtx/yolov5/
mkdir build
cp yolov5s.wts ./build
cd build
cmake ..
make

# Usage:
#./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
#./yolov5_det -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

# For example yolov5s
./yolov5_det -s yolov5s.wts yolov5s.engine s
./yolov5_det -d yolov5s.engine ../images
```





















