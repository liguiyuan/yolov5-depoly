## yolov5 v6.1



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

todo!



















