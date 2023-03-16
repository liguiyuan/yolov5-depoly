## yolov5 tensorrt depolyment



**1. Download `tensorrtx` project**

url: https://github.com/wang-xinyu/tensorrtx.git

```bash
git clone https://github.com/wang-xinyu/tensorrtx.git
```

Then, copy `tensorrtx/yolov5/gen_wts.py` to `yolov5` root directory.

```bash
cp tensorrtx/yolov5/gen_wts.py ./yolov5
```



**2.Generate `yolov5s.wts` file**

```bash
cd yolov5/
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
cp yolov5s.wts ../tensorrtx/yolov5/
```



**3.Generate `yolov5s.engine` file**

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



If run "make" command error, such as:

> tensorrtx/yolov5/yololayer.h:6:10: fatal error: NvInfer.h: 没有那个文件或目录
>  #include <NvInfer.h>
>           ^~~~~~~~~~~
> compilation terminated.
> CMake Error at myplugins_generated_yololayer.cu.o.Debug.cmake:219 (message):
>   Error generating

​		

That maybe cause by not found  `NvInfer.h` file.

Modify `CMakeLists.txt` file `cuda, tensorrt` path to your own path:

```cmake
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/liguiyuan/software/TensorRT-7.0.0.11/include/)
link_directories(/home/liguiyuan/software/TensorRT-7.0.0.11/lib)
```

Re run "make" again.

Success!



**6.Inference by TensorRT** 

```bash
./yolov5_det -d yolov5s.engine ../images
```

Model inference time only 5ms.



<img src="https://github.com/liguiyuan/yolov5-depoly/tree/main/depoly/tensorrt/images/_bus.jpg"/>

