## yolov5 tensorrt depolyment



**1. Download `tensorrtx` project**

url: https://github.com/wang-xinyu/tensorrtx.git

Then, copy `tensorrtx/yolov5/gen_wts.py` to `yolov5` directory.

```bash
cp tensorrtx/yolov5/gen_wts.py ./yolov5
```



**2.Generate `yolov5s.wts` file**

```bash
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
```



**3.Download TensorRT and build, install in Ubuntu18.04**

see: https://blog.csdn.net/u012505617/article/details/110437890



**4.Build `tensorrtx/yolov5` project**

```bash
cd {tensorrtx}/yolov5/
mkdir build
cd build
cp yolov5/yolov5s.wts tensorrtx/yolov5/build
cmake ..
make
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

We can modify `CMakeLists.txt` file to solve this error:

```cmake
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
# modify：
include_directories(/home/liguiyuan/software/TensorRT-7.0.0.11/include/)
link_directories(/home/liguiyuan/software/TensorRT-7.0.0.11/lib)
```

Re run "make" again.

Success!



**5.Generate engine model**

```bash
./yolov5 -s  yolov5s.wts yolov5s.engine s
```

This step will generate `yolov5s.engine`, `libmyplugins.so` file.



**6.We can run inference by TensorRT** 

```bash
./yolov5 -d yolov5s.engine ../samples
```



Model inference used time only 5ms.



