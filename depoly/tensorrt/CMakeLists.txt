# requirement of cmake version
cmake_minimum_required(VERSION 3.5)
# project name
PROJECT(yolovt_trt CXX)
set(CMAKE_CXX_STANDARD 17)
# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)   # 输出文件夹路径
set(CMAKE_BUILD_TYPE "Release")  # 编译类型

# find required opencv
find_package(OpenCV REQUIRED)   # 使用find_package获取opencv的相关信息,如lib、include、version等
#find_package(glog REQUIRED)

# tensorrt
set(TENSORRT_ROOT_PATH /home/liguiyuan/software/TensorRT-7.0.0.11)   # 根目录
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT_PATH}/include/)  # 头文件夹
set(TENSORRT_LIB ${TENSORRT_ROOT_PATH}/lib/libnvinfer.so ${TENSORRT_ROOT_PATH}/lib/libnvonnxparser.so)  # 依赖动态库

# cuda
set(CUDA_ROOT_PATH /usr/local/cuda)  # 根目录
set(CUDA_INCLUDE_DIRS ${CUDA_ROOT_PATH}/include/)  # 头文件夹
set(CUDA_LIB ${CUDA_ROOT_PATH}/lib64/libcudart.so) # 依赖动态库

# name of executable file and path of source file
add_executable(yolov5_trt src/yolov5_trt.cpp)

# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})  # 头文件夹
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})  # 指定第三方库所在路径,该路径添加到第三方库搜索路径中
target_link_libraries(yolov5_trt ${OpenCV_LIBS} ${TENSORRT_LIB} ${CUDA_LIB})  # 相对路径




