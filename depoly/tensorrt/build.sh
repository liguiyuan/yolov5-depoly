cd ./build
rm -rf *
cmake ..
make
cd ../bin
./yolov5_trt
