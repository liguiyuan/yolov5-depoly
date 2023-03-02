cxx = g++
common = `pkg-config --cflags opencv`
LIB = ../ncnn/lib/libncnn.a
LIB += `pkg-config --libs opencv`
INCLUDE = -I../ncnn/include/ncnn
COMMON += -fopenmp -pthread -std=c++17
DEPS = $(wildcard *.h)


all : yolov5
yolov5 : yolov5.cpp
	$(CXX) $(COMMON) $(INCLUDE) $^ -o $@ $(LIB)
%.o : %.cpp $(DEPS)
	$(CXX) $(COMMON) $(INCLUDE) -c $< -o $@
.PHONY : clean
clean :
	rm -rf yolov5
