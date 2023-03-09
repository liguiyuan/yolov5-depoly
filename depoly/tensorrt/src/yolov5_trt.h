#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <NvInfer.h>        // nvidia加载模型进行推理的插件
#include <NvOnnxParser.h>
#include <cuda_runtime.h>


// 自定义配置结构
struct Configuration
{
    float confThreshold;    // Confidence threshold
    float nmsThreshold;     // Non-maximum suppression threshold
    float objThreshold;     // Object Confidence threshold
    std::string modelpath;
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;


class YOLOv5
{
public:
    YOLOv5(Configuration config);
    ~YOLOv5();
    void UnInit();
    void detect(cv::Mat& frame);

private:
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    int inpWidth;
    int inpHeight;

    void loadOnnx(const std::string strName);
    void loadTrt(const std::string strName);

    cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);


    nvinfer1::ICudaEngine *m_CudaEngine;
    nvinfer1::IRuntime *m_CudaRuntime;
    nvinfer1::IExecutionContext *m_CudaContext;
    cudaStream_t m_CudaStream;  // 初始化流,CUDA流的类型为cudaStream_t 

    int m_iInputIndex;
    int m_iOutputIndex;
    int m_iClassNums;
    int m_iBoxNums;

    cv::Size m_InputSize;

    void* m_ArrayDevMemory[2]{0};
    void* m_ArrayHostMemory[2]{0};
    int m_ArraySize[2]{0};

    std::vector<cv::Mat> m_InputWrappers{};

    
};



#endif  // YOLOV5_TRT_H



