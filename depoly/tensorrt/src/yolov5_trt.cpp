#include <fstream>
#include <iostream>
#include <ctime>
#include <sys/stat.h>
#include <glog/logging.h>
#include "yolov5_trt.h"


using namespace nvinfer1;

// Logger for TRT info/warning/errors, https://github.com/onnx/onnx-tensorrt/blob/main/onnx_trt_backend.cpp
class TRT_Logger : public nvinfer1::ILogger
{
    nvinfer1::ILogger::Severity _verbosity;
    std::ostream* _ostream;

public:
    TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream& ostream = std::cout)
        : _verbosity(verbosity)
        , _ostream(&ostream)
    {
    }
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= _verbosity)
        {
            time_t rawtime = std::time(0);
            char buf[256];
            strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR
                        ? "  ERROR"
                        : severity == Severity::kWARNING ? "WARNING" : severity == Severity::kINFO ? "   INFO"
                                                                                                   : "UNKNOWN");
            (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
        }
    }  
};



void YOLOv5::loadTrt(const std::string strName)
{
    TRT_Logger gLogger;

    this->m_CudaRuntime = createInferRuntime(gLogger);
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr); // runtime对象反序列化
    m_CudaContext = m_CudaEngine->createExecutionContext();  // 可以查询引擎获取有关网络的输入和输出的张量信息--维度/数据格式/数据类型
    m_CudaRuntime->destroy();
}


void YOLOv5::loadOnnx(const std::string strModelName)
{
    TRT_Logger gLogger;

    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(1);  // batchsize
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 显式批处理
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);   // 定义模型
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);  // 使用nvonnxparser 定义一个可用的onnx解析器
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));  // 解析onnx

    // 使用builder对象构建engine
    IBuilderConfig* config = builder->createBuilderConfig();
    // 特别重要的属性是最大工作空间大小
    config->setMaxWorkspaceSize(1ULL << 30);  // 分配内存空间
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config); // 来创建一个 ICudaEngine 类型的对象，在构建引擎时，TensorRT会复制权重

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    IHostMemory *gieModelStream = m_CudaEngine->serialize();  // 将引擎序列化
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());

    // memcpy内存拷贝函数 ，从源内存地址的起始位置开始拷贝若干个字节到目标内存地址中
    memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
    serialize_output_stream.open(strTrtName.c_str());
    serialize_output_stream<<serialize_str;  // 将引擎序列化数据转储到文件中
    serialize_output_stream.close();
    m_CudaContext = m_CudaEngine->createExecutionContext();

    // 使用一次，销毁parser，network, builder, and config
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}

static bool ifFileExists(const char *FileName)
{
    struct stat my_start;
    return (stat(FileName, &my_start) == 0);
}


YOLOv5::YOLOv5(Configuration config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;
    this->inpHeight = 640;
    this->inpWidth = 640;

    std::string model_path = config.modelpath;  // 模型权重路径
    std::string strTrtName = config.modelpath;  // 加载模型权重
    size_t sep_pos = model_path.find_last_of(".");  // 
    strTrtName = model_path.substr(0, sep_pos) + ".engine";  // ".trt"

    if (ifFileExists(strTrtName.c_str()))
    {
        loadTrt(strTrtName);
    } else {
        loadOnnx(config.modelpath);
    }

    // 利用加载的模型获取输入输出信息
    // 使用输入和输出blob名来获取输入和输出索引
    m_iInputIndex = m_CudaEngine->getBindingIndex("images");  // 模型输入
    m_iOutputIndex = m_CudaEngine->getBindingIndex("output"); // 模型输出

    Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);  // 输入
    int size1 = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];  // 展平
    m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);  // 输入尺寸(W, H)

    Dims dims_o = m_CudaEngine->getBindingDimensions(m_iOutputIndex); // 输出，维度[0,1,2,3]NHWC
    int size2 = dims_o.d[0] * dims_o.d[1] * dims_o.d[2];  // 所有大小
    m_iClassNums = dims_o.d[2] - 5;  // [,,classes+5]
    m_iBoxNums = dims_o.d[1];  // [b,num_pre_boxes,classes+5]

    // 分配内存大小
    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size1 * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(size1 * sizeof(float));
    m_ArraySize[m_iInputIndex] = size1 * sizeof(float);
    cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size2 * sizeof(float));
    m_ArrayHostMemory[m_iOutputIndex] = malloc(size2 * sizeof(float));
    m_ArraySize[m_iOutputIndex] = size2 * sizeof(float);

    // bgr
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    
}


void YOLOv5::UnInit()
{

}

YOLOv5::~YOLOv5()
{
    UnInit();
}


cv::Mat YOLOv5::resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows;
    int srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    cv::Mat dstimg;



}


void YOLOv5::detect(cv::Mat& frame)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
}


int main(int argc, char const *argv[])
{
    clock_t startTime, endTime;  // 计算时间
    Configuration yolo_nets = {0.3, 0.5, 0.3, "../models/yolov5s.onnx"};
    YOLOv5 yolo_model(yolo_nets);

    std::string imgpath = "../images/bus.jpg";
    cv::Mat srcimg = cv::imread(imgpath);

    double timeStart = (double)cv::getTickCount();
    startTime = clock();

    yolo_model.detect(srcimg);
    endTime = clock();


    return 0;
}

