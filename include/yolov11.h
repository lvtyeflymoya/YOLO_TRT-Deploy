#include "common.hpp"




/**
 * @brief 使用tensorRT进行YOLOv11的推理的类
 */
class YOLOv11 {
public:
    YOLOv11(const std::string &engine_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    void img_preprocess(cv::Mat &img, cv::Mat &resized_img, cv::Size &size);
    void preprocess(cv::Mat &img);
    void infer();
    void postprocess(std::vector<Object> &objects);
    void draw(cv::Mat &img, std::vector<Object> &objects);
                    
private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime = nullptr;
    cudaStream_t stream;

    float* cpu_output_buffer = nullptr;  // 输出cpu缓冲区
    void* gpu_buffers[2];  // gpu缓冲区：引擎执行时需要

    // 模型参数
    int input_w;
    int input_h;
    int num_detections; // 模型检测时的输出数量
    int detection_attribute_size; // 模型检测时的输出维度
    int num_classes = 80;   // 模型的类别数量
    float conf_threshold = 0.5f; // 置信度阈值
    float nms_threshold = 0.4f; // NMS阈值
    std::vector<std::string> class_names; // 类别名称
    std::vector<std::vector<unsigned int>> colors;  // 类别颜色
};