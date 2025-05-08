#include "yolov11.h"
#include <fstream>

// 模型是否预热
#define warmup true

YOLOv11::YOLOv11(const std::string &engine_path, nvinfer1::ILogger& logger) {
    // 加载模型引擎文件,以二进制方式
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    // 分配内存
    std::unique_ptr<char[]> trtModelBuffer(new char[size]);
    file.read(trtModelBuffer.get(), size);
    file.close();

    // 反序列化引擎
    this->runtime.reset(nvinfer1::createInferRuntime(logger));
    this->engine.reset(this->runtime->deserializeCudaEngine(trtModelBuffer.get(), size));
    delete[] trtModelBuffer.get(); 
    this->context.reset(this->engine->createExecutionContext());

    // 获取输入输出维度信息
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    this->input_h = input_dims.d[2];
    this->input_w = input_dims.d[3];
    auto resized_imgput_dims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = resized_imgput_dims.d[1];    // 每个目标属性大小：类别数+置信度+4个坐标
    num_detections = resized_imgput_dims.d[2];             // 模型输出的目标数量
    num_classes = detection_attribute_size - 4;

    // 为输出缓冲区分配cpu内存
    cpu_output_buffer = new float[num_detections * detection_attribute_size];
    // 输入缓冲区分配gpu内存
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], input_h * input_w * 3 * sizeof(float)));
    // 输出缓冲区分配gpu内存
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float)));

    // 初始化CUDA流
    CUDA_CHECK(cudaStreamCreate(&stream));\

    // 模型预热
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        std::cout << "warmup 10 times" << std::endl;
    }
}

YOLOv11::~YOLOv11() {
    // 同步并销毁 CUDA 流，确保所有在该流上排队的 CUDA 操作完成后再释放相关资源
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 释放GPU内存
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1])); 

    // 释放CPU内存
    delete[] cpu_output_buffer;
}

/**
 * @brief 对输入图像进行预处理操作，包括调整大小、填充、归一化、颜色通道转换
 * @param img 待处理的输入图像，类型为cv::Mat
 * @param resized_img 处理后的图像，类型为cv::Mat
 * @param size 目标图像大小，类型为cv::Size
 */
void YOLOv11::img_preprocess(cv::Mat &img, cv::Mat &resized_img, cv::Size &size) {
    const float input_h = size.height;
    const float input_w = size.width;
    float height = img.rows;
    float width = img.cols;

    float r = std::min(input_h / height, input_w / width);
    int pad_h = std::round(r * height);
    int pad_w = std::round(r * width);

    cv::Mat tmp;
    // 如果原始图像尺寸和缩放后尺寸不同，则进行缩放操作
    if ((int)width != pad_w || (int)height != pad_h)
    {
        cv::resize(img, tmp, cv::Size(pad_w, pad_h));
    }
    else
    {
        tmp = img.clone();
    }

    float dw = input_w - pad_w;
    float dh = input_h - pad_h;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // 创建一个大小为 (1, 3, input_h, input_w) 的矩阵
    resized_img.create({1, 3, (int)input_h, (int)input_w}, CV_32F);

    // 将图像分离为三个通道
    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    // 创建三个通道的矩阵视图，分别指向 resized_img 的不同部分，每个通道依次偏移一个通道大小
    cv::Mat c0((int)input_h, (int)input_w, CV_32F, (float *)resized_img.data);
    cv::Mat c1((int)input_h, (int)input_w, CV_32F, (float *)resized_img.data + (int)input_h * (int)input_w);
    cv::Mat c2((int)input_h, (int)input_w, CV_32F, (float *)resized_img.data + (int)input_h * (int)input_w * 2);

    // 将通道数据转换为 float 类型，并归一化到 [0, 1] 范围
    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);
}

/**
 * @brief 将img处理成模型需要的输入格式，然后拷贝到device端
 */

void YOLOv11::preprocess(cv::Mat &img) {
    cv::Mat resized_img;
    this->img_preprocess(img, resized_img, cv::Size(input_w, input_h));

    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], resized_img.data, 
        resized_img.total() * resized_img.elemSize(), cudaMemcpyHostToDevice, this->stream));

    // 同步 CUDA 流，确保所有 CUDA 操作完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer() {
    this->context->enqueueV3(this->stream);
}

void YOLOv11::postprocess(std::vector<Object> &objects) {
    // 从GPU内存中拷贝输出结果到CPU内存中，异步操作
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, this->stream)); 
    // 同步 CUDA 流，确保所有 CUDA 操作完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<cv::Rect> boxes;          // Bounding boxes
    std::vector<int> class_ids;       // Class IDs
    std::vector<float> confidences;   // Confidence scores

    // Create a matrix view of the detection output
    const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        cv::Point class_id_point;
        double score;
        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Check if the confidence score exceeds the threshold
        if (score > conf_threshold) {
            // Extract bounding box coordinates
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            // Calculate top-left corner of the bounding box
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            // Set width and height of the bounding box
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            // Store the bounding box, class ID, and confidence
            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
    // Apply NMS to remove overlapping boxes
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // Iterate over NMS results and populate the output detections
    for (int i = 0; i < nms_result.size(); i++)
    {
        Object result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        objects.push_back(result);
    }
}

void YOLOv11::draw(cv::Mat &img, std::vector<Object> &objects) {
    // 计算缩放比例
    const float ratio_h = input_h / float(img.rows);
    const float ratio_w = input_w / float(img.cols);

    // 遍历每个检测到的目标
    for (auto &obj : objects) {
        // 通过类别id分配颜色
        cv::Scalar color = cv::Scalar(COLORS[obj.class_id][0], COLORS[obj.class_id][1], COLORS[obj.class_id][2]);

        // 调整边界框比例
        auto bbox = obj.bbox;
        if (ratio_h < ratio_w) {
            bbox.x = bbox.x / ratio_w;
            bbox.y = (bbox.y - (input_h - ratio_w * img.rows) / 2) / ratio_w;
            bbox.width = bbox.width / ratio_w;
            bbox.height = bbox.height / ratio_w;
        } 
        else {
            bbox.x = (bbox.x - (input_w - ratio_h * img.cols) / 2) / ratio_h;
            bbox.y = bbox.y / ratio_h;
            bbox.width = bbox.width / ratio_h;
            bbox.height = bbox.height / ratio_h;
        }
        
        // 绘制边界框
        cv::rectangle(img, bbox, color, 2);

        // 绘制类别名称和置信度
        std::string class_string = CLASS_NAMES[obj.class_id] + ": " + std::to_string(obj.conf);
        // 获取文本的大小
        cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        // 定义文本框矩形
        cv::Rect text_rect(bbox.x, bbox.y - 40, text_size.width + 10, text_size.height + 20);
        // 绘制文本框背景矩形
        cv::rectangle(img, text_rect, color, cv::FILLED);
        // 绘制文本
        cv::putText(img, class_string, cv::Point(bbox.x + 5, bbox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    
    }
}
