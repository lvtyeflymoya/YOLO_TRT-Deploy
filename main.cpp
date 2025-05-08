
#include "../include/yolov11.h"

int main(int argc, char** argv) {
    // 示例用法
    Logger logger;
    YOLOv11 detector("D:/Cpp_Project/PanoramicTracking/onnx_tensorRT/models/粗检测/230314-all/rough_waternet.engine", logger);
    cv::Mat image = cv::imread("D:/ImageAnnotation/chuanzha/VOCdevkit/VOCshipoutside/JPEGImages/00009.jpg");
    
    // 执行推理流程
    detector.preprocess(image);
    detector.infer();
    
    std::vector<Object> results;
    detector.postprocess(results);
    
    detector.draw(image, results);
    cv::imshow("Result", image);
    cv::waitKey(0);
    return 0;
}