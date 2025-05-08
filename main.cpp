#include "../include/yolov11.h"
int main(int argc, char** argv) {
    Logger logger;
    YOLOv11 detector("D:/ExperimentResult/yolo_deploy/yolo11n.engine", logger);
    cv::Mat image = cv::imread("C:/Users/Zhang/Desktop/Snipaste_2025-05-08_20-29-51.png");
    cv::imshow("Image", image);
    cv::waitKey(0);
    
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