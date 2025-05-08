from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
    model = YOLO("D:/ExperimentResult/yolo_deploy/yolo11n.pt")  # load an official model
    model.export(format='onnx')  # export the model to ONNX format